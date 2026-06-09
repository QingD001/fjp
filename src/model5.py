import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from collections import Counter
from copy import deepcopy
from feature4cnn import build_cnn_texts

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    from tensorflow.keras.utils import Progbar
except ImportError:
    from keras.utils import Progbar

DEFAULT_SEED = 42


def _seed_everything(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=512):
        self.texts = torch.stack([self._encode(t, word2idx, max_len) for t in texts])
        values = labels.values if hasattr(labels, 'values') else labels
        self.labels = torch.tensor(np.asarray(values).copy(), dtype=torch.float32)

    def _encode(self, text, word2idx, max_len):
        ids = [word2idx.get(w, 1) for w in str(text).lower().split()[:max_len]]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=192,
        num_filters=128,
        kernel_sizes=(2, 3, 4, 5),
        embedding_dropout=0.15,
        dropout=0.4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout1d(embedding_dropout)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.embedding_dropout(x)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)

class CNNWrapper:
    def __init__(self, model, word2idx, max_len=512, device='cuda', threshold=0.5):
        self.model = model
        self.word2idx = word2idx
        self.max_len = max_len
        self.device = device
        self.threshold = threshold

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= self.threshold).astype(int)

    def predict_proba(self, X, desc='Predict'):
        texts = X['text']
        ds = TextDataset(texts, pd.Series([0]*len(texts)), self.word2idx, self.max_len)
        dl = DataLoader(
            ds,
            batch_size=128,
            shuffle=False,
            pin_memory=self.device == 'cuda',
        )
        self.model.eval()
        probs = []
        print(desc)
        progress = Progbar(len(dl))
        with torch.no_grad():
            for i, (x, _) in enumerate(dl, start=1):
                x = x.to(self.device)
                p = torch.sigmoid(self.model(x)).cpu().numpy()
                probs.append(p)
                progress.update(i)
        p1 = np.concatenate(probs)
        return np.column_stack([1 - p1, p1])

def build_vocab(texts, min_freq=2, max_vocab=100000):
    counter = Counter()
    for t in texts:
        counter.update(str(t).lower().split())
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in counter.most_common(max_vocab - len(word2idx)):
        if count < min_freq:
            break
        word2idx[word] = len(word2idx)
    return word2idx

def _join_texts(df):
    if 'text' in df.columns:
        return df['text'].fillna('')
    return build_cnn_texts(df)

def _predict_model(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            probs.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(probs)


def _best_f1_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, y_prob >= threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _ema_update(ema_state, model, decay):
    current = model.state_dict()
    if ema_state is None:
        return {name: value.detach().clone() for name, value in current.items()}
    for name, value in current.items():
        if value.is_floating_point():
            ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[name].copy_(value)
    return ema_state


def train(
    df: pd.DataFrame,
    y: pd.Series,
    epochs: int = 12,
    batch_size: int = 64,
    seed: int = DEFAULT_SEED,
    val_df: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    patience: int = 3,
) -> CNNWrapper:
    _seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'TextCNN train: rows={len(df)}, epochs={epochs}, batch_size={batch_size}, device={device}')
    texts = _join_texts(df)
    word2idx = build_vocab(texts)
    lengths = texts.str.split().str.len()
    max_len = max(64, min(768, int(np.ceil(lengths.quantile(0.95)))))
    print(f'TextCNN vocab={len(word2idx)}, max_len={max_len}')

    ds = TextDataset(texts, y, word2idx, max_len)
    generator = torch.Generator().manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device == 'cuda',
    )

    val_loader = None
    if val_df is not None and y_val is not None:
        val_texts = _join_texts(val_df)
        val_ds = TextDataset(val_texts, y_val, word2idx, max_len)
        val_loader = DataLoader(
            val_ds,
            batch_size=128,
            shuffle=False,
            pin_memory=device == 'cuda',
        )

    model = TextCNN(len(word2idx)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    ema_state = None
    best_state = None
    best_ap = -np.inf
    best_threshold = 0.5
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        print(f'Epoch {epoch}/{epochs}')
        progress = Progbar(len(dl))
        for step, (x, labels) in enumerate(dl, start=1):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running_loss += loss.item()
            progress.update(
                step,
                values=[('loss', running_loss / step)],
            )
        scheduler.step()

        if epoch >= max(3, epochs // 3):
            ema_state = _ema_update(ema_state, model, decay=0.9)

        if val_loader is not None:
            candidate = deepcopy(model.state_dict())
            if ema_state is not None:
                model.load_state_dict(ema_state)
            val_prob = _predict_model(model, val_loader, device)
            val_ap = average_precision_score(y_val, val_prob)
            threshold = _best_f1_threshold(y_val, val_prob)
            print(f'Validation epoch {epoch}: AP={val_ap:.4f}, threshold={threshold:.3f}')
            if val_ap > best_ap + 1e-4:
                best_ap = val_ap
                best_state = deepcopy(model.state_dict())
                best_threshold = threshold
                stale_epochs = 0
            else:
                stale_epochs += 1
            model.load_state_dict(candidate)
            if stale_epochs >= patience:
                print(f'Early stopping at epoch {epoch}; best AP={best_ap:.4f}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif ema_state is not None:
        model.load_state_dict(ema_state)

    return CNNWrapper(model, word2idx, max_len, device, best_threshold)

def evaluate(wrapper: CNNWrapper, df: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> dict:
    # Simple train/val split evaluation for speed
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print(f'TextCNN eval: train={len(train_df)}, val={len(val_df)}')
    wrapper2 = train(
        train_df,
        y_train,
        epochs=15,
        val_df=val_df,
        y_val=y_val,
    )
    y_prob = wrapper2.predict_proba(val_df, desc='Eval predict')[:, 1]
    y_pred = (y_prob >= wrapper2.threshold).astype(int)

    return {
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'ap': average_precision_score(y_val, y_prob),
        'auc': roc_auc_score(y_val, y_prob)
    }
