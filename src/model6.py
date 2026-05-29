import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from collections import Counter
from feature4cnn import build_cnn_texts

def _show_progress(label, current, total, loss=None, width=28):
    total = max(total, 1)
    current = min(current, total)
    ratio = current / total
    filled = int(width * ratio)
    bar = '#' * filled + '-' * (width - filled)
    spinner = '|/-\\'[current % 4]
    msg = f'\r{spinner} {label} [{bar}] {current}/{total} {ratio:6.2%}'
    if loss is not None:
        msg += f' loss={loss:.4f}'
    sys.stdout.write(msg)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write('\n')
        sys.stdout.flush()

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=512):
        self.texts = [self._encode(t, word2idx, max_len) for t in texts]
        self.labels = labels.values if hasattr(labels, 'values') else labels

    def _encode(self, text, word2idx, max_len):
        ids = [word2idx.get(w, 1) for w in str(text).lower().split()[:max_len]]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = torch.max(x, dim=1).values
        x = self.dropout(x)
        return self.fc(x).squeeze(1)

class LSTMWrapper:
    def __init__(self, model, word2idx, max_len=512, device='cuda'):
        self.model = model
        self.word2idx = word2idx
        self.max_len = max_len
        self.device = device

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)

    def predict_proba(self, X, desc='Predict'):
        texts = X['text']
        ds = TextDataset(texts, pd.Series([0]*len(texts)), self.word2idx, self.max_len)
        dl = DataLoader(ds, batch_size=128, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (x, _) in enumerate(dl, start=1):
                x = x.to(self.device)
                p = torch.sigmoid(self.model(x)).cpu().numpy()
                probs.append(p)
                _show_progress(desc, i, len(dl))
        p1 = np.concatenate(probs)
        return np.column_stack([1 - p1, p1])

def build_vocab(texts, min_freq=3):
    counter = Counter()
    for t in texts:
        counter.update(str(t).lower().split())
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for w, c in counter.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)
    return word2idx

def _join_texts(df):
    if 'text' in df.columns:
        return df['text'].fillna('')
    return build_cnn_texts(df)

def train(df: pd.DataFrame, y: pd.Series, epochs: int = 10, batch_size: int = 64) -> LSTMWrapper:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'BiLSTM train: rows={len(df)}, epochs={epochs}, batch_size={batch_size}, device={device}')
    texts = _join_texts(df)
    word2idx = build_vocab(texts)
    max_len = min(512, texts.str.split().str.len().max())
    print(f'BiLSTM vocab={len(word2idx)}, max_len={max_len}')

    ds = TextDataset(texts, y, word2idx, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = BiLSTM(len(word2idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, (x, labels) in enumerate(dl, start=1):
            x, labels = x.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            _show_progress(f'Train epoch {epoch}/{epochs}', step, len(dl), running_loss / step)

    return LSTMWrapper(model, word2idx, max_len, device)

def evaluate(wrapper: LSTMWrapper, df: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> dict:
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print(f'BiLSTM eval: train={len(train_df)}, val={len(val_df)}')
    wrapper2 = train(train_df, y_train, epochs=10)
    y_prob = wrapper2.predict_proba(val_df, desc='Eval predict')[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'ap': average_precision_score(y_val, y_prob),
        'auc': roc_auc_score(y_val, y_prob)
    }
