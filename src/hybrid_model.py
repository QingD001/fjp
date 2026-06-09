import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


TEXT_COLS = ['title', 'company_profile', 'description', 'requirements', 'benefits']
CAT_COLS = [
    'employment_type',
    'required_experience',
    'required_education',
    'industry',
    'function',
    'department',
]
NUM_COLS = ['telecommuting', 'has_company_logo', 'has_questions']


def _text_column(data):
    if isinstance(data, pd.DataFrame):
        return data['text'].fillna('')
    return pd.Series(data).fillna('')


def _raw_text(data):
    if not isinstance(data, pd.DataFrame):
        return pd.Series(data).fillna('')
    cols = [col for col in TEXT_COLS if col in data.columns]
    if not cols:
        return _text_column(data)
    return data[cols].fillna('').agg(' '.join, axis=1)


def _title_text(data):
    if isinstance(data, pd.DataFrame) and 'title' in data.columns:
        return data['title'].fillna('')
    return _text_column(data)


def _category_text(data):
    if not isinstance(data, pd.DataFrame):
        return pd.Series(['missing'] * len(data))
    cols = [col for col in CAT_COLS + ['location', 'salary_range'] if col in data.columns]
    if not cols:
        return pd.Series(['missing'] * len(data), index=data.index)
    return data[cols].fillna('missing').astype(str).agg(' '.join, axis=1)


def _numeric_block(data):
    if not isinstance(data, pd.DataFrame):
        return np.zeros((len(data), 1), dtype=np.float32)

    frame = pd.DataFrame(index=data.index)
    for col in TEXT_COLS:
        text = data[col].fillna('').astype(str) if col in data.columns else pd.Series('', index=data.index)
        stripped = text.str.strip()
        frame[f'{col}_missing'] = (stripped == '').astype(float)
        frame[f'{col}_len'] = text.str.len().astype(float)
        frame[f'{col}_words'] = text.str.split().str.len().fillna(0).astype(float)
        lower = text.str.lower()
        frame[f'{col}_email'] = (text.str.contains('@', regex=False) | text.str.contains('#EMAIL_', regex=False)).astype(float)
        frame[f'{col}_url'] = (
            lower.str.contains('http', regex=False)
            | lower.str.contains('www.', regex=False)
            | text.str.contains('#URL_', regex=False)
        ).astype(float)
        frame[f'{col}_html'] = (text.str.contains('<', regex=False) & text.str.contains('>', regex=False)).astype(float)

    for col in NUM_COLS:
        if col in data.columns:
            frame[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)
        else:
            frame[col] = 0.0

    return frame.fillna(0.0)


def build_sparse_base_model(kind='lr'):
    features = FeatureUnion([
        (
            'engineered_word',
            Pipeline([
                ('select', FunctionTransformer(_text_column, validate=False)),
                ('tfidf', TfidfVectorizer(
                    lowercase=True,
                    strip_accents='unicode',
                    sublinear_tf=True,
                    min_df=2,
                    max_df=0.995,
                    max_features=35000,
                    ngram_range=(1, 2),
                )),
            ]),
        ),
        (
            'engineered_char',
            Pipeline([
                ('select', FunctionTransformer(_text_column, validate=False)),
                ('tfidf', TfidfVectorizer(
                    analyzer='char_wb',
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=2,
                    max_features=35000,
                    ngram_range=(3, 6),
                )),
            ]),
        ),
        (
            'raw_word',
            Pipeline([
                ('select', FunctionTransformer(_raw_text, validate=False)),
                ('tfidf', TfidfVectorizer(
                    lowercase=True,
                    strip_accents='unicode',
                    sublinear_tf=True,
                    min_df=2,
                    max_features=20000,
                    ngram_range=(1, 2),
                )),
            ]),
        ),
        (
            'title_word',
            Pipeline([
                ('select', FunctionTransformer(_title_text, validate=False)),
                ('tfidf', TfidfVectorizer(
                    lowercase=True,
                    strip_accents='unicode',
                    sublinear_tf=True,
                    min_df=2,
                    max_features=4000,
                    ngram_range=(1, 3),
                )),
            ]),
        ),
        (
            'category_word',
            Pipeline([
                ('select', FunctionTransformer(_category_text, validate=False)),
                ('tfidf', TfidfVectorizer(
                    lowercase=True,
                    strip_accents='unicode',
                    sublinear_tf=True,
                    min_df=1,
                    max_features=8000,
                    ngram_range=(1, 2),
                )),
            ]),
        ),
        (
            'numeric',
            Pipeline([
                ('select', FunctionTransformer(_numeric_block, validate=False)),
                ('scale', StandardScaler(with_mean=False)),
            ]),
        ),
    ])

    if kind == 'svm':
        classifier = LinearSVC(
            C=0.25,
            class_weight='balanced',
            max_iter=8000,
            random_state=42,
        )
    else:
        classifier = LogisticRegression(
            C=1.25,
            class_weight='balanced',
            max_iter=4000,
            solver='liblinear',
            random_state=42,
        )

    return Pipeline([
        ('features', features),
        ('classifier', classifier),
    ])


def _decision_to_unit(scores):
    scores = np.clip(scores, -30, 30)
    return 1.0 / (1.0 + np.exp(-scores))


class SparseBlendModel:
    def __init__(self, lr_weight=0.65):
        self.lr_weight = lr_weight
        self.lr_model = build_sparse_base_model('lr')
        self.svm_model = build_sparse_base_model('svm')

    def fit(self, data, y):
        self.lr_model.fit(data, y)
        self.svm_model.fit(data, y)
        return self

    def predict_proba(self, data):
        lr_prob = self.lr_model.predict_proba(data)[:, 1]
        svm_prob = _decision_to_unit(self.svm_model.decision_function(data))
        p1 = self.lr_weight * rank_normalize(lr_prob) + (1.0 - self.lr_weight) * rank_normalize(svm_prob)
        return np.column_stack([1 - p1, p1])


def build_sparse_model():
    return SparseBlendModel()


def train_sparse(data, y):
    model = build_sparse_model()
    model.fit(data, y)
    return model


def predict_sparse(model, data):
    return model.predict_proba(data)[:, 1]


def rank_normalize(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) > 1:
        ranks /= len(values) - 1
    return ranks


def blend_predictions(cnn_prob, sparse_prob, cnn_weight):
    cnn_rank = rank_normalize(cnn_prob)
    sparse_rank = rank_normalize(sparse_prob)
    return cnn_weight * cnn_rank + (1.0 - cnn_weight) * sparse_rank


def find_best_weight(y_true, cnn_prob, sparse_prob):
    best_weight = 0.0
    best_ap = -np.inf

    for weight in np.linspace(0.0, 1.0, 101):
        blended = blend_predictions(cnn_prob, sparse_prob, weight)
        score = average_precision_score(y_true, blended)
        if score > best_ap:
            best_ap = score
            best_weight = float(weight)

    return best_weight, best_ap


def classification_metrics(y_true, probability):
    thresholds = np.linspace(0.05, 0.95, 181)
    f1_scores = [f1_score(y_true, probability >= value) for value in thresholds]
    threshold = float(thresholds[int(np.argmax(f1_scores))])
    prediction = (probability >= threshold).astype(int)

    return {
        'precision': precision_score(y_true, prediction),
        'recall': recall_score(y_true, prediction),
        'f1': f1_score(y_true, prediction),
        'ap': average_precision_score(y_true, probability),
        'auc': roc_auc_score(y_true, probability),
        'threshold': threshold,
    }
