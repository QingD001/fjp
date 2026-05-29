import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class LinearSVMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=5000):
        self.C = C
        self.max_iter = max_iter
        self.model = Pipeline([
            ('scale', StandardScaler(with_mean=False)),
            ('clf', LinearSVC(
                random_state=42,
                class_weight='balanced',
                C=self.C,
                max_iter=self.max_iter
            ))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.named_steps['clf'].classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = np.clip(scores, -30, 30)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1 - p1, p1])


def train(X: pd.DataFrame, y: pd.Series) -> LinearSVMWrapper:
    model = LinearSVMWrapper()
    model.fit(X, y)
    return model


def evaluate(model: LinearSVMWrapper, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    from sklearn.metrics import roc_auc_score, average_precision_score
    return {
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'ap': average_precision_score(y, y_prob),
        'auc': roc_auc_score(y, y_prob)
    }
