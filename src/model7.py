import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline([
        ('scale', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=3000,
            solver='liblinear'
        ))
    ])
    model.fit(X, y)
    return model


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> dict:
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
