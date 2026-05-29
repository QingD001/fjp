import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def train(X: pd.DataFrame, y: pd.Series) -> CatBoostClassifier:
    model = CatBoostClassifier(
        random_seed=42,
        depth=7,
        learning_rate=0.2,
        iterations=200,
        verbose=0
    )
    model.fit(X, y)
    return model

def evaluate(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> dict:
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
