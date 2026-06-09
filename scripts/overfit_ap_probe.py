import os
import sys

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC


TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "department",
]
NUM_COLS = ["telecommuting", "has_company_logo", "has_questions"]


def basic_text(df):
    return df[TEXT_COLS].fillna("").agg(" ".join, axis=1)


def title_text(df):
    return df["title"].fillna("")


def cat_text(df):
    return df[CAT_COLS + ["location", "salary_range"]].fillna("missing").agg(" ".join, axis=1)


def numeric_block(df):
    rows = []
    for _, row in df.iterrows():
        out = {}
        for col in TEXT_COLS:
            text = "" if pd.isna(row.get(col)) else str(row.get(col))
            out[f"{col}_missing"] = float(text.strip() == "")
            out[f"{col}_len"] = float(len(text))
            out[f"{col}_words"] = float(len(text.split()))
            out[f"{col}_email"] = float("#EMAIL_" in text or "@" in text)
            out[f"{col}_url"] = float("#URL_" in text or "http" in text.lower() or "www." in text.lower())
            out[f"{col}_html"] = float("<" in text and ">" in text)
        for col in NUM_COLS:
            out[col] = float(row.get(col) or 0)
        rows.append(out)
    return pd.DataFrame(rows).fillna(0.0)


def build_sparse_text_model(kind="lr"):
    union = FeatureUnion(
        [
            (
                "word",
                Pipeline(
                    [
                        ("select", FunctionTransformer(basic_text, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=8000,
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "char",
                Pipeline(
                    [
                        ("select", FunctionTransformer(basic_text, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                max_features=8000,
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "title",
                Pipeline(
                    [
                        ("select", FunctionTransformer(title_text, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=1500,
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "cat_text",
                Pipeline(
                    [
                        ("select", FunctionTransformer(cat_text, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=1500,
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "num",
                Pipeline(
                    [
                        ("select", FunctionTransformer(numeric_block, validate=False)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
        ]
    )
    if kind == "svm":
        clf = LinearSVC(C=0.25, class_weight="balanced", max_iter=8000, random_state=42)
    else:
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            solver="liblinear",
            random_state=42,
        )
    return Pipeline([("features", union), ("clf", clf)])


def decision_to_unit(scores):
    scores = np.clip(scores, -30, 30)
    return 1.0 / (1.0 + np.exp(-scores))


def oof_score(name, estimator, X, y, predict_kind="proba", folds=3):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)
    train_scores = []
    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X.iloc[tr], y.iloc[tr])
        if predict_kind == "decision":
            val_prob = decision_to_unit(model.decision_function(X.iloc[va]))
            tr_prob = decision_to_unit(model.decision_function(X.iloc[tr]))
        else:
            val_prob = model.predict_proba(X.iloc[va])[:, 1]
            tr_prob = model.predict_proba(X.iloc[tr])[:, 1]
        oof[va] = val_prob
        train_scores.append(average_precision_score(y.iloc[tr], tr_prob))
        print(
            f"{name} fold{fold}: train_ap={train_scores[-1]:.4f} "
            f"val_ap={average_precision_score(y.iloc[va], val_prob):.4f}",
            flush=True,
        )
    print(
        f"{name} OOF: ap={average_precision_score(y, oof):.4f} "
        f"auc={roc_auc_score(y, oof):.4f} train_ap_mean={np.mean(train_scores):.4f}",
        flush=True,
    )
    return oof


def main():
    data = pd.read_csv(os.path.join(ROOT, "data", "job_postings_train.csv"))
    y = data["fraudulent"].astype(int)
    X_raw = data.drop(columns=["fraudulent"])
    print(f"rows={len(data)} positives={int(y.sum())} pos_rate={y.mean():.4%}", flush=True)

    lr_oof = oof_score("sparse_lr", build_sparse_text_model("lr"), X_raw, y)
    svm_oof = oof_score("sparse_svm", build_sparse_text_model("svm"), X_raw, y, "decision")
    blend = 0.65 * lr_oof + 0.35 * svm_oof
    print(
        f"sparse_blend OOF: ap={average_precision_score(y, blend):.4f} "
        f"auc={roc_auc_score(y, blend):.4f}",
        flush=True,
    )

if __name__ == "__main__":
    main()
