import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from feature4cnn import add_cnn_text_col
from hybrid_model import (
    blend_predictions,
    classification_metrics,
    find_best_weight,
    predict_sparse,
    train_sparse,
)
from model5 import train as train_textcnn


TRAIN_PATH = 'data/job_postings_train.csv'
TEST_PATH = 'data/job_postings_test.csv'
SUB_PATH = 'data/job_postings_sample_submission.csv'
OUT_PATH = 'submission.csv'
RANDOM_STATE = 42


def print_metrics(name, metrics):
    print(f'\n{name}:')
    print(f"Precision={metrics['precision']:.2%}")
    print(f"Recall={metrics['recall']:.2%}")
    print(f"AP={metrics['ap']:.4f}")
    print(f"F1={metrics['f1']:.2%}")
    print(f"AUC={metrics['auc']:.4f}")


if __name__ == '__main__':
    raw_train = pd.read_csv(TRAIN_PATH)
    train_data = add_cnn_text_col(raw_train)
    y = raw_train['fraudulent'].astype(int)

    indices = np.arange(len(train_data))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    fit_data = train_data.iloc[train_idx].copy()
    val_data = train_data.iloc[val_idx].copy()
    y_fit = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print('Stage 1: find the best validation blend weight')
    cnn_val_model = train_textcnn(
        fit_data,
        y_fit,
        epochs=15,
        val_df=val_data,
        y_val=y_val,
    )
    cnn_val_prob = cnn_val_model.predict_proba(
        val_data,
        desc='CNN validation',
    )[:, 1]

    sparse_val_model = train_sparse(fit_data, y_fit)
    sparse_val_prob = predict_sparse(sparse_val_model, val_data)

    cnn_ap = average_precision_score(y_val, cnn_val_prob)
    sparse_ap = average_precision_score(y_val, sparse_val_prob)
    cnn_weight, blend_ap = find_best_weight(
        y_val,
        cnn_val_prob,
        sparse_val_prob,
    )
    blend_val_prob = blend_predictions(
        cnn_val_prob,
        sparse_val_prob,
        cnn_weight,
    )
    metrics = classification_metrics(y_val, blend_val_prob)

    print(f'CNN validation AP={cnn_ap:.4f}')
    print(f'TF-IDF validation AP={sparse_ap:.4f}')
    print(
        f'Best blend: CNN={cnn_weight:.3f}, '
        f'TF-IDF={1.0 - cnn_weight:.3f}, AP={blend_ap:.4f}'
    )
    print_metrics('Hybrid validation', metrics)

    print('\nStage 2: retrain both branches on all training data')
    final_cnn = train_textcnn(train_data, y)
    final_sparse = train_sparse(train_data, y)

    test_data = add_cnn_text_col(pd.read_csv(TEST_PATH))
    cnn_test_prob = final_cnn.predict_proba(
        test_data,
        desc='CNN test',
    )[:, 1]
    sparse_test_prob = predict_sparse(final_sparse, test_data)
    test_prob = blend_predictions(
        cnn_test_prob,
        sparse_test_prob,
        cnn_weight,
    )

    submission = pd.read_csv(SUB_PATH)
    submission['fraudulent'] = test_prob
    submission.to_csv(OUT_PATH, index=False)
    print(f'Saved {len(submission)} predictions to {OUT_PATH}.')
