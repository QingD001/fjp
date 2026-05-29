import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import pandas as pd
from FeatureEngineering import feature_engineering
from feature4cnn import add_cnn_text_col
from model import train as train_xgb, evaluate as eval_xgb
from model2 import train as train_lgb, evaluate as eval_lgb
from model3 import train as train_cat, evaluate as eval_cat
from model4 import train as train_rf, evaluate as eval_rf
from model5 import train as train_cnn, evaluate as eval_cnn
from model6 import train as train_lstm, evaluate as eval_lstm
from model7 import train as train_lr, evaluate as eval_lr
from model8 import train as train_svm, evaluate as eval_svm

TRAIN_PATH = 'data/job_postings_train.csv'
TEST_PATH = 'data/job_postings_test.csv'
SUB_PATH = 'data/job_postings_sample_submission.csv'
OUT_PATH = 'submission.csv'

MODEL = 'cnn'

MODELS = {
    'xgb':  (train_xgb,  eval_xgb,  'XGBoost'),
    'lgb':  (train_lgb,  eval_lgb,  'LightGBM'),
    'cat':  (train_cat,  eval_cat,  'CatBoost'),
    'rf':   (train_rf,   eval_rf,   'RandomForest'),
    'cnn':  (train_cnn,  eval_cnn,  'TextCNN'),
    'lstm': (train_lstm, eval_lstm, 'BiLSTM'),
    'lr':   (train_lr,   eval_lr,   'LogisticRegression'),
    'svm':  (train_svm,  eval_svm,  'LinearSVM'),
}

DL_MODELS = {'cnn', 'lstm'}

def add_text_col(df):
    return add_cnn_text_col(df)

if __name__ == "__main__":
    data = pd.read_csv(TRAIN_PATH)
    y = data['fraudulent']
    data = add_text_col(data)

    train_fn, eval_fn, name = MODELS[MODEL]

    if MODEL in DL_MODELS:
        model = train_fn(data, y)
        metrics = eval_fn(model, data, y)
        test = add_text_col(pd.read_csv(TEST_PATH))
        test_prob = model.predict_proba(test)[:, 1]
    else:
        X = feature_engineering(data, fit=True).fillna(0)
        model = train_fn(X, y)
        metrics = eval_fn(model, X, y)
        test = pd.read_csv(TEST_PATH)
        X_test = feature_engineering(test, fit=False).fillna(0)
        for c in X.columns:
            if c not in X_test.columns:
                X_test[c] = 0.0
        X_test = X_test[X.columns]
        test_prob = model.predict_proba(X_test)[:, 1]

    print(f"{name}方法：")
    print(f"精确率={metrics['precision']:.2%}")
    print(f"召回率={metrics['recall']:.2%}")
    print(f"AP={metrics['ap']:.4f}")
    print(f"F1={metrics['f1']:.2%}")
    print(f"AUC={metrics['auc']:.4f}")

    sub = pd.read_csv(SUB_PATH)
    sub['fraudulent'] = test_prob
    sub.to_csv(OUT_PATH, index=False)
    print(f"提交文件已保存至{OUT_PATH}，共{len(sub)}条预测")
