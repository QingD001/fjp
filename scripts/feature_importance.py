import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from FeatureEngineering import feature_engineering
from model import train

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'job_postings_train.csv')

NEW_FEATURES = {
    'cp_missing', 'cp_excl_density', 'cp_sentence_count', 'cp_has_inc',
    'desc_link_count', 'desc_digit_ratio', 'desc_has_bullet',
    'department_missing', 'salary_ratio', 'is_us', 'title_fraud_score',
    'logo_and_questions', 'country_rate',
}

def main():
    data = pd.read_csv(DATA_PATH)
    y = data['fraudulent']
    X = feature_engineering(data, fit=True, y=y)
    X = X.fillna(0)

    model = train(X, y)

    imp = model.get_booster().get_score(importance_type='gain')
    df_gain = pd.DataFrame({'feature': list(imp.keys()), 'importance': list(imp.values())})
    df_gain = df_gain.sort_values('importance', ascending=False).reset_index(drop=True)

    print('=== XGBoost Gain 重要性 Top 50 ===')
    for i, row in df_gain.head(50).iterrows():
        tag = ' [新]' if row['feature'] in NEW_FEATURES else ''
        print(f'{i+1:>3}. {row["feature"]:<40s} {row["importance"]:>10.2f}{tag}')

if __name__ == '__main__':
    main()
