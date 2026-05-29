import numpy as np
import pandas as pd
from typing import Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def tofloat(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None

def text_features(text) -> dict:
    res = {}
    if pd.isna(text) or str(text).strip() == '':
        res['char_count'] = res['word_count'] = res['avg_word_len'] = res['url_placeholder'] = res['email_placeholder'] = res['html'] = 0
        return res
    t = str(text)
    res['char_count'] = len(t)
    words = t.split()
    res['word_count'] = len(words)
    res['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0.0
    res['url_placeholder'] = 1 if '#URL_' in t else 0
    res['email_placeholder'] = 1 if '#EMAIL_' in t else 0
    res['html'] = 1 if ('<' in t and '>' in t) else 0
    return res

def parse_salary(v) -> dict:
    res = {'salary_min': 0.0, 'salary_max': 0.0, 'salary_avg': 0.0, 'has_salary': 0}
    if pd.isna(v) or str(v).strip() == '':
        return res
    nums = re.findall(r'\d+', str(v))
    if len(nums) >= 2:
        res['salary_min'] = tofloat(nums[0]) or 0.0
        res['salary_max'] = tofloat(nums[1]) or 0.0
        res['salary_avg'] = (res['salary_min'] + res['salary_max']) / 2.0
        res['has_salary'] = 1
    return res

def get_features(row) -> pd.DataFrame:
    res = {}
    for col, _ in [('company_profile', 'cp'), ('description', 'desc'), ('requirements', 'req'), ('benefits', 'ben')]:
        feats = text_features(row[col])
        for k, v in feats.items():
            res[f'{_}_{k}'] = v

    for col in ['telecommuting', 'has_company_logo', 'has_questions']:
        res[col] = tofloat(row[col]) or 0.0
    sal = parse_salary(row['salary_range'])
    res.update(sal)
    return pd.DataFrame([res])

_tfidf = None

def feature_engineering(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    global _tfidf

    frames = [get_features(row) for _, row in df.iterrows()]
    result = pd.concat(frames, ignore_index=True)

    cols = ['employment_type', 'required_experience', 'required_education']
    for col in cols:
        dummies = pd.get_dummies(df[col].fillna('Missing'), prefix=col)
        result = pd.concat([result, dummies.reset_index(drop=True)], axis=1)

    texts = df[['company_profile', 'description', 'requirements', 'benefits']].fillna('').apply(
        lambda r: ' '.join(r), axis=1
    )

    if fit:
        _tfidf = TfidfVectorizer(
            max_features=800,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            strip_accents='unicode'
        )
        tfidf = _tfidf.fit_transform(texts)
    else:
        assert _tfidf is not None
        tfidf = _tfidf.transform(texts)

    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf.shape[1])])
    result = pd.concat([result, tfidf_df.reset_index(drop=True)], axis=1)

    return result
