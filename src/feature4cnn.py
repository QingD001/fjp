import re

import numpy as np
import pandas as pd


# Put the job-specific fields first. TextCNN truncates long documents, so a long
# generic company profile should not crowd out the description and requirements.
TEXT_COLS = ['title', 'description', 'requirements', 'benefits', 'company_profile']
CATEGORY_COLS = [
    'employment_type',
    'required_experience',
    'required_education',
    'industry',
    'function',
    'department',
]

RISK_PHRASES = {
    'work_from_home': ['work from home', 'work at home', 'remote job'],
    'urgent': ['urgent', 'immediate start', 'start immediately'],
    'easy_money': ['easy money', 'earn money', 'extra income'],
    'data_entry': ['data entry'],
    'no_experience': ['no experience', 'no experience required'],
    'assistant': ['assistant', 'personal assistant', 'administrative assistant'],
    'wire_transfer': ['wire transfer', 'western union', 'moneygram'],
    'commission': ['commission', 'uncapped commission'],
}


def _safe_text(value) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip()


def _token(value) -> str:
    text = _safe_text(value).lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'missing'


def _clean_free_text(value) -> str:
    text = _safe_text(value).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' url_token ', text)
    text = re.sub(r'\S+@\S+', ' email_token ', text)
    text = re.sub(r'<[^>]+>', ' html_token ', text)
    text = re.sub(r'[^a-z0-9#@_.$%+-]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _text_stats_tokens(name: str, value) -> list[str]:
    text = _safe_text(value)
    if not text:
        return [f'{name}_missing']

    words = text.split()
    chars = len(text)
    upper = sum(1 for ch in text if ch.isupper())
    digits = sum(1 for ch in text if ch.isdigit())
    exclaims = text.count('!')

    tokens = [f'{name}_present']
    if len(words) < 20:
        tokens.append(f'{name}_very_short')
    elif len(words) > 250:
        tokens.append(f'{name}_very_long')

    if chars:
        if upper / chars > 0.12:
            tokens.append(f'{name}_high_upper')
        if digits / chars > 0.08:
            tokens.append(f'{name}_high_digit')
    if exclaims >= 2:
        tokens.append(f'{name}_many_exclaim')
    if '#URL_' in text or 'http' in text.lower() or 'www.' in text.lower():
        tokens.append(f'{name}_has_url')
    if '#EMAIL_' in text or '@' in text:
        tokens.append(f'{name}_has_email')
    if '<' in text and '>' in text:
        tokens.append(f'{name}_has_html')

    return tokens


def _salary_tokens(value) -> list[str]:
    text = _safe_text(value)
    if not text:
        return ['salary_missing']

    nums = [float(x) for x in re.findall(r'\d+', text)]
    tokens = ['salary_present']
    if len(nums) >= 2:
        lo, hi = nums[0], nums[1]
        avg = (lo + hi) / 2.0
        width = hi - lo
        tokens.append('salary_zero_zero' if lo == 0 and hi == 0 else 'salary_range')
        if avg >= 100000:
            tokens.append('salary_high')
        elif avg <= 1000:
            tokens.append('salary_low')
        if width > max(avg, 1) * 1.5:
            tokens.append('salary_wide')
    return tokens


def _location_tokens(value) -> list[str]:
    text = _safe_text(value)
    if not text:
        return ['location_missing']

    parts = [p.strip() for p in text.split(',')]
    country = _token(parts[0]) if len(parts) > 0 else 'missing'
    state = _token(parts[1]) if len(parts) > 1 else 'missing'
    city = _token(parts[2]) if len(parts) > 2 else 'missing'
    tokens = [f'country_{country}', f'state_{state}', f'city_{city}']
    if country == 'us':
        tokens.append('is_us')
    elif country != 'missing':
        tokens.append('is_non_us')
    return tokens


def _binary_tokens(row) -> list[str]:
    tokens = []
    for col in ['telecommuting', 'has_company_logo', 'has_questions']:
        value = _safe_text(row.get(col, '0'))
        state = 'yes' if value in {'1', '1.0', 'true', 'True'} else 'no'
        tokens.append(f'{col}_{state}')

    if _safe_text(row.get('has_company_logo')) in {'0', '0.0', ''} and _safe_text(row.get('has_questions')) in {'0', '0.0', ''}:
        tokens.append('no_logo_no_questions')
    if _safe_text(row.get('telecommuting')) in {'1', '1.0', 'true', 'True'}:
        tokens.append('remote_flag')
    return tokens


def _risk_tokens(row) -> list[str]:
    joined = ' '.join(_safe_text(row.get(col)) for col in TEXT_COLS).lower()
    tokens = []
    for name, phrases in RISK_PHRASES.items():
        if any(phrase in joined for phrase in phrases):
            tokens.append(f'risk_{name}')
    return tokens


def build_cnn_text(row: pd.Series) -> str:
    parts = []

    parts.extend(_binary_tokens(row))
    parts.extend(_salary_tokens(row.get('salary_range')))
    parts.extend(_location_tokens(row.get('location')))
    parts.extend(_risk_tokens(row))

    missing_count = 0
    for col in TEXT_COLS + CATEGORY_COLS + ['salary_range', 'location']:
        if not _safe_text(row.get(col)):
            missing_count += 1
            parts.append(f'{col}_missing')
    parts.append(f'missing_count_{min(missing_count, 8)}')

    for col in CATEGORY_COLS:
        parts.append(f'{col}_{_token(row.get(col))}')

    for col in TEXT_COLS:
        parts.extend(_text_stats_tokens(col, row.get(col)))
        text = _clean_free_text(row.get(col))
        if text:
            parts.append(f'field_{col}')
            parts.append(text)

    return ' '.join(parts)


def build_cnn_texts(df: pd.DataFrame) -> pd.Series:
    return df.apply(build_cnn_text, axis=1)


def add_cnn_text_col(df: pd.DataFrame, col: str = 'text') -> pd.DataFrame:
    df = df.copy()
    df[col] = build_cnn_texts(df)
    return df
