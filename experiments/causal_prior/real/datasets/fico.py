"""FICO HELOC loader (Explainable ML Challenge).

n=10459, 23 features (continuous credit scores/durations/ratios) and a binary
target. FICO's special codes (-7/-8/-9: no record / no usable trades / condition
not met) are treated as informative missingness when --sentinel-nan is set: each
affected feature is median-imputed and a `<feat>_missing` indicator is appended.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA = REPO_ROOT / 'data/real/raw/HelocData.csv'
TARGET = 'RiskFlag'
POS_LABEL = 'Bad'
SENTINELS = (-7, -8, -9)


def load_features(df, sentinel_nan):
    names = [c for c in df.columns if c != TARGET]
    raw = df[names].apply(pd.to_numeric, errors='coerce')
    if not sentinel_nan:
        return raw.fillna(0.0).to_numpy(float), names
    cols, out_names = [], []
    for name in names:
        s = raw[name].astype(float)
        miss = s.isin(SENTINELS)
        valid = s.mask(miss)
        cols.append(valid.fillna(valid.median()).fillna(0.0).to_numpy(float))
        out_names.append(name)
        if miss.any():
            cols.append(miss.astype(float).to_numpy())
            out_names.append(f'{name}_missing')
    return np.column_stack(cols), out_names


def load(args):
    if not DATA.exists():
        raise SystemExit(f'FICO CSV not found at {DATA}')
    df = pd.read_csv(DATA)
    y = np.where(df[TARGET].astype(str).str.strip() == POS_LABEL, 1, -1).astype(int)
    X_orig, names = load_features(df, getattr(args, 'sentinel_nan', True))
    return X_orig, np.asarray(names), y
