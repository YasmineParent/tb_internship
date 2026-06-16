"""Mammographic Mass loader (UCI), a mixed clinical interpretable-scoring benchmark.

~830 rows after dropping missing. features: age (continuous), density (ordinal),
shape and margin (nominal, one-hot). target: severity (malignant vs benign).

we exclude the BI-RADS attribute on purpose: it is the radiologist's own
malignancy assessment, so using it as a feature leaks the target. the remaining
features are objective lesion attributes, the honest predictive set.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE = REPO_ROOT / 'data/real/raw/mammographic_masses.data'
URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
       'mammographic-masses/mammographic_masses.data')
CONT = ['age', 'density']          # density is ordinal 1-4, kept numeric
CAT = ['shape', 'margin']          # nominal, one-hot


def load(args=None):
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    cols = ['birads', 'age', 'shape', 'margin', 'density', 'severity']
    df = pd.read_csv(CACHE, header=None, names=cols, na_values='?')
    df = df.dropna(subset=['age', 'shape', 'margin', 'density', 'severity']).reset_index(drop=True)
    y = np.where(df['severity'].to_numpy() == 1, 1, -1).astype(int)
    out, names = [], []
    for c in CONT:
        out.append(df[c].to_numpy(float)); names.append(c)
    for c in CAT:
        for lv in sorted(df[c].dropna().unique()):
            out.append((df[c] == lv).to_numpy(float)); names.append(f'{c}={int(lv)}')
    return np.column_stack(out), np.asarray(names), y
