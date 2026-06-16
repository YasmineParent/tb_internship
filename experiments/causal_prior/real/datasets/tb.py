"""TB-DLM loader (delamanid resistance), prevalence-filtered regime.

n~152, binary mutation features (DLM-pathway genes, prevalence in [0.05, 0.98])
plus lineage indicators; target = agar R/S call. Features are already binary so
the binarizer passes them through.

note: this exposes TB to the GENERIC runners, which discover q themselves (e.g.
ges_cg). That is a complementary experiment, NOT the §6.3 result, which uses the
domain CMM stability q (the bespoke tb_dlm_*.py scripts). Use those for §6.3.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA = REPO_ROOT / 'data/real/processed/tb_pheno_geno_clean.csv'
TARGET = 'interp_dlm016'
DLM_GENES = ('rv0678', 'mmpl5', 'mmps5', 'atpe', 'pepq', 'rv1979c', 'fgd1', 'ddn')
PREV_LO, PREV_HI = 0.05, 0.98


def load(args=None):
    if not DATA.exists():
        raise SystemExit(f'TB data not found at {DATA}')
    df = pd.read_csv(DATA)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = np.where(df[TARGET].to_numpy() == 2.0, 1, -1).astype(int)

    mut_cols = [c for c in df.columns if any(c.startswith(g + '_') for g in DLM_GENES)]
    M = df[mut_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0)
    prev = M.mean()
    feasible = sorted(prev[(prev >= PREV_LO) & (prev <= PREV_HI)].index)

    lineage = df['lineage'].astype('Int64')
    lin = {f'lineage_{lv}': (lineage == lv).astype(float).to_numpy() for lv in (2, 4)}
    names = feasible + list(lin)
    X = np.column_stack([M[feasible].to_numpy()] + [lin[c] for c in lin])
    return X, np.asarray(names), y
