"""§6.3 robustness: is the vanilla->causal support-stability lift on TB-DLM real?

Loops the tb_dlm_firstpass comparison over seeds (CV fold splits), two agar
thresholds (interp_dlm016, interp_dlm06), and several K, reporting whether the
stability(causal) >= stability(vanilla) result holds or was a lucky split.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402

DATA = ROOT / 'data/real/processed/tb_pheno_geno_clean.csv'
QCSV = ROOT / ('results/mixed_cmm/subsampling/tb_subsampling_dlm_mp4_k6_mcc4/'
               'with_lineage_forbid_to_mic/edge_stability.csv')
MIC_NODE = 'dlm_mic'
DLM_GENES = ('rv0678', 'mmpl5', 'mmps5', 'atpe', 'pepq', 'rv1979c', 'fgd1', 'ddn')
PREV_LO, PREV_HI = 0.05, 0.98
N_SPLITS = 10
SEEDS = range(6)  # robustness sanity check; enough to tell a lucky split from a real effect
TARGETS = ('interp_dlm016', 'interp_dlm06')
KS = (3, 5, 7)
MU_REL_GRID = np.array([0.0, 0.5, 1.5, 5.0])  # coarse: CV can still pick 0 (vanilla) or a nonzero mu


def build_xyq(df_full: pd.DataFrame, target_col: str):
    d = df_full[df_full[target_col].notna()].reset_index(drop=True)
    y = np.where(d[target_col].to_numpy() == 2.0, 1, -1).astype(int)
    mut_cols = [c for c in d.columns if any(c.startswith(g + '_') for g in DLM_GENES)]
    M = d[mut_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0)
    prev = M.mean()
    feasible = sorted(prev[(prev >= PREV_LO) & (prev <= PREV_HI)].index)
    lin = {f'lineage_{lv}': (d['lineage'].astype('Int64') == lv).astype(float).to_numpy()
           for lv in (2, 4)}
    feat_names = feasible + list(lin)
    X = np.column_stack([M[feasible].to_numpy()] + [lin[c] for c in lin])
    edges = pd.read_csv(QCSV)
    into_mic = edges[edges['target'] == MIC_NODE].set_index('source')['frequency']
    q = np.array([float(into_mic.get(n, 0.0)) for n in feat_names])
    return X, y, q, feat_names


def run(X, y, q, K, seed):
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_grid = MU_REL_GRID * mu_scale
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = cv_pick_mu(X, y, K=K, mu_grid=np.array([0.0]), q=None,
                         n_splits=N_SPLITS, rng=np.random.default_rng(seed))
        cau = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=q,
                         n_splits=N_SPLITS, rng=np.random.default_rng(seed))
    return dict(mu_hat_rel=cau.mu_star / mu_scale if mu_scale else 0.0,
                stab_van=van.stability_star, stab_cau=cau.stability_star,
                auc_van=van.auc_star, auc_cau=cau.auc_star,
                sup_van=tuple(van.support), sup_cau=tuple(cau.support))


def main() -> None:
    df = pd.read_csv(DATA)
    print(f'{"target":14s} {"K":>2s} {"n":>4s} {"pos":>4s} | '
          f'{"stab_van":>8s} {"stab_cau":>8s} {"d_stab":>7s} {"cau>=van":>8s} | '
          f'{"mu_hat":>7s} {"mu>0":>5s} | {"auc_v":>5s} {"auc_c":>5s} | {"pepq":>5s}', flush=True)
    for target in TARGETS:
        X, y, q, feat_names = build_xyq(df, target)
        pepq_i = feat_names.index('pepq_Ala87Gly') if 'pepq_Ala87Gly' in feat_names else -1
        for K in KS:
            rs = [run(X, y, q, K, s) for s in SEEDS]
            sv = np.array([r['stab_van'] for r in rs])
            sc = np.array([r['stab_cau'] for r in rs])
            mu = np.array([r['mu_hat_rel'] for r in rs])
            av = np.array([r['auc_van'] for r in rs])
            ac = np.array([r['auc_cau'] for r in rs])
            cau_wins = np.mean(sc >= sv)
            mu_on = np.mean(mu > 0)
            pepq_frac = np.mean([pepq_i in r['sup_cau'] for r in rs]) if pepq_i >= 0 else float('nan')
            print(f'{target:14s} {K:>2d} {len(y):>4d} {int((y==1).sum()):>4d} | '
                  f'{sv.mean():>8.3f} {sc.mean():>8.3f} {sc.mean()-sv.mean():>+7.3f} {cau_wins:>8.0%} | '
                  f'{np.median(mu):>7.2f} {mu_on:>5.0%} | {av.mean():>5.3f} {ac.mean():>5.3f} | {pepq_frac:>5.0%}',
                  flush=True)


if __name__ == '__main__':
    main()
