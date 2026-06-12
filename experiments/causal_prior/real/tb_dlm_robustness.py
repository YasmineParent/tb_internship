"""Robustness: is the vanilla->causal support-stability lift on TB-DLM real?

Loops the tb_dlm_firstpass comparison over seeds (CV fold splits), two agar
thresholds (interp_dlm016, interp_dlm06), and several K, reporting whether the
stability(causal) >= stability(vanilla) result holds or was a lucky split.

Usage:
    python experiments/causal_prior/real/tb_dlm_robustness.py
    python experiments/causal_prior/real/tb_dlm_robustness.py --smoke
    python experiments/causal_prior/real/tb_dlm_robustness.py --ks 5 --n-seeds 12
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402

DATA = ROOT / 'data/real/processed/tb_pheno_geno_clean.csv'
QCSV = ROOT / ('results/mixed_cmm/subsampling/tb_subsampling_dlm_mp4_k6_mcc4/'
               'with_lineage_forbid_to_mic/edge_stability.csv')
OUT_DIR = ROOT / 'results' / 'causal_prior' / 'tb_dlm_robustness'
MIC_NODE = 'dlm_mic'
DLM_GENES = ('rv0678', 'mmpl5', 'mmps5', 'atpe', 'pepq', 'rv1979c', 'fgd1', 'ddn')
PREV_LO, PREV_HI = 0.05, 0.98
MU_REL_GRID = np.array([0.0, 0.5, 1.5, 5.0])  # coarse: CV can still pick 0 (vanilla) or nonzero


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-splits', type=int, default=10, help='CV folds for stability/mu')
    p.add_argument('--n-seeds', type=int, default=6)
    p.add_argument('--targets', type=str, default='interp_dlm016,interp_dlm06',
                   help='comma-separated agar R/S call columns')
    p.add_argument('--ks', type=str, default='3,5,7',
                   help='comma-separated matched-sparsity values')
    p.add_argument('--smoke', action='store_true',
                   help='shrink the run for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.n_seeds, args.n_splits, args.targets, args.ks = 2, 3, 'interp_dlm016', '3'
    args.targets = tuple(args.targets.split(','))
    args.ks = tuple(int(x) for x in args.ks.split(','))
    return args


def build_xyq(df_full, target_col):
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
    q = np.array([float(into_mic.get(name, 0.0)) for name in feat_names])
    return X, y, q, feat_names


def run(X, y, q, K, seed, n_splits):
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_grid = MU_REL_GRID * mu_scale
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = cv_pick_mu(X, y, K=K, mu_grid=np.array([0.0]), q=None,
                         n_splits=n_splits, rng=np.random.default_rng(seed))
        cau = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=q,
                         n_splits=n_splits, rng=np.random.default_rng(seed))
    return dict(mu_hat_rel=cau.mu_star / mu_scale if mu_scale else 0.0,
                stab_van=van.stability_star, stab_cau=cau.stability_star,
                auc_van=van.auc_star, auc_cau=cau.auc_star,
                sup_van=tuple(van.support), sup_cau=tuple(cau.support))


def main():
    args = parse_args()
    df = pd.read_csv(DATA)
    seed_rows, summary_rows = [], []
    for target in args.targets:
        X, y, q, feat_names = build_xyq(df, target)
        pepq_i = feat_names.index('pepq_Ala87Gly') if 'pepq_Ala87Gly' in feat_names else -1
        for K in args.ks:
            rs = [run(X, y, q, K, s, args.n_splits) for s in range(args.n_seeds)]
            for s, r in enumerate(rs):
                seed_rows.append({'target': target, 'K': K, 'seed': s,
                                  'mu_hat_rel': r['mu_hat_rel'],
                                  'stab_van': r['stab_van'], 'stab_cau': r['stab_cau'],
                                  'auc_van': r['auc_van'], 'auc_cau': r['auc_cau'],
                                  'pepq': int(pepq_i in r['sup_cau']) if pepq_i >= 0 else None})
            sv = np.array([r['stab_van'] for r in rs])
            sc = np.array([r['stab_cau'] for r in rs])
            mu = np.array([r['mu_hat_rel'] for r in rs])
            summary_rows.append({
                'target': target, 'K': K, 'n': len(y), 'pos': int((y == 1).sum()),
                'stab_van': sv.mean(), 'stab_cau': sc.mean(), 'd_stab': sc.mean() - sv.mean(),
                'cau_ge_van': float(np.mean(sc >= sv)),
                'mu_hat_median': float(np.median(mu)), 'mu_on': float(np.mean(mu > 0)),
                'auc_van': np.mean([r['auc_van'] for r in rs]),
                'auc_cau': np.mean([r['auc_cau'] for r in rs]),
                'pepq_frac': float(np.mean([pepq_i in r['sup_cau'] for r in rs]))
                if pepq_i >= 0 else float('nan')})

    summary = pd.DataFrame(summary_rows)

    suffix = 'tb_dlm_robustness' + ('_smoke' if args.smoke else '')
    output_dir = new_run_dir(OUT_DIR / suffix, {**vars(args), 'data': str(DATA)})
    pd.DataFrame(seed_rows).to_csv(output_dir / 'seeds.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)

    print(summary.to_string(index=False), flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
