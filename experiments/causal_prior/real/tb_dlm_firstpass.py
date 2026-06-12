"""First pass: causal-prior FasterRisk vs vanilla on real TB-DLM data.

Target = agar delamanid R/S call (interp_dlm016); features = prevalence-feasible
[5%, 98%] dlm-pathway mutations + lineage indicators; prior q_j = stability
frequency of (j -> dlm_mic) from the CMM with_lineage_forbid_to_mic run.

Vanilla (mu=0) vs causal (mu by log-loss CV) at matched sparsity K. Success:
mu_hat > 0, stability(causal) >= stability(vanilla), support biologically
defensible. AUC is reference only (wide CIs at this n). Writes q and the
per-arm results to results/.

Usage:
    python experiments/causal_prior/real/tb_dlm_firstpass.py
    python experiments/causal_prior/real/tb_dlm_firstpass.py --smoke
    python experiments/causal_prior/real/tb_dlm_firstpass.py --k 7 --n-splits 20
"""
from __future__ import annotations

import argparse
import json
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
OUT_DIR = ROOT / 'results' / 'causal_prior' / 'tb_dlm_firstpass'
MIC_NODE = 'dlm_mic'
TARGET = 'interp_dlm016'
DLM_GENES = ('rv0678', 'mmpl5', 'mmps5', 'atpe', 'pepq', 'rv1979c', 'fgd1', 'ddn')
PREV_LO, PREV_HI = 0.05, 0.98


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=5, help='matched sparsity for both arms')
    p.add_argument('--n-splits', type=int, default=10, help='CV folds for stability/mu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true',
                   help='shrink the run for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.n_splits = 3
    return args


def build_xyq(df):
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = np.where(df[TARGET].to_numpy() == 2.0, 1, -1).astype(int)

    mut_cols = [c for c in df.columns if any(c.startswith(g + '_') for g in DLM_GENES)]
    M = df[mut_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0)
    prev = M.mean()
    feasible = sorted(prev[(prev >= PREV_LO) & (prev <= PREV_HI)].index)

    lineage = df['lineage'].astype('Int64')
    lin_feats = {f'lineage_{lv}': (lineage == lv).astype(float).to_numpy()
                 for lv in (2, 4)}  # lineages 1/3 merged into reference
    feat_names = feasible + list(lin_feats)
    X = np.column_stack([M[feasible].to_numpy()] + [lin_feats[c] for c in lin_feats])

    edges = pd.read_csv(QCSV)
    into_mic = edges[edges['target'] == MIC_NODE].set_index('source')['frequency']
    q = np.array([float(into_mic.get(name, 0.0)) for name in feat_names])
    return X, y, q, feat_names


def main():
    args = parse_args()
    X, y, q, feat_names = build_xyq(pd.read_csv(DATA))
    n_pos = int((y == 1).sum())
    print(f'n={len(y)}  resistant={n_pos} ({n_pos/len(y):.0%})  '
          f'p={len(feat_names)}  K={args.k}', flush=True)

    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, 12)]) * mu_scale
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vanilla = cv_pick_mu(X, y, K=args.k, mu_grid=np.array([0.0]), q=None,
                             n_splits=args.n_splits, criterion='log_loss',
                             rng=np.random.default_rng(args.seed))
        causal = cv_pick_mu(X, y, K=args.k, mu_grid=mu_grid, q=q,
                            n_splits=args.n_splits, criterion='log_loss',
                            rng=np.random.default_rng(args.seed))

    def names(support):
        return [feat_names[j] for j in support]

    mu_hat_rel = causal.mu_star / mu_scale if mu_scale else 0.0
    results = pd.DataFrame([
        {'arm': 'vanilla', 'mu_hat_rel': 0.0, 'stability': vanilla.stability_star,
         'auc': vanilla.auc_star, 'log_loss': vanilla.log_loss_star,
         'support': json.dumps(names(vanilla.support))},
        {'arm': 'causal', 'mu_hat_rel': mu_hat_rel, 'stability': causal.stability_star,
         'auc': causal.auc_star, 'log_loss': causal.log_loss_star,
         'support': json.dumps(names(causal.support))},
    ])

    suffix = f'tb_dlm_firstpass_k{args.k}' + ('_smoke' if args.smoke else '')
    output_dir = new_run_dir(OUT_DIR / suffix,
                             {**vars(args), 'data': str(DATA), 'target': TARGET,
                              'n': len(y), 'n_pos': n_pos, 'p': len(feat_names),
                              'mu_scale': mu_scale})
    (pd.DataFrame({'feature': feat_names, 'prev': X.mean(axis=0), 'q': q})
     .sort_values('q', ascending=False).to_csv(output_dir / 'q.csv', index=False))
    results.to_csv(output_dir / 'results.csv', index=False)

    print(results.to_string(index=False), flush=True)
    has_pepq = 'pepq_Ala87Gly' in names(causal.support)
    print(f'mu_hat>0: {mu_hat_rel > 0}  |  '
          f'stability(causal>=vanilla): {causal.stability_star >= vanilla.stability_star} '
          f'({causal.stability_star:.3f} vs {vanilla.stability_star:.3f})  |  '
          f'pepq_Ala87Gly selected: {has_pepq}', flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
