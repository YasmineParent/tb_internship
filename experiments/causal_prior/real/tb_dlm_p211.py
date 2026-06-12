"""Causal prior on the raw (p >> n) TB-DLM mutation set.

Both arms run on the full mutation matrix (all genes, constant columns dropped).
The causal prior q_j is nonzero only on the features CMM evaluated (edges into
dlm_mic from the with_lineage_forbid_to_mic stability run); q_j = 0 elsewhere
encodes 'no causal evidence'. This is the regime where the prior should help most:
at p >> n vanilla can grab any of ~200 spurious mutations and its support thrashes
across folds, while the prior should pin selection to the causally-supported
features and stabilise it.

Vanilla (mu=0) vs causal (mu by log-loss CV) at matched K, over several seeds.
Headline = 10-fold support stability (Jaccard); AUC for reference.

Usage:
    python experiments/causal_prior/real/tb_dlm_p211.py
    python experiments/causal_prior/real/tb_dlm_p211.py --smoke
    python experiments/causal_prior/real/tb_dlm_p211.py --min-pos 1   # keep singletons (true p>>n)
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
OUT_DIR = ROOT / 'results' / 'causal_prior' / 'tb_dlm_p211'
MIC_NODE = 'dlm_mic'
TARGET = 'interp_dlm016'
MU_REL_GRID = np.array([0.0, 0.5, 1.5, 5.0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=5, help='matched sparsity for both arms')
    p.add_argument('--n-splits', type=int, default=10, help='CV folds for stability/mu')
    p.add_argument('--n-seeds', type=int, default=6)
    p.add_argument('--min-pos', type=int, default=2,
                   help='min positive count to keep a mutation (1 keeps singletons)')
    p.add_argument('--smoke', action='store_true',
                   help='shrink the run for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.n_seeds, args.n_splits = 2, 3
    return args


def build(min_pos):
    df = pd.read_csv(DATA)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = np.where(df[TARGET].to_numpy() == 2.0, 1, -1).astype(int)

    meta = {'isolate', 'glims', 'type', 'lineage'}
    mut_cols = [c for c in df.columns
                if c not in meta
                and not c.startswith(('interp_', 'prop_mutants'))
                and 'mic' not in c]
    M = df[mut_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0)
    pos = (M > 0).sum()
    keep = sorted(pos[(pos >= min_pos) & (pos <= len(df) - min_pos)].index)
    lin = {f'lineage_{lv}': (df['lineage'].astype('Int64') == lv).astype(float).to_numpy()
           for lv in (2, 4)}
    feat_names = keep + list(lin)
    X = np.column_stack([M[keep].to_numpy()] + [lin[c] for c in lin])

    edges = pd.read_csv(QCSV)
    into_mic = edges[edges['target'] == MIC_NODE].set_index('source')['frequency']
    q = np.array([float(into_mic.get(name, 0.0)) for name in feat_names])
    return X, y, q, feat_names


def main():
    args = parse_args()
    X, y, q, feat_names = build(args.min_pos)
    print(f'n={len(y)}  resistant={int((y==1).sum())}  p={len(feat_names)}  '
          f'nonzero_q={int((q>0).sum())}  K={args.k}', flush=True)
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_grid = MU_REL_GRID * mu_scale

    def names(support):
        return [feat_names[j] for j in support]

    rows = []
    for s in range(args.n_seeds):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            van = cv_pick_mu(X, y, K=args.k, mu_grid=np.array([0.0]), q=None,
                             n_splits=args.n_splits, rng=np.random.default_rng(s))
            cau = cv_pick_mu(X, y, K=args.k, mu_grid=mu_grid, q=q,
                             n_splits=args.n_splits, rng=np.random.default_rng(s))
        mu_rel = cau.mu_star / mu_scale if mu_scale else 0.0
        rows.append({'seed': s, 'stab_van': van.stability_star,
                     'stab_cau': cau.stability_star, 'mu_hat_rel': mu_rel,
                     'auc_van': van.auc_star, 'auc_cau': cau.auc_star,
                     'sup_van': json.dumps(names(van.support)),
                     'sup_cau': json.dumps(names(cau.support))})
        print(f'  seed {s}: stab {van.stability_star:.3f}->{cau.stability_star:.3f} '
              f'(d={cau.stability_star-van.stability_star:+.3f})  mu_hat={mu_rel:.2f}  '
              f'auc {van.auc_star:.3f}->{cau.auc_star:.3f}', flush=True)

    df = pd.DataFrame(rows)
    pepq_frac = np.mean(['pepq_Ala87Gly' in json.loads(r['sup_cau']) for r in rows])

    suffix = f'tb_dlm_p211_k{args.k}_minpos{args.min_pos}' + ('_smoke' if args.smoke else '')
    output_dir = new_run_dir(OUT_DIR / suffix,
                             {**vars(args), 'data': str(DATA), 'target': TARGET,
                              'n': len(y), 'p': len(feat_names),
                              'nonzero_q': int((q > 0).sum()), 'mu_scale': mu_scale})
    (pd.DataFrame({'feature': feat_names, 'q': q})
     .sort_values('q', ascending=False).to_csv(output_dir / 'q.csv', index=False))
    df.to_csv(output_dir / 'results.csv', index=False)

    print(f"\nMEAN stab {df['stab_van'].mean():.3f}->{df['stab_cau'].mean():.3f}  "
          f"(d={df['stab_cau'].mean()-df['stab_van'].mean():+.3f}, "
          f"causal>=vanilla {np.mean(df['stab_cau']>=df['stab_van']):.0%})  |  "
          f"AUC {df['auc_van'].mean():.3f}->{df['auc_cau'].mean():.3f}  |  "
          f"pepq_Ala87Gly in {pepq_frac:.0%} of seeds", flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
