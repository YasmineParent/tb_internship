"""Parity: causal-prior FasterRisk vs vanilla, on any benchmark (--dataset).

No causal ground truth on a real benchmark, so the bar is parity: causal AUC
within noise of vanilla at matched sparsity. Leakage-free: q is discovered once on
a held-out discovery set (disjoint from all eval rows), and per split the
quantile-binarizer and mu scale are fit on train rows only and applied to test.
vanilla (mu=0) vs causal (mu by inner-CV log-loss) at matched K over the splits.

usage:
    python experiments/causal_prior/real/parity.py --dataset fico --sentinel-nan --qsrc ges_cg
    python experiments/causal_prior/real/parity.py --dataset heart
    python experiments/causal_prior/real/parity.py --dataset fico --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.priors import discover_q  # noqa: E402
from src.causal_prior.scorecard import fit_eval, import_fasterrisk  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='fico')
    p.add_argument('--qsrc', choices=['pc', 'ges', 'pc_cg', 'ges_cg'], default='ges_cg')
    p.add_argument('--k', type=int, default=10, help='matched sparsity for both arms')
    p.add_argument('--splits', type=int, default=10, help='outer train/test splits')
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--discovery-frac', type=float, default=0.3,
                   help='fraction held out as the q-discovery set (disjoint from eval)')
    p.add_argument('--n_cv', type=int, default=5, help='inner CV folds for mu selection')
    p.add_argument('--b', type=int, default=100, help='discovery stability subsamples')
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--sentinel-nan', action='store_true',
                   help='(fico) treat -7/-8/-9 as missing + add _missing indicators')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b, args.splits = 10, 2
    return args


def main():
    args = parse_args()
    X_orig, names, y = load_dataset(args.dataset, args)
    n, p_orig = X_orig.shape
    print(f'{args.dataset}: n={n}  features={p_orig}  positive={(y > 0).mean():.0%}', flush=True)

    # leakage-free: discover q once on a held-out set disjoint from all eval rows
    pool_idx, disc_idx = train_test_split(np.arange(n), test_size=args.discovery_frac,
                                          stratify=(y > 0), random_state=args.seed)
    print(f'discovery n={len(disc_idx)} (held out), eval pool n={len(pool_idx)}', flush=True)
    print(f'{args.qsrc.upper()} discovery (B={args.b}) on held-out set...', flush=True)
    q_orig = discover_q(args.qsrc, X_orig[disc_idx], y[disc_idx].astype(float), args.b, args.seed)

    FasterRisk = import_fasterrisk()
    sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size,
                                 random_state=args.seed)
    records = []
    for s, (trp, tep) in enumerate(sss.split(pool_idx, (y[pool_idx] > 0).astype(int))):
        tr, te = pool_idx[trp], pool_idx[tep]
        spec, _, parent = fit_binarizer(X_orig[tr], names, args.n_thresholds)
        q_bin = q_orig[parent]
        Xtr, Xte = apply_binarizer(X_orig[tr], spec), apply_binarizer(X_orig[te], spec)
        ytr, yte = y[tr], y[te]
        mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ytr)))
        mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, 3 if args.smoke else 12)]) * mu_scale
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            van = fit_eval(FasterRisk, Xtr, ytr, Xte, yte, 0.0, None, args.k)
            cv = cv_pick_mu(Xtr, ytr, K=args.k, mu_grid=mu_grid, q=q_bin,
                            n_splits=args.n_cv, criterion='log_loss',
                            rng=np.random.default_rng(args.seed + s))
            cau = fit_eval(FasterRisk, Xtr, ytr, Xte, yte, cv.mu_star, q_bin, args.k)
        mu_hat_rel = cv.mu_star / mu_scale if mu_scale else 0.0
        records.append({'split': s, 'arm': 'vanilla', 'mu_hat_rel': 0.0, **van})
        records.append({'split': s, 'arm': 'causal', 'mu_hat_rel': mu_hat_rel, **cau})
        print(f'  split {s + 1}/{args.splits} done (mu_hat_rel={mu_hat_rel:.2f})', flush=True)

    df_splits = pd.DataFrame(records)
    metrics = ['auc', 'brier', 'ece', 'nfeat']
    df_parity = (df_splits.groupby('arm')[metrics]
                 .agg(['mean', 'std']).loc[['vanilla', 'causal']])
    mu_hats = df_splits.loc[df_splits['arm'] == 'causal', 'mu_hat_rel'].to_numpy()

    suffix = (f'{args.dataset}_{args.qsrc}_k{args.k}' + ('_sent' if args.sentinel_nan else '')
              + ('_smoke' if args.smoke else ''))
    config = {**vars(args), 'n': int(n), 'p_orig': int(p_orig),
              'discovery_n': int(len(disc_idx)), 'eval_pool_n': int(len(pool_idx)),
              'leakage_free': True, 'mu_hat_rel_mean': float(mu_hats.mean()),
              'mu_hat_nonzero_splits': int((mu_hats > 0).sum())}
    output_dir = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'parity' / suffix, config)

    (pd.DataFrame({'feature': names, 'q': q_orig})
     .sort_values('q', ascending=False).to_csv(output_dir / 'q.csv', index=False))
    df_splits.to_csv(output_dir / 'splits.csv', index=False)
    df_parity.to_csv(output_dir / 'parity.csv')

    print(df_parity.to_string(), flush=True)
    print(f'mu_hat_rel mean {mu_hats.mean():.3f}, '
          f'nonzero in {int((mu_hats > 0).sum())}/{args.splits} splits', flush=True)
    print(f'Done (leakage-free). Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
