"""FICO HELOC parity *curve*: test AUC vs model size, vanilla vs causal.

The §6.2 parity claim made the way the FasterRisk paper makes it (their Fig 4):
sweep the sparsity k and plot test AUC for both arms with error bars over splits.
"Curves lie on top of each other" *is* the parity claim, read in two seconds by
anyone who knows the FasterRisk figure.

Leakage-free protocol: the causal q is discovered ONCE on a held-out discovery
set (disjoint from all train/test rows), so the prior never sees the rows it is
evaluated on. Within each train/test split the binarization thresholds and the
mu scale are fit on train rows only and applied to test. This is the defensible
version of the earlier full-data-q runner.

usage:
    python experiments/causal_prior/real/fico_ksweep.py --qsrc ges_cg --sentinel-nan --n-jobs 16
    python experiments/causal_prior/real/fico_ksweep.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.causal_prior.cv_mu import cv_pick_mu, make_mu_grid  # noqa: E402
from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.priors import discover_q  # noqa: E402
from src.causal_prior.scorecard import fit_eval, import_fasterrisk  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='fico')
    p.add_argument('--qsrc', choices=['pc', 'ges', 'pc_cg', 'ges_cg'], default='ges_cg')
    p.add_argument('--k-grid', default='2,4,6,8,10',
                   help='comma-separated sparsities to sweep')
    p.add_argument('--splits', type=int, default=10, help='train/test splits on the eval pool')
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--discovery-frac', type=float, default=0.3,
                   help='fraction of data held out as the q-discovery set (disjoint from eval)')
    p.add_argument('--n_cv', type=int, default=5)
    p.add_argument('--n-mu', type=int, default=12,
                   help='log-spaced mu grid points for the causal CV (dominant cost)')
    p.add_argument('--b', type=int, default=100, help='discovery stability subsamples')
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--n-jobs', type=int, default=1,
                   help='parallel workers over the train/test splits')
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b, args.splits, args.k_grid = 10, 2, '2,6,10'
    args.k_grid = [int(k) for k in args.k_grid.split(',')]
    return args


def _split_unit(s, train_abs, test_abs, X_orig, y, q_orig, names, k_grid,
                n_thresholds, n_mu, n_cv, smoke, seed):
    """One train/test split, all k, both arms. Binarization thresholds and mu
    scale are fit on this split's TRAIN rows only (no leakage); q_orig comes from
    the disjoint held-out discovery set. Records metrics, the fitted scorecards,
    and the causal arm's cv trace. Module-level + pure-Python so it pickles for
    joblib (no R here; discovery already ran once in the parent)."""
    FasterRisk = import_fasterrisk()
    spec, col_names, parent = fit_binarizer(X_orig[train_abs], names, n_thresholds)
    q_bin = q_orig[parent]
    Xtr, Xte = apply_binarizer(X_orig[train_abs], spec), apply_binarizer(X_orig[test_abs], spec)
    ytr, yte = y[train_abs], y[test_abs]
    mu_scale, mu_grid = make_mu_grid(Xtr, ytr, 3 if smoke else n_mu)

    metrics, coefs, trace = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k in k_grid:
            van = fit_eval(FasterRisk, Xtr, ytr, Xte, yte, 0.0, None, k, return_card=True)
            cv = cv_pick_mu(Xtr, ytr, K=k, mu_grid=mu_grid, q=q_bin, n_splits=n_cv,
                            criterion='log_loss', rng=np.random.default_rng(seed + s))
            cau = fit_eval(FasterRisk, Xtr, ytr, Xte, yte, cv.mu_star, q_bin, k,
                           return_card=True)
            mu_hat_rel = cv.mu_star / mu_scale if mu_scale else 0.0
            vc, cc = van.pop('card'), cau.pop('card')
            metrics += [
                {'split': s, 'k': k, 'arm': 'vanilla', 'mu_hat_rel': 0.0, 'mu_star': 0.0,
                 'intercept': vc['intercept'], 'multiplier': vc['multiplier'], **van},
                {'split': s, 'k': k, 'arm': 'causal', 'mu_hat_rel': mu_hat_rel,
                 'mu_star': float(cv.mu_star), 'intercept': cc['intercept'],
                 'multiplier': cc['multiplier'], **cau},
            ]
            for arm, card in (('vanilla', vc), ('causal', cc)):
                coefs += [{'split': s, 'k': k, 'arm': arm, 'feature': col_names[j],
                           'coef': int(b)}
                          for j, b in enumerate(card['betas']) if b != 0]
            trace += [{'split': s, 'k': k, 'mu_index': i,
                       'mu_rel': float(mu_grid[i] / (mu_scale or 1.0)),
                       'cv_auc': float(a), 'cv_log_loss': float(ll), 'cv_stability': float(st)}
                      for i, (a, ll, st) in enumerate(zip(cv.aucs_per_mu,
                                                          cv.log_losses_per_mu,
                                                          cv.stabilities_per_mu))]
    return {'metrics': metrics, 'coefs': coefs, 'trace': trace}


def main():
    args = parse_args()
    X_orig, names, y = load_dataset(args.dataset, args)
    n, p_orig = X_orig.shape

    # hold out a disjoint discovery set; q never sees the eval rows
    pool_idx, disc_idx = train_test_split(
        np.arange(n), test_size=args.discovery_frac,
        stratify=(y > 0), random_state=args.seed)
    print(f'{args.dataset}: n={n}  features={p_orig}  positive={(y > 0).mean():.0%}; '
          f'discovery set n={len(disc_idx)} (held out), eval pool n={len(pool_idx)}', flush=True)

    print(f'{args.qsrc.upper()} discovery (B={args.b}) on held-out set...', flush=True)
    q_orig = discover_q(args.qsrc, X_orig[disc_idx], y[disc_idx].astype(float), args.b, args.seed)

    sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size,
                                 random_state=args.seed)
    splits = [(pool_idx[trp], pool_idx[tep])
              for trp, tep in sss.split(pool_idx, (y[pool_idx] > 0).astype(int))]
    print(f'{len(splits)} splits on eval pool; n_jobs={args.n_jobs}', flush=True)

    ua = (X_orig, y, q_orig, names, args.k_grid, args.n_thresholds,
          args.n_mu, args.n_cv, args.smoke, args.seed)
    if args.n_jobs == 1:
        results = [_split_unit(s, tr, te, *ua) for s, (tr, te) in enumerate(splits)]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=5)(
            delayed(_split_unit)(s, tr, te, *ua) for s, (tr, te) in enumerate(splits))

    metric_rows = [r for u in results for r in u['metrics']]
    coef_rows = [r for u in results for r in u['coefs']]
    trace_rows = [r for u in results for r in u['trace']]

    df_splits = pd.DataFrame(metric_rows)
    metrics = ['auc', 'brier', 'ece', 'nfeat']
    df_agg = (df_splits.groupby(['k', 'arm'])[metrics]
              .agg(['mean', 'std']).reset_index())
    df_agg.columns = ['k', 'arm'] + [f'{m}_{s}' for m in metrics for s in ('mean', 'std')]

    suffix = (f'{args.dataset}_{args.qsrc}_ksweep' + ('_sent' if args.sentinel_nan else '')
              + ('_smoke' if args.smoke else ''))
    config = {**vars(args), 'k_grid': args.k_grid, 'n': int(n),
              'p_orig': int(p_orig), 'discovery_n': int(len(disc_idx)),
              'eval_pool_n': int(len(pool_idx)), 'leakage_free': True}
    output_dir = new_run_dir(ROOT / 'results' / 'causal_prior' / 'ksweep' / suffix,
                             config)
    (pd.DataFrame({'feature': names, 'q': q_orig})
     .sort_values('q', ascending=False)
     .to_csv(output_dir / 'q.csv', index=False))
    df_splits.to_csv(output_dir / 'splits.csv', index=False)
    df_agg.to_csv(output_dir / 'ksweep.csv', index=False)
    pd.DataFrame(coef_rows).to_csv(output_dir / 'coefficients.csv', index=False)
    pd.DataFrame(trace_rows).to_csv(output_dir / 'cv_trace.csv', index=False)

    print(df_agg[['k', 'arm', 'auc_mean', 'auc_std', 'nfeat_mean']].to_string(index=False),
          flush=True)
    print(f'Done (leakage-free). Results + figure in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
