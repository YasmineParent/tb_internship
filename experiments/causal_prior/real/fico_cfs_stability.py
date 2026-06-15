"""FICO: selection stability, causal-prior vs CFS baselines, the honest way.

The stability claim is that hard feature selection jumps around under data
perturbation and the soft causal prior stays consistent. To measure that fairly
EVERY method must re-select on each resample (computing a blanket once would make
CFS trivially stable and hide the effect). So: draw a training subsample, let each
method select features AND fit a scorecard on it, score on a fixed held-out test
set, and report the cross-resample Jaccard of the selected (original-feature)
support per method, alongside test AUC.

Note on fairness: our q is a stability selection (B subsamples) by construction,
while the CFS baselines are single-run. A stability-aggregated CFS would be the
fully matched variant (open decision, see notes/cfs_comparison_design.md).

Run on the vm (or under `caffeinate -i` locally); this re-runs causal discovery
and the CFS blankets per resample, so it is selection-heavy.

usage:
    python experiments/causal_prior/real/fico_cfs_stability.py --sentinel-nan --n 600 --reps 15
    python experiments/causal_prior/real/fico_cfs_stability.py --sentinel-nan --n-grid 150 200 300 600 --reps 20
    python experiments/causal_prior/real/fico_cfs_stability.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'external' / 'pyCausalFS'))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from src.causal_prior.priors import bnlearn_mb  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.fico_parity import (  # noqa: E402
    DATA, TARGET, POS_LABEL, load_features, fit_binarizer, apply_binarizer,
    discover_q, fit_eval, _import_fasterrisk)
from experiments.causal_prior.real.fico_cfs import (  # noqa: E402
    _cfs_fisherz, _jaccard, _stability, _nogueira)


def _orig_support(betas, cols, parent, names):
    nz = np.nonzero(np.asarray(betas))[0]
    return frozenset(names[parent[cols[i]]] for i in nz)


def _unit(r, train_pool, test_abs, X_orig, y, names, n, k, b, n_thresholds,
          alpha, n_cv, n_mu, qsrc, seed):
    """One resample: every method selects features AND fits a scorecard on the same
    n-row subsample, scored on the fixed test set. Returns (auc, support) per arm."""
    from sklearn.metrics import roc_auc_score
    FR = _import_fasterrisk()
    sub = np.random.default_rng((seed, n, r)).choice(train_pool, size=n, replace=False)
    Xs, ys = X_orig[sub], y[sub]

    # per-resample selections (this is the point: each is recomputed on this subsample)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = discover_q(qsrc, Xs, ys.astype(float), b, seed)
        mbs = {'cfs_iamb': _cfs_fisherz('iamb', Xs, ys, alpha),
               'cfs_hiton_mb': _cfs_fisherz('hiton_mb', Xs, ys, alpha),
               'cfs_cg': bnlearn_mb(Xs, ys, method='iamb', test='mi-cg', alpha=alpha)}

    spec, _, parent = fit_binarizer(Xs, names.tolist(), n_thresholds)
    q_bin = q[parent]
    Xtr, Xte = apply_binarizer(Xs, spec), apply_binarizer(X_orig[test_abs], spec)
    yte = (y[test_abs] > 0).astype(int)
    all_cols = np.arange(Xtr.shape[1])
    mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ys)))
    mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, n_mu)]) * mu_scale

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = fit_eval(FR, Xtr, ys, Xte, yte, 0.0, None, k, return_card=True)
        out['vanilla'] = (van['auc'], _orig_support(van['card']['betas'], all_cols, parent, names))
        cv = cv_pick_mu(Xtr, ys, K=k, mu_grid=mu_grid, q=q_bin, n_splits=n_cv,
                        criterion='log_loss', rng=np.random.default_rng((seed, n, r, 1)))
        cau = fit_eval(FR, Xtr, ys, Xte, yte, cv.mu_star, q_bin, k, return_card=True)
        out['causal'] = (cau['auc'], _orig_support(cau['card']['betas'], all_cols, parent, names))
        for arm, mb in mbs.items():
            cols = all_cols[np.isin(parent, mb)] if mb else all_cols[:0]
            if len(cols) == 0:
                out[arm] = (float('nan'), frozenset())
                continue
            fr = fit_eval(FR, Xtr[:, cols], ys, Xte[:, cols], yte, 0.0, None, k, return_card=True)
            out[arm] = (fr['auc'], _orig_support(fr['card']['betas'], cols, parent, names))
    return r, out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--qsrc', default='ges_cg')
    p.add_argument('--n', type=int, default=600, help='training subsample size per resample')
    p.add_argument('--n-grid', type=int, nargs='+', default=None,
                   help='scarcity sweep: run the protocol at each n (overrides --n)')
    p.add_argument('--reps', type=int, default=15)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--test-n', type=int, default=2500)
    p.add_argument('--b', type=int, default=50, help='subsamples for the causal-prior stability selection')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n_cv', type=int, default=5)
    p.add_argument('--n-mu', type=int, default=8)
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--n-jobs', type=int, default=1,
                   help='parallel over resamples; >1 inits R per worker (heavier, can be flaky)')
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b, args.reps, args.n_grid = 10, 4, [300, 600]
    return args


ARMS = ['vanilla', 'causal', 'cfs_iamb', 'cfs_hiton_mb', 'cfs_cg']


def _run_at_n(n, args, train_pool, test_abs, X_orig, y, names):
    """run all resamples at one subsample size, return the summary rows (one per arm)."""
    ua = (train_pool, test_abs, X_orig, y, names, n, args.k, args.b,
          args.n_thresholds, args.alpha, args.n_cv, args.n_mu, args.qsrc, args.seed)
    if args.n_jobs == 1:
        results = [_unit(r, *ua) for r in range(args.reps)]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=5)(
            delayed(_unit)(r, *ua) for r in range(args.reps))

    aucs = {a: [] for a in ARMS}
    supports = {a: [] for a in ARMS}
    for _, out in results:
        for a in ARMS:
            auc, supp = out[a]
            aucs[a].append(auc)
            supports[a].append(supp)

    d = len(names)  # full feature universe, the common chance baseline for nogueira
    return pd.DataFrame({
        'n': n,
        'arm': ARMS,
        'auc_mean': [float(np.nanmean(aucs[a])) for a in ARMS],
        'auc_std': [float(np.nanstd(aucs[a])) for a in ARMS],
        'stability_jaccard': [_stability(supports[a]) for a in ARMS],
        'stability_nogueira': [_nogueira(supports[a], d) for a in ARMS],
        'mean_support_size': [float(np.mean([len(s) for s in supports[a]])) for a in ARMS],
    }).round(4)


def main():
    args = parse_args()
    if not DATA.exists():
        sys.exit(f'FICO CSV not found at {DATA}')

    df = pd.read_csv(DATA)
    y = np.where(df[TARGET].astype(str).str.strip() == POS_LABEL, 1, -1).astype(int)
    X_orig, names = load_features(df, args.sentinel_nan)
    names = np.asarray(names)

    grid = args.n_grid or [args.n]
    train_pool, test_abs = train_test_split(np.arange(len(y)), test_size=args.test_n,
                                            stratify=(y > 0), random_state=args.seed)
    if max(grid) > len(train_pool):
        sys.exit(f'max n ({max(grid)}) exceeds train pool ({len(train_pool)})')
    print(f'FICO: train pool={len(train_pool)}, fixed test={len(test_abs)}; '
          f'{args.reps} resamples per n, re-selecting each time; n grid={grid}', flush=True)

    out_dir = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'fico_cfs_stability'
                          / f'fico_sweep_k{args.k}', {**vars(args), 'n_grid': grid, 'leakage_free': True})
    parts = []
    for n in grid:
        print(f'\n=== n={n} ===', flush=True)
        part = _run_at_n(n, args, train_pool, test_abs, X_orig, y, names)
        print(part.to_string(index=False), flush=True)
        parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(out_dir / 'summary.csv', index=False)

    print(f'\nDone. Results in {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
