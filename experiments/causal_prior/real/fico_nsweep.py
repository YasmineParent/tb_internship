"""FICO scarcity sweep: support-stability gain vs training size, leakage-free.

Tests the §6.3-style claim on a public benchmark: does a reliable external prior
stabilize scorecard feature selection when training data is scarce, and does the
effect close as data grows (as the §5.1 theorem predicts)?

Leakage-free: q is discovered ONCE on a held-out discovery set, disjoint from both
the eval pool (where training subsamples are drawn) and the fixed test set. For
each training size n, binarization thresholds and the mu scale are fit on the
subsample (train) only. Support stability = mean pairwise Jaccard of the selected
support across resamples, measured at the ORIGINAL-feature level (binarized columns
mapped back to their parent feature), so it is not confused by per-resample
threshold differences.

Run under `caffeinate -i` on a laptop (App Nap suspends background jobs); prefer the vm.

usage:
    python experiments/causal_prior/real/fico_nsweep.py --qsrc ges_cg --sentinel-nan --n-jobs 8
    python experiments/causal_prior/real/fico_nsweep.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.fico_parity import (  # noqa: E402
    DATA, TARGET, POS_LABEL, load_features, fit_binarizer, apply_binarizer,
    discover_q, _import_fasterrisk)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--qsrc', choices=['pc', 'ges', 'pc_cg', 'ges_cg'], default='ges_cg')
    p.add_argument('--n-grid', default='150,300,600,1200')
    p.add_argument('--reps', type=int, default=10, help='resamples per training size')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--discovery-frac', type=float, default=0.3)
    p.add_argument('--test-n', type=int, default=2500, help='fixed held-out test set size')
    p.add_argument('--n-mu', type=int, default=6)
    p.add_argument('--mu-rel', type=float, default=-1.0,
                   help='fixed relative mu; >=0 skips inner cv (faster)')
    p.add_argument('--b', type=int, default=100, help='discovery stability subsamples')
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b, args.reps, args.n_grid = 10, 3, '150,600'
    args.n_grid = [int(x) for x in args.n_grid.split(',')]
    return args


def _unit(n, r, train_pool, test_abs, X_orig, y, q_orig, names, k,
          n_thresholds, n_mu, mu_rel, n_cv, seed):
    """One resample at size n: fit train-only binarizer + scorecards, score on the
    fixed test set. Returns auc and the original-feature support set for each arm."""
    FasterRisk = _import_fasterrisk()
    sub = np.random.default_rng((seed, n, r)).choice(train_pool, size=n, replace=False)
    spec, _, parent = fit_binarizer(X_orig[sub], names, n_thresholds)
    q_bin = q_orig[parent]
    Xtr, Xte = apply_binarizer(X_orig[sub], spec), apply_binarizer(X_orig[test_abs], spec)
    ytr, yte = y[sub], (y[test_abs] > 0).astype(int)
    names_arr = np.asarray(names)
    mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ytr)))

    def supp_feats(fr):  # selected columns -> parent original features
        cols = np.nonzero(np.asarray(fr.betas_[0]))[0]
        return frozenset(names_arr[parent[cols]].tolist())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = FasterRisk(k=k, mu=0.0, freq=None); van.fit(Xtr, ytr)
        if mu_rel >= 0:
            mu_star = mu_rel * mu_scale
        else:
            mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, n_mu)]) * mu_scale
            mu_star = cv_pick_mu(Xtr, ytr, K=k, mu_grid=mu_grid, q=q_bin, n_splits=n_cv,
                                 criterion='log_loss',
                                 rng=np.random.default_rng((seed, n, r, 1))).mu_star
        cau = FasterRisk(k=k, mu=float(mu_star), freq=q_bin.astype(float)); cau.fit(Xtr, ytr)
    av = roc_auc_score(yte, np.clip(van.predict_proba(Xte), 1e-7, 1 - 1e-7))
    ac = roc_auc_score(yte, np.clip(cau.predict_proba(Xte), 1e-7, 1 - 1e-7))
    return {'n': n, 'rep': r, 'auc_van': float(av), 'auc_cau': float(ac),
            'supp_van': supp_feats(van), 'supp_cau': supp_feats(cau),
            'mu_hat_rel': mu_star / (mu_scale or 1.0)}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _stability(supports):
    pairs = [_jaccard(a, b) for a, b in combinations(supports, 2)]
    return float(np.mean(pairs)) if pairs else float('nan')


def plot_nsweep(df, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, c in (('van', 'C0'), ('cau', 'C1')):
        axes[0].plot(df['n'], df[f'stab_{arm}'], marker='o', color=c,
                     label='vanilla' if arm == 'van' else 'causal')
        axes[1].plot(df['n'], df[f'auc_{arm}'], marker='o', color=c,
                     label='vanilla' if arm == 'van' else 'causal')
    axes[0].set(xlabel='train n', ylabel='support stability (Jaccard)',
                title='scarcity-regime stability', ylim=(0, 1))
    axes[1].set(xlabel='train n', ylabel='test AUC', title='accuracy (parity)')
    for ax in axes:
        ax.set_xscale('log'); ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if not DATA.exists():
        sys.exit(f'FICO CSV not found at {DATA}')

    df = pd.read_csv(DATA)
    y = np.where(df[TARGET].astype(str).str.strip() == POS_LABEL, 1, -1).astype(int)
    X_orig, names = load_features(df, args.sentinel_nan)
    n_total = len(y)

    rest, disc_idx = train_test_split(np.arange(n_total), test_size=args.discovery_frac,
                                      stratify=(y > 0), random_state=args.seed)
    train_pool, test_abs = train_test_split(rest, test_size=args.test_n,
                                            stratify=(y[rest] > 0), random_state=args.seed)
    if max(args.n_grid) > len(train_pool):
        sys.exit(f'largest n ({max(args.n_grid)}) exceeds train pool ({len(train_pool)})')
    print(f'FICO n={n_total}: discovery={len(disc_idx)} (held out), '
          f'train pool={len(train_pool)}, fixed test={len(test_abs)}', flush=True)

    print(f'{args.qsrc.upper()} discovery (B={args.b}) on held-out set...', flush=True)
    q_orig = discover_q(args.qsrc, X_orig[disc_idx], y[disc_idx].astype(float), args.b, args.seed)

    jobs = [(n, r) for n in args.n_grid for r in range(args.reps)]
    ua = (train_pool, test_abs, X_orig, y, q_orig, names, args.k,
          args.n_thresholds, args.n_mu, args.mu_rel, 5, args.seed)
    print(f'{len(jobs)} resamples on n_jobs={args.n_jobs}', flush=True)
    if args.n_jobs == 1:
        res = [_unit(n, r, *ua) for n, r in jobs]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=5)(
            delayed(_unit)(n, r, *ua) for n, r in jobs)

    rows = []
    for n in args.n_grid:
        rn = [x for x in res if x['n'] == n]
        rows.append({
            'n': n,
            'auc_van': float(np.mean([x['auc_van'] for x in rn])),
            'auc_cau': float(np.mean([x['auc_cau'] for x in rn])),
            'stab_van': _stability([x['supp_van'] for x in rn]),
            'stab_cau': _stability([x['supp_cau'] for x in rn]),
            'mu_nonzero_frac': float(np.mean([x['mu_hat_rel'] > 0 for x in rn])),
        })
    table = pd.DataFrame(rows)
    table['auc_delta'] = table['auc_cau'] - table['auc_van']
    table['stab_delta'] = table['stab_cau'] - table['stab_van']

    suffix = f'fico_{args.qsrc}' + ('_sent' if args.sentinel_nan else '') + ('_smoke' if args.smoke else '')
    out = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'fico_nsweep' / suffix,
                      {**vars(args), 'leakage_free': True})
    table.to_csv(out / 'nsweep.csv', index=False)
    plot_nsweep(table, out / 'nsweep.png')

    print(table.round(3).to_string(index=False), flush=True)
    print(f'Done (leakage-free). Results + figure in {out}/', flush=True)


if __name__ == '__main__':
    main()
