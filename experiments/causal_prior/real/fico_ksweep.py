"""FICO HELOC parity *curve*: test AUC vs model size, vanilla vs causal.

The §6.2 parity claim made the way the FasterRisk paper makes it (their Fig 4):
sweep the sparsity k and plot test AUC for both arms with error bars over splits.
"Curves lie on top of each other" *is* the parity claim, read in two seconds by
anyone who knows the FasterRisk figure. q is k-independent, so it's discovered
once; only the FasterRisk fit loops over k. Reuses every helper from
fico_parity so the two stay in lockstep.

Usage:
    python experiments/causal_prior/real/fico_ksweep.py --qsrc ges_cg --sentinel-nan
    python experiments/causal_prior/real/fico_ksweep.py --k-grid 2,4,6,8,10
    python experiments/causal_prior/real/fico_ksweep.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.fico_parity import (  # noqa: E402
    DATA, TARGET, POS_LABEL, load_features, binarize, discover_q, fit_eval,
    _import_fasterrisk)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--qsrc', choices=['pc', 'ges', 'pc_cg', 'ges_cg'], default='ges_cg')
    p.add_argument('--k-grid', default='2,4,6,8,10',
                   help='comma-separated sparsities to sweep')
    p.add_argument('--splits', type=int, default=10)
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--n_cv', type=int, default=5)
    p.add_argument('--n-mu', type=int, default=12,
                   help='log-spaced mu grid points for the causal CV (dominant cost)')
    p.add_argument('--b', type=int, default=100)
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--n-jobs', type=int, default=1,
                   help='parallel workers over the (split x k) units; q discovery '
                        'stays single-session. Set near the box core count.')
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b, args.splits, args.k_grid = 10, 2, '2,6,10'
    args.k_grid = [int(k) for k in args.k_grid.split(',')]
    return args


def _ksweep_unit(s, tr, te, k, X_bin, y, q_bin, mu_grid, mu_scale, n_cv, seed):
    """One (split, k) unit, both arms. Module-level and pure-Python (no R) so it
    is picklable for joblib; large arrays are memmapped by the loky backend.
    Vanilla refits per k; causal reselects mu per k (mu interacts with k). The cv
    rng is keyed on the split only, so the causal arm is reproducible across the
    sequential and parallel paths regardless of unit scheduling order."""
    FasterRisk = _import_fasterrisk()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = fit_eval(FasterRisk, X_bin[tr], y[tr], X_bin[te], y[te], 0.0, None, k)
        cv = cv_pick_mu(X_bin[tr], y[tr], K=k, mu_grid=mu_grid, q=q_bin,
                        n_splits=n_cv, criterion='log_loss',
                        rng=np.random.default_rng(seed + s))
        cau = fit_eval(FasterRisk, X_bin[tr], y[tr], X_bin[te], y[te],
                       cv.mu_star, q_bin, k)
    mu_hat_rel = cv.mu_star / mu_scale if mu_scale else 0.0
    return [{'split': s, 'k': k, 'arm': 'vanilla', 'mu_hat_rel': 0.0, **van},
            {'split': s, 'k': k, 'arm': 'causal', 'mu_hat_rel': mu_hat_rel, **cau}]


def plot_curve(df_agg, qsrc, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for arm, color in (('vanilla', 'C0'), ('causal', 'C1')):
        d = df_agg[df_agg['arm'] == arm].sort_values('k')
        ax.errorbar(d['k'], d['auc_mean'], yerr=d['auc_std'], marker='o',
                    capsize=3, color=color, label=arm)
    ax.set_xlabel('model size (k)')
    ax.set_ylabel('test AUC')
    ax.set_title(f'FICO parity curve ({qsrc})')
    ax.legend()
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
    n, p_orig = X_orig.shape
    print(f'FICO: n={n}  features={p_orig}  positive ({POS_LABEL})={(y == 1).mean():.0%}',
          flush=True)

    print(f'{args.qsrc.upper()} subsample-stability (B={args.b})...', flush=True)
    q_orig = discover_q(args.qsrc, X_orig, y.astype(float), args.b, args.seed)
    X_bin, _, parent = binarize(X_orig, names, args.n_thresholds)
    q_bin = q_orig[parent]
    print(f'binarized: {X_bin.shape[1]} cols; q_bin nonzero on {int((q_bin > 0).sum())}',
          flush=True)

    mu_scale = float(np.median(0.5 * np.abs(X_bin.T @ y)))
    mu_rel = np.logspace(-2, 1, 3 if args.smoke else args.n_mu)
    mu_grid = np.concatenate([[0.0], mu_rel]) * mu_scale

    sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size,
                                 random_state=args.seed)
    splits = list(sss.split(X_bin, (y > 0).astype(int)))
    jobs = [(s, tr, te, k) for s, (tr, te) in enumerate(splits) for k in args.k_grid]
    print(f'{len(jobs)} units ({len(splits)} splits x {len(args.k_grid)} k) '
          f'on n_jobs={args.n_jobs}', flush=True)

    unit_args = (X_bin, y, q_bin, mu_grid, mu_scale, args.n_cv, args.seed)
    if args.n_jobs == 1:
        results = [_ksweep_unit(s, tr, te, k, *unit_args) for (s, tr, te, k) in jobs]
    else:
        # module-level worker + array args so loky memmaps X_bin once, not cloudpickle per task
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=5)(
            delayed(_ksweep_unit)(s, tr, te, k, *unit_args) for (s, tr, te, k) in jobs)
    records = [r for unit in results for r in unit]

    df_splits = pd.DataFrame(records)
    metrics = ['auc', 'brier', 'ece', 'nfeat']
    df_agg = (df_splits.groupby(['k', 'arm'])[metrics]
              .agg(['mean', 'std']).reset_index())
    df_agg.columns = ['k', 'arm'] + [f'{m}_{s}' for m in metrics for s in ('mean', 'std')]

    suffix = (f'fico_{args.qsrc}_ksweep' + ('_sent' if args.sentinel_nan else '')
              + ('_smoke' if args.smoke else ''))
    config = {**vars(args), 'k_grid': args.k_grid, 'data': str(DATA),
              'n': int(n), 'p_orig': int(p_orig), 'mu_scale': mu_scale}
    output_dir = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'fico_ksweep' / suffix,
                             config)
    df_splits.to_csv(output_dir / 'splits.csv', index=False)
    df_agg.to_csv(output_dir / 'ksweep.csv', index=False)
    plot_curve(df_agg, args.qsrc, output_dir / 'parity_curve.png')

    print(df_agg[['k', 'arm', 'auc_mean', 'auc_std', 'nfeat_mean']].to_string(index=False),
          flush=True)
    print(f'Done. Results + figure in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
