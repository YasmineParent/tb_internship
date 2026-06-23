"""wall-clock runtime of the soft-causal-prior pipeline, decomposed by stage.

times the real code paths (no toy reimplementation) on each public benchmark, on
the full dataset (the single-deployment cost), and reports where the time goes:

  discovery   iamb soft q, B subsamples of bnlearn iamb + mi-cg (the deployed q)
  cfs_select  one iamb + mi-cg markov blanket (the hard-selection baseline's cost)
  fit         one fasterrisk fit at k (soft prior = same fit as vanilla, freq only)
  mu_cv       cv_pick_mu over the mu grid (n_mu x n_cv fasterrisk fits)

derived totals (what a practitioner actually waits for, per deployment):
  vanilla     = fit
  ours_full   = discovery + mu_cv + fit      (the headline pipeline)
  ours_fast   = discovery + fit              (skip the mu grid, fixed mu_rel)
  cfs_hard    = cfs_select + fit             (hard pre-selection baseline)

the soft prior adds ~0 over vanilla at fit time, and discovery is the SAME step
the hard cfs baseline pays, so ours is not slower than hard selection; the only
extra cost is the mu grid, which ours_fast drops. usage:

    python experiments/causal_prior/real/runtime_bench.py
    python experiments/causal_prior/real/runtime_bench.py --datasets fico,heart --repeats 3
"""
from __future__ import annotations

import argparse
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.cv_mu import cv_pick_mu, make_mu_grid  # noqa: E402
from src.causal_prior.priors import bnlearn_mb, bnlearn_mb_stability_q  # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk, fit_eval  # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset  # noqa: E402

IN_REGIME = ['fico', 'heart', 'mammographic', 'ilpd', 'german']


def _loader_args(sentinel_nan: bool) -> types.SimpleNamespace:
    # loaders read a few options off the args namespace (e.g. fico's --sentinel-nan)
    return types.SimpleNamespace(sentinel_nan=sentinel_nan)


def bench_dataset(name: str, k: int, n_thresholds: int, b: int,
                  n_mu: int, n_cv: int, alpha: float, mu_fast_rel: float,
                  test_frac: float, seed: int) -> dict:
    """one train/test split: discover q on train, then time AND score (test AUC)
    each variant. the fast variant uses a fixed mu (no grid); the full variant
    runs cv_pick_mu. timing is on the train split (~the deployment cost)."""
    args = _loader_args(sentinel_nan=(name == 'fico'))
    X_orig, names, y = load_dataset(name, args)
    n, p_orig = X_orig.shape

    (tr, te), = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    ).split(X_orig, (y > 0).astype(int))
    Xtr_o, ytr, Xte_o, yte = X_orig[tr], y[tr], X_orig[te], y[te]

    spec, _, parent = fit_binarizer(Xtr_o, names.tolist(), n_thresholds)
    Xtr, Xte = apply_binarizer(Xtr_o, spec), apply_binarizer(Xte_o, spec)
    p_bin = Xtr.shape[1]
    all_cols = np.arange(p_bin)

    mu_scale, mu_grid = make_mu_grid(Xtr, ytr, n_mu)
    mu_fast = mu_fast_rel * mu_scale

    FR = import_fasterrisk()
    rng = np.random.default_rng(seed)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # discovery: deployed iamb soft q (B subsamples on the train split)
        t0 = time.perf_counter()
        q = bnlearn_mb_stability_q(Xtr_o, ytr, test='mi-cg', alpha=alpha, B=b,
                                   rng=np.random.default_rng((seed, 1)))
        t_disc = time.perf_counter() - t0
        q_bin = q[parent]

        # vanilla: one fit at mu=0, scored on test
        t0 = time.perf_counter()
        van = fit_eval(FR, Xtr, ytr, Xte, yte, mu=0.0, q=None, k=k)
        t_van_fit = time.perf_counter() - t0

        # ours_fast: one fit at a fixed mu with the soft prior, scored on test
        t0 = time.perf_counter()
        fast = fit_eval(FR, Xtr, ytr, Xte, yte, mu=mu_fast, q=q_bin, k=k)
        t_fast_fit = time.perf_counter() - t0

        # ours_full: cv pick mu over the grid, then refit at mu_star and score
        t0 = time.perf_counter()
        cv = cv_pick_mu(Xtr, ytr, K=k, mu_grid=mu_grid, q=q_bin, n_splits=n_cv,
                        criterion='log_loss', rng=rng)
        t_mucv = time.perf_counter() - t0
        full = fit_eval(FR, Xtr, ytr, Xte, yte, mu=cv.mu_star, q=q_bin, k=k)

        # cfs hard: select a blanket, fit on those columns, scored on test
        t0 = time.perf_counter()
        mb = bnlearn_mb(Xtr_o, ytr, method='iamb', test='mi-cg', alpha=alpha)
        t_cfs_sel = time.perf_counter() - t0
        cols = all_cols[np.isin(parent, mb)] if mb else all_cols[:0]
        if len(cols):
            cfs = fit_eval(FR, Xtr[:, cols], ytr, Xte[:, cols], yte, mu=0.0, q=None, k=k)
            auc_cfs = cfs['auc']
        else:
            auc_cfs = float('nan')

    return {
        'dataset': name, 'n': n, 'p_orig': p_orig, 'p_bin': p_bin,
        't_discovery': t_disc, 't_cfs_select': t_cfs_sel,
        't_fit': t_fast_fit, 't_mu_cv': t_mucv, 'mu_star_rel': cv.mu_star / mu_scale,
        # derived per-deployment wall-clock
        't_vanilla': t_van_fit,
        't_ours_fast': t_disc + t_fast_fit,
        't_ours_full': t_disc + t_mucv + t_fast_fit,
        't_cfs_hard': t_cfs_sel + t_fast_fit,
        # test auc per variant
        'auc_vanilla': van['auc'], 'auc_ours_fast': fast['auc'],
        'auc_ours_full': full['auc'], 'auc_cfs_hard': auc_cfs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default=','.join(IN_REGIME),
                   help='comma-separated benchmark names')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--n-thresholds', type=int, default=4)
    p.add_argument('--b', type=int, default=50, help='iamb stability subsamples')
    p.add_argument('--n-mu', type=int, default=8)
    p.add_argument('--n-cv', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--mu-fast-rel', type=float, default=0.1,
                   help='fixed relative mu for the fast variant (no cv grid)')
    p.add_argument('--test-frac', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', type=Path,
                   default=ROOT / 'results' / 'causal_prior' / 'real' / 'runtime')
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    rows = []
    for name in datasets:
        print(f'timing {name}...', flush=True)
        t0 = time.perf_counter()
        row = bench_dataset(name, args.k, args.n_thresholds, args.b, args.n_mu,
                            args.n_cv, args.alpha, args.mu_fast_rel, args.test_frac,
                            args.seed)
        rows.append(row)
        print(f'  {name}: n={row["n"]} p_bin={row["p_bin"]} '
              f'discovery={row["t_discovery"]:.2f}s mu_cv={row["t_mu_cv"]:.2f}s '
              f'fit={row["t_fit"]:.3f}s | auc fast={row["auc_ours_fast"]:.3f} '
              f'full={row["auc_ours_full"]:.3f}  (wall {time.perf_counter()-t0:.1f}s)',
              flush=True)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / 'runtime.csv'
    df.to_csv(out, index=False)

    stages = df[['dataset', 'n', 'p_bin', 't_discovery', 't_cfs_select',
                 't_fit', 't_mu_cv', 'mu_star_rel']].round(3)
    wall = df[['dataset', 't_vanilla', 't_cfs_hard', 't_ours_fast',
               't_ours_full']].round(3)
    auc = df[['dataset', 'auc_vanilla', 'auc_cfs_hard', 'auc_ours_fast',
              'auc_ours_full']].round(4)
    print('\n== per-stage seconds ==\n' + stages.to_string(index=False), flush=True)
    print('\n== per-deployment wall-clock (s) ==\n' + wall.to_string(index=False), flush=True)
    print('\n== test auc ==\n' + auc.to_string(index=False), flush=True)
    speedup = (df['t_ours_full'] / df['t_ours_fast']).round(1)
    print(f'\nfast speedup over full (x): {speedup.tolist()}', flush=True)
    print(f'\nsaved {out}', flush=True)


if __name__ == '__main__':
    main()
