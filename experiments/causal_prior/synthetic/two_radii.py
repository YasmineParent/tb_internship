"""Empirical test of the two perturbation radii (notes/perturbation_theorem.md).

On one synthetic cell, sweep mu and measure, at each mu:
  data_stab  = mean pairwise Jaccard of the support across CV folds
               (proxy for the data-perturbation radius eta*; from cv_pick_mu)
  prior_stab = mean Jaccard(support(q), support(q')) over random q-perturbations
               q' = clip(q + N(0, sigma), 0, 1)  (proxy for the prior radius eps*)
  transition = True when the full-data support changed from the previous mu
               (a MAP support transition, where Delta(q) -> 0)

Prediction (corrected theory): above the binding threshold mu_0, prior_stab falls
as mu grows (explicit 1/mu in eps*); both stabilities dip at transitions; the
monotone object is the ratio, not either curve alone.

Usage:
    python experiments/causal_prior/synthetic/two_radii.py
    python experiments/causal_prior/synthetic/two_radii.py --smoke
    python experiments/causal_prior/synthetic/two_radii.py --p-edge 0.2 --n 500
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
from src.data.synthetic_lingauss import LinGaussSyntheticData      # noqa: E402
from src.causal_prior.q_sources import oracle_q                    # noqa: E402
from src.causal_prior.cv_mu import cv_pick_mu                      # noqa: E402
from experiments._io import new_run_dir                            # noqa: E402

OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'two_radii'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--p', type=int, default=30, help='feature count')
    p.add_argument('--n', type=int, default=300, help='sample size')
    p.add_argument('--k-star', type=int, default=5, help='true sparsity')
    p.add_argument('--p-edge', type=float, default=0.3, help='DAG edge probability')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-mu', type=int, default=10, help='mu grid points')
    p.add_argument('--sigma-pert', type=float, default=0.15, help='std of the q-perturbation')
    p.add_argument('--n-pert', type=int, default=12, help='q-perturbations per mu')
    p.add_argument('--n-splits', type=int, default=5, help='CV folds for data stability')
    p.add_argument('--smoke', action='store_true',
                   help='shrink the run for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.n, args.n_mu, args.n_pert, args.n_splits = 100, 4, 3, 3
    args.k = 2 * args.k_star
    return args


def _fr():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
    return FasterRisk


def support(FR, X, y, K, mu, q):
    fr = FR(k=K, mu=float(mu), freq=None if q is None else q.astype(float))
    fr.fit(X, y)
    b = fr.betas_[0]
    return frozenset(int(j) for j in np.where(np.abs(b) > 0)[0])


def jaccard(a, b):
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def main():
    args = parse_args()
    d = LinGaussSyntheticData(p=args.p, n_samples=args.n, k_star=args.k_star,
                              p_edge=args.p_edge, seed=args.seed)
    X, y = d.X, d.y
    q = oracle_q(args.p, d.S_star, sigma=0.0)
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_rel = np.logspace(-2, 1, args.n_mu)
    mu_grid = mu_rel * mu_scale
    FR = _fr()
    rng = np.random.default_rng(args.seed)

    # data radius: CV support stability across the mu grid (one call)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cv = cv_pick_mu(X, y, K=args.k, mu_grid=mu_grid, q=q, n_splits=args.n_splits,
                        rng=np.random.default_rng(args.seed))
    data_stab = cv.stabilities_per_mu

    print(f'cell p={args.p} n={args.n} k*={args.k_star} p_edge={args.p_edge} K={args.k} '
          f'|S*|={len(d.S_star)} mu_scale={mu_scale:.2f}', flush=True)
    rows, prev = [], None
    for mu, mrel, ds in zip(mu_grid, mu_rel, data_stab):
        S0 = support(FR, X, y, args.k, mu, q)
        js = []
        for _ in range(args.n_pert):
            qp = np.clip(q + rng.normal(0, args.sigma_pert, args.p), 0.0, 1.0)
            js.append(jaccard(S0, support(FR, X, y, args.k, mu, qp)))
        rows.append({'mu_rel': mrel, 'data_stab': ds, 'prior_stab': float(np.mean(js)),
                     'transition': prev is not None and S0 != prev})
        prev = S0
    df = pd.DataFrame(rows)

    suffix = (f'two_radii_p{args.p}_n{args.n}_k{args.k_star}_pedge{args.p_edge}'
              + ('_smoke' if args.smoke else ''))
    output_dir = new_run_dir(OUT_DIR / suffix,
                             {**vars(args), 'mu_scale': mu_scale,
                              'S_star': sorted(int(j) for j in d.S_star)})
    df.to_csv(output_dir / 'two_radii.csv', index=False)

    print(df.to_string(index=False), flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
