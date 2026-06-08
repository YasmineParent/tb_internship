"""Empirical test of the two perturbation radii (notes/perturbation_theorem.md).

On one synthetic cell, sweep mu and measure, at each mu:
  data_stab  = mean pairwise Jaccard of the support across CV folds
               (proxy for the data-perturbation radius eta*; from cv_pick_mu)
  prior_stab = mean Jaccard(support(q), support(q')) over random q-perturbations
               q' = clip(q + N(0, sigma), 0, 1)  (proxy for the prior radius eps*)
  trans      = '*' when the full-data support changed from the previous mu
               (a MAP support transition, where Delta(q) -> 0)

Predictions (corrected theory): above the binding threshold mu_0, prior_stab
falls as mu grows (explicit 1/mu in eps*); both stabilities dip at transitions;
the monotone object is the ratio, not either curve alone.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.synthetic_lingauss import LinGaussSyntheticData      # noqa: E402
from src.causal_prior.q_sources import oracle_q                    # noqa: E402
from src.causal_prior.cv_mu import cv_pick_mu                      # noqa: E402

import os
P = 30
N = int(os.environ.get('N', '300'))
K_STAR = 5
P_EDGE = float(os.environ.get('P_EDGE', '0.3'))
SEED = 0
K = 2 * K_STAR
N_MU = 10
SIGMA_PERT = 0.15      # std of the q-perturbation
N_PERT = 12            # perturbations per mu
N_SPLITS = 5


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
    d = LinGaussSyntheticData(p=P, n_samples=N, k_star=K_STAR, p_edge=P_EDGE, seed=SEED)
    X, y = d.X, d.y
    q = oracle_q(P, d.S_star, sigma=0.0)
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_rel = np.logspace(-2, 1, N_MU)
    mu_grid = mu_rel * mu_scale
    FR = _fr()
    rng = np.random.default_rng(0)

    # data radius: CV support stability across the mu grid (one call)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cv = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=q, n_splits=N_SPLITS,
                        rng=np.random.default_rng(0))
    data_stab = cv.stabilities_per_mu

    print(f'cell p={P} n={N} k*={K_STAR} p_edge={P_EDGE} K={K} '
          f'|S*|={len(d.S_star)} mu_scale={mu_scale:.2f}')
    print(f'{"mu_rel":>8s} | {"data_stab":>9s} {"prior_stab":>10s} | trans', flush=True)
    prev = None
    for mu, mrel, ds in zip(mu_grid, mu_rel, data_stab):
        S0 = support(FR, X, y, K, mu, q)
        js = []
        for _ in range(N_PERT):
            qp = np.clip(q + rng.normal(0, SIGMA_PERT, P), 0.0, 1.0)
            js.append(jaccard(S0, support(FR, X, y, K, mu, qp)))
        ps = float(np.mean(js))
        trans = '*' if (prev is not None and S0 != prev) else ' '
        print(f'{mrel:>8.3f} | {ds:>9.3f} {ps:>10.3f} |   {trans}', flush=True)
        prev = S0


if __name__ == '__main__':
    main()
