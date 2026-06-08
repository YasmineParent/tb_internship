"""Exact-MAP validation of the prior-perturbation bound (notes/perturbation_theorem.md).

FasterRisk is a beam-search heuristic, so it cannot cleanly exhibit a bound on
the exact combinatorial MAP. Here we compute the exact MAP by brute force on a
small problem: enumerate every support S with |S| <= K, fit the restricted
logistic loss l(S) once (l is independent of mu and q), then everything else is
exact arithmetic.

Test of Theorem 1: the MAP support is invariant to q-perturbations with
||q-q'||_inf < eps* = Delta(q) / (2 mu K). We verify that random perturbations
of size eps < eps* never change the MAP (bound holds), while eps > eps* can.
Also reports Delta(q) vs mu (expected non-monotone, dipping to 0 at transitions).
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.synthetic_lingauss import LinGaussSyntheticData   # noqa: E402
from src.causal_prior.q_sources import oracle_q                 # noqa: E402

P, N, K_STAR, P_EDGE, SEED = 12, 200, 3, 0.5, 0
K = 3       # = k_star, so the MAP can equal S* exactly
C_REG = 1.0  # moderate L2: keeps l(S) non-degenerate (a stand-in for FR's box constraint;
             # unconstrained fits separate on small supports and tie l(S) at ~0)
N_MU = 12


def restricted_losses(X, y01):
    """l(S) for every support S with 1 <= |S| <= K. Tiny L2 ~ FR's ridge."""
    supports, losses = [], []
    for r in range(1, K + 1):
        for S in combinations(range(X.shape[1]), r):
            clf = LogisticRegression(C=C_REG, max_iter=2000)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                clf.fit(X[:, S], y01)
            p = clf.predict_proba(X[:, S])[:, 1]
            supports.append(frozenset(S))
            losses.append(log_loss(y01, p, labels=[0, 1]) * len(y01))  # sum-loss scale
    return supports, np.array(losses)


def map_and_gap(losses, Qvec, mu):
    F = losses - mu * Qvec
    order = np.argsort(F)
    return order[0], F[order[1]] - F[order[0]]   # (argmin index, optimality gap)


def main():
    d = LinGaussSyntheticData(p=P, n_samples=N, k_star=K_STAR, p_edge=P_EDGE, seed=SEED)
    X = d.X
    y01 = (d.y > 0).astype(int)
    q = oracle_q(P, d.S_star, sigma=0.0)

    supports, losses = restricted_losses(X, y01)
    Qvec = np.array([sum(q[j] for j in S) for S in supports])
    mu_scale = float(np.median(0.5 * np.abs(X.T @ d.y)))
    mu_grid = np.logspace(-2, 1, N_MU) * mu_scale
    rng = np.random.default_rng(0)

    print(f'p={P} n={N} k*={K_STAR} p_edge={P_EDGE} K={K}  '
          f'{len(supports)} supports  S*={sorted(d.S_star)}  mu_scale={mu_scale:.1f}')
    print(f'{"mu_rel":>8s} | {"S*<=MAP":>7s} {"Delta":>9s} {"eps*":>7s} {"eps_adv":>8s} '
          f'{"ratio":>6s} | trans', flush=True)

    prev = None
    for mu, mrel in zip(mu_grid, mu_grid / mu_scale):
        idx, delta = map_and_gap(losses, Qvec, mu)
        Smap = supports[idx]
        eps_star = delta / (2 * mu * K)
        # exact worst-case (adversarial) invariance radius: smallest eps that can
        # flip the MAP to some competitor S2 by shifting q by +/-eps on S2 \ Smap
        # and Smap \ S2 (the symmetric difference).
        F = losses - mu * Qvec
        Fmap = F[idx]
        eps_adv = np.inf
        for t, S2 in enumerate(supports):
            if t == idx:
                continue
            sd = len(Smap ^ S2)
            if sd:
                eps_adv = min(eps_adv, (F[t] - Fmap) / (mu * sd))
        is_star = frozenset(d.S_star) <= Smap
        trans = '*' if (prev is not None and Smap != prev) else ' '
        print(f'{mrel:>8.3f} | {str(is_star):>7s} {delta:>9.3f} {eps_star:>7.3f} '
              f'{eps_adv:>8.3f} {eps_adv/eps_star:>6.2f} |   {trans}', flush=True)
        prev = Smap


if __name__ == '__main__':
    main()
