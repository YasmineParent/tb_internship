"""Exact-MAP validation of the prior-perturbation bound (notes/perturbation_theorem.md).

FasterRisk is a beam-search heuristic, so it cannot cleanly exhibit a bound on
the exact combinatorial MAP. Here we compute the exact MAP by brute force on a
small problem: enumerate every support S with |S| <= K, fit the restricted
logistic loss l(S) once (l is independent of mu and q), then everything else is
exact arithmetic.

Test of Theorem 1 (tight form): the MAP support is invariant to q-perturbations
with ||q-q'||_inf < eps* = min_S G_q(S) / (mu |S delta S*|) (the code's eps_adv).
This is the exact radius; the easy one-line bound eps_easy = Delta / (2 mu K)
(the code's eps_star column) is looser by 2K / |S delta S*|, which is bound slack,
not a property of the MAP. We verify eps_easy <= eps* at every mu, and that the
ratio is the cardinality slack. Also reports Delta(q) vs mu (non-monotone).
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
N_BOOT = 20                          # bootstraps for the Theorem 2 (data radius) check
MU_REL_THM2 = (0.0, 0.05, 0.5, 5.0)  # incl mu=0 (vanilla) to show the data-stability gain


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
    print(f'{"mu_rel":>8s} | {"S*<=MAP":>7s} {"Delta":>9s} {"eps_easy":>8s} {"eps*":>7s} '
          f'{"slack":>6s} | trans', flush=True)

    prev = None
    for mu, mrel in zip(mu_grid, mu_grid / mu_scale):
        idx, delta = map_and_gap(losses, Qvec, mu)
        Smap = supports[idx]
        eps_easy = delta / (2 * mu * K)   # loose one-line bound (Thm 1 remark)
        # tight invariance radius eps* (Thm 1): smallest eps that can flip the MAP to
        # some competitor S2 by shifting q by +/-eps on S2 \ Smap and Smap \ S2 (the
        # symmetric difference). Equals min_S G_q(S) / (mu |S delta S*|).
        F = losses - mu * Qvec
        Fmap = F[idx]
        eps_star = np.inf
        for t, S2 in enumerate(supports):
            if t == idx:
                continue
            sd = len(Smap ^ S2)
            if sd:
                eps_star = min(eps_star, (F[t] - Fmap) / (mu * sd))
        is_star = frozenset(d.S_star) <= Smap
        trans = '*' if (prev is not None and Smap != prev) else ' '
        print(f'{mrel:>8.3f} | {str(is_star):>7s} {delta:>9.3f} {eps_easy:>8.3f} '
              f'{eps_star:>7.3f} {eps_star/eps_easy:>6.2f} |   {trans}', flush=True)
        prev = Smap

    # --- Theorem 2 (data radius): bootstrap MAP stability vs mu ---
    # l_b(S) does not depend on mu, so compute each bootstrap's losses once and
    # evaluate all mu by arithmetic. r_l = Delta/2; the theorem guarantees the
    # MAP is preserved whenever the per-support loss shift eta < r_l.
    boot_losses = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, N, N)        # bootstrap resample (same n -> comparable scale)
        _, lb = restricted_losses(X[ix], y01[ix])
        boot_losses.append(lb)

    print(f'\n=== Theorem 2 (data radius) : {N_BOOT} bootstraps ===')
    print(f'{"mu_rel":>8s} | {"r_l=D/2":>8s} {"mean_eta":>8s} | '
          f'{"MAP-stable":>10s} {"S*-recov":>8s} {"viol":>5s}', flush=True)
    for mrel in MU_REL_THM2:
        mu = mrel * mu_scale
        idx, delta = map_and_gap(losses, Qvec, mu)
        S0 = supports[idx]
        r_l = delta / 2.0
        etas, stable, srec, viol = [], 0, 0, 0
        for lb in boot_losses:
            eta = float(np.max(np.abs(lb - losses)))
            Sb = supports[int(np.argmin(lb - mu * Qvec))]
            etas.append(eta)
            stable += (Sb == S0)
            srec += (frozenset(d.S_star) <= Sb)
            if eta < r_l and Sb != S0:     # theorem says this can never happen
                viol += 1
        print(f'{mrel:>8.3f} | {r_l:>8.2f} {np.mean(etas):>8.2f} | '
              f'{stable/N_BOOT:>10.0%} {srec/N_BOOT:>8.0%} {viol:>5d}', flush=True)


if __name__ == '__main__':
    main()
