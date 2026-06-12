"""Exact-MAP validation of the prior-perturbation bound (notes/perturbation_theorem.md).

FasterRisk is a beam-search heuristic, so it cannot cleanly exhibit a bound on
the exact combinatorial MAP. Here we compute the exact MAP by brute force on a
small problem: enumerate every support S with |S| <= K, fit the restricted
logistic loss l(S) once (l is independent of mu and q), then everything else is
exact arithmetic.

Test of Theorem 1 (tight form): the MAP support is invariant to q-perturbations
with ||q-q'||_inf < eps* = min_S G_q(S) / (mu |S delta S*|) (the code's eps_star).
This is the exact radius; the easy one-line bound eps_easy = Delta / (2 mu K) is
looser by 2K / |S delta S*|, which is bound slack, not a property of the MAP. We
verify eps_easy <= eps* at every mu, and that the ratio is the cardinality slack.
Theorem 2 (data radius): bootstrap MAP stability vs mu, with r_l = Delta/2.

Usage:
    python experiments/causal_prior/synthetic/exact_radii.py
    python experiments/causal_prior/synthetic/exact_radii.py --smoke
    python experiments/causal_prior/synthetic/exact_radii.py --p 12 --n 200 --p-edge 0.5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.synthetic_lingauss import LinGaussSyntheticData   # noqa: E402
from src.causal_prior.q_sources import oracle_q                 # noqa: E402
from experiments._io import new_run_dir                         # noqa: E402

OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'exact_radii'
C_REG = 1.0  # moderate L2: keeps l(S) non-degenerate (stand-in for FR's box constraint;
             # unconstrained fits separate on small supports and tie l(S) at ~0)
MU_REL_THM2 = (0.0, 0.05, 0.5, 5.0)  # incl mu=0 (vanilla) to show the data-stability gain


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--p', type=int, default=12, help='feature count')
    p.add_argument('--n', type=int, default=200, help='sample size')
    p.add_argument('--k-star', type=int, default=3, help='true sparsity')
    p.add_argument('--p-edge', type=float, default=0.5, help='DAG edge probability')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-mu', type=int, default=12, help='mu grid points (Theorem 1 sweep)')
    p.add_argument('--n-boot', type=int, default=20, help='bootstraps for the Theorem 2 check')
    p.add_argument('--smoke', action='store_true',
                   help='shrink the run for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.p, args.n, args.n_mu, args.n_boot = 10, 100, 4, 5
    args.k = args.k_star  # K = k_star, so the MAP can equal S* exactly
    return args


def restricted_losses(X, y01, K, c_reg):
    """l(S) for every support S with 1 <= |S| <= K. Tiny L2 ~ FR's ridge."""
    supports, losses = [], []
    for r in range(1, K + 1):
        for S in combinations(range(X.shape[1]), r):
            clf = LogisticRegression(C=c_reg, max_iter=2000)
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


def theorem1(supports, losses, Qvec, mu_grid, mu_scale, S_star, K):
    """Per-mu invariance radii: eps_easy (loose bound) vs eps_star (exact)."""
    rows, prev = [], None
    for mu, mrel in zip(mu_grid, mu_grid / mu_scale):
        idx, delta = map_and_gap(losses, Qvec, mu)
        Smap = supports[idx]
        eps_easy = delta / (2 * mu * K)   # loose one-line bound
        # tight invariance radius eps_star: smallest eps that can flip the MAP to some
        # competitor S2 by shifting q by +/-eps on the symmetric difference.
        F = losses - mu * Qvec
        Fmap = F[idx]
        eps_star = np.inf
        for t, S2 in enumerate(supports):
            if t == idx:
                continue
            sd = len(Smap ^ S2)
            if sd:
                eps_star = min(eps_star, (F[t] - Fmap) / (mu * sd))
        rows.append({'mu_rel': mrel, 'S_star_in_MAP': bool(frozenset(S_star) <= Smap),
                     'Delta': delta, 'eps_easy': eps_easy, 'eps_star': eps_star,
                     'slack': eps_star / eps_easy,
                     'transition': prev is not None and Smap != prev})
        prev = Smap
    return pd.DataFrame(rows)


def theorem2(X, y01, supports, losses, Qvec, mu_scale, S_star, rng, n_boot):
    """Bootstrap MAP stability vs mu; r_l = Delta/2 should never be violated."""
    boot_losses = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y01), len(y01))   # bootstrap resample
        _, lb = restricted_losses(X[ix], y01[ix], max(len(s) for s in supports), C_REG)
        boot_losses.append(lb)

    rows = []
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
            srec += (frozenset(S_star) <= Sb)
            if eta < r_l and Sb != S0:   # theorem says this can never happen
                viol += 1
        rows.append({'mu_rel': mrel, 'r_l': r_l, 'mean_eta': float(np.mean(etas)),
                     'map_stable': stable / n_boot, 'S_star_recov': srec / n_boot,
                     'violations': viol})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    d = LinGaussSyntheticData(p=args.p, n_samples=args.n, k_star=args.k_star,
                              p_edge=args.p_edge, seed=args.seed)
    X = d.X
    y01 = (d.y > 0).astype(int)
    q = oracle_q(args.p, d.S_star, sigma=0.0)

    supports, losses = restricted_losses(X, y01, args.k, C_REG)
    Qvec = np.array([sum(q[j] for j in S) for S in supports])
    mu_scale = float(np.median(0.5 * np.abs(X.T @ d.y)))
    mu_grid = np.logspace(-2, 1, args.n_mu) * mu_scale
    rng = np.random.default_rng(args.seed)
    print(f'p={args.p} n={args.n} k*={args.k_star} p_edge={args.p_edge} K={args.k}  '
          f'{len(supports)} supports  S*={sorted(d.S_star)}  mu_scale={mu_scale:.1f}',
          flush=True)

    thm1 = theorem1(supports, losses, Qvec, mu_grid, mu_scale, d.S_star, args.k)
    thm2 = theorem2(X, y01, supports, losses, Qvec, mu_scale, d.S_star, rng, args.n_boot)

    suffix = (f'exact_radii_p{args.p}_n{args.n}_k{args.k_star}_pedge{args.p_edge}'
              + ('_smoke' if args.smoke else ''))
    output_dir = new_run_dir(
        OUT_DIR / suffix,
        {**vars(args), 'c_reg': C_REG, 'mu_rel_thm2': list(MU_REL_THM2),
         'mu_scale': mu_scale, 'n_supports': len(supports),
         'S_star': sorted(int(j) for j in d.S_star)})
    thm1.to_csv(output_dir / 'theorem1_radii.csv', index=False)
    thm2.to_csv(output_dir / 'theorem2_data.csv', index=False)

    print('\nTheorem 1 (prior radius), per mu:', flush=True)
    print(thm1.to_string(index=False), flush=True)
    print('\nTheorem 2 (data radius), per mu:', flush=True)
    print(thm2.to_string(index=False), flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
