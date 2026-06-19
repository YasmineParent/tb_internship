"""Exact integer-MAP validation: the radii and the beam gap on FasterRisk's actual
(integer) objective, without a MIP solver.

RiskSLIM would give a certified integer optimum but needs CPLEX. At small scale we
get the exact integer MAP by brute force over both supports and integer coefficients,
which is provably optimal and has no solver dependency. For each support S with
|S| <= K we compute

    ell_int(S) = min over integer w in [-C,C]^|S| and integer intercept w0 in [-C0,C0]
                 of the summed logistic loss,

and the exact integer MAP for (q, mu) is argmin_S [ell_int(S) - mu Q(S)]. Because the
radii proofs use only that ell(S) is a fixed per-support number independent of q, they
apply verbatim to ell_int; this script validates them on the integer objective and,
crucially, measures the beam gap against the right reference:

  - exact_is_Sstar      : the exact integer MAP recovers the planted S*
  - beam_match          : FasterRisk top-1 support == exact integer MAP support
  - map_in_pool         : exact integer MAP support is in FasterRisk's diverse pool
  - mu0                 : separation threshold (Lemma 1) on the integer objective

This isolates beam-search suboptimality from the continuous-vs-integer objective
mismatch that confounds beam_gap.py.

Usage:
    python experiments/causal_prior/synthetic/integer_radii.py --smoke
    python experiments/causal_prior/synthetic/integer_radii.py --p 12 --k-star 3 --C 5
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.synthetic_lingauss import LinGaussSyntheticData   # noqa: E402
from src.causal_prior.q_sources import oracle_q                 # noqa: E402
from src.causal_prior.priors import ges_stability_q             # noqa: E402
from experiments._io import new_run_dir                         # noqa: E402

OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'integer_radii'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--p', type=int, default=12, help='feature count')
    p.add_argument('--n', type=int, default=200, help='sample size')
    p.add_argument('--k-star', type=int, default=3, help='true sparsity')
    p.add_argument('--p-edge', type=float, default=0.5, help='DAG edge probability')
    p.add_argument('--C', type=int, default=5, help='integer coefficient bound (matches FR lb/ub)')
    p.add_argument('--C0', type=int, default=8, help='integer intercept bound')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--b-disc', type=int, default=50, help='GES stability subsamples')
    p.add_argument('--mu-rel', type=str, default='0.5,2.0',
                   help='comma-separated mu_rel values for the prior conditions')
    p.add_argument('--parent-size', type=int, default=10, help='FasterRisk beam width')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.p, args.n, args.k_star, args.C, args.C0 = 10, 100, 2, 3, 5
        args.b_disc = 10
    args.k = args.k_star                     # K = k*, so the exact MAP can equal S*
    args.mu_rel = tuple(float(x) for x in args.mu_rel.split(','))
    return args


def ell_int(Xs, y_signed, C, C0):
    """exact integer-restricted minimum of the summed logistic loss on this support.
    brute force over integer coefficients in [-C,C]^d and integer intercept in [-C0,C0]."""
    n, d = Xs.shape
    if d == 0:
        w0 = np.arange(-C0, C0 + 1)[None, :]                       # (1, J)
        return float(np.logaddexp(0.0, -(y_signed[:, None] * w0)).sum(0).min())
    grid = np.array(list(product(range(-C, C + 1), repeat=d)))     # (M, d)
    margins = Xs @ grid.T                                          # (n, M)
    best = np.inf
    for w0 in range(-C0, C0 + 1):
        L = np.logaddexp(0.0, -(y_signed[:, None] * (margins + w0))).sum(0)  # (M,)
        best = min(best, float(L.min()))
    return best


def all_restricted_losses(X, y_signed, K, C, C0):
    """ell_int(S) for every support with 1 <= |S| <= K, as a {frozenset: loss} map."""
    out = {}
    for r in range(1, K + 1):
        for S in combinations(range(X.shape[1]), r):
            out[frozenset(S)] = ell_int(X[:, S], y_signed, C, C0)
    return out


def map_support(losses, q, mu, p):
    """exact integer MAP support = argmin_S [ell_int(S) - mu Q(S)]."""
    best_S, best_F = None, np.inf
    for S, ell in losses.items():
        F = ell - mu * float(q[list(S)].sum())
        if F < best_F:
            best_F, best_S = F, S
    return best_S


def mu0_threshold(losses, q):
    """Lemma 1 separation threshold: smallest mu at which the MAP leaves the loss
    optimum S_loss, = min over competitors with Q(S) > Q(S_loss) of the line crossing."""
    S_loss = min(losses, key=losses.get)
    Q_loss = float(q[list(S_loss)].sum())
    ell_loss = losses[S_loss]
    mu0 = np.inf
    for S, ell in losses.items():
        Q = float(q[list(S)].sum())
        if Q > Q_loss:
            mu0 = min(mu0, (ell - ell_loss) / (Q - Q_loss))
    return float(mu0), S_loss


def thm1_radius(losses, q, mu, S_map):
    """Theorem 1 on the integer objective: tight prior radius eps* and a direct check
    that the worst-case q-perturbation just below eps* leaves the integer MAP fixed
    (radius_holds), and just above flips it (radius_tight; box clipping at q in {0,1}
    can legitimately prevent the flip, then eps* is only a lower bound)."""
    if mu <= 0:
        return np.inf, 1, 1
    F_map = losses[S_map] - mu * float(q[list(S_map)].sum())
    eps_star, binding = np.inf, None
    for S, ell in losses.items():
        if S == S_map:
            continue
        G = (ell - mu * float(q[list(S)].sum())) - F_map      # >= 0 since S_map is MAP
        r = G / (mu * len(S ^ S_map))
        if r < eps_star:
            eps_star, binding = r, S
    if binding is None:
        return np.inf, 1, 1

    def map_at(qp):
        return min(losses, key=lambda S: losses[S] - mu * float(qp[list(S)].sum()))

    def adversary(eps):
        qp = q.copy()
        for j in binding - S_map:
            qp[j] = min(1.0, qp[j] + eps)
        for j in S_map - binding:
            qp[j] = max(0.0, qp[j] - eps)
        return qp

    holds = int(map_at(adversary(0.99 * eps_star)) == S_map)
    tight = int(map_at(adversary(1.01 * eps_star)) != S_map)
    return eps_star, holds, tight


def fit_fr(X, y_signed, k, mu, q, parent_size):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
        fr = FasterRisk(k=k, mu=float(mu), parent_size=parent_size,
                        freq=None if q is None else q.astype(float))
        fr.fit(X, y_signed)
    pool = [frozenset(int(j) for j in np.where(np.abs(b) > 0)[0]) for b in fr.betas_]
    return pool[0], pool


def main():
    args = parse_args()
    d = LinGaussSyntheticData(p=args.p, n_samples=args.n, k_star=args.k_star,
                              p_edge=args.p_edge, seed=args.seed)
    X, y_signed = d.X, d.y
    S_star = frozenset(int(j) for j in d.S_star)
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y_signed)))
    q_or = oracle_q(args.p, sorted(S_star))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q_ge, _ = ges_stability_q(X, d.y_continuous, B=args.b_disc,
                                  rng=np.random.default_rng(args.seed))

    print(f'integer_radii: p={args.p} n={args.n} k*={args.k_star} K={args.k} '
          f'C={args.C} C0={args.C0} seed={args.seed}', flush=True)
    print(f'enumerating exact integer ell on {sum(1 for r in range(1, args.k+1) for _ in combinations(range(args.p), r))} supports...', flush=True)
    losses = all_restricted_losses(X, y_signed, args.k, args.C, args.C0)

    conditions = [('vanilla', q_or, 0.0)]
    for mr in args.mu_rel:
        conditions += [(f'oracle@{mr}', q_or, mr * mu_scale),
                       (f'ges@{mr}', q_ge, mr * mu_scale)]

    rows = []
    for name, q, mu in conditions:
        S_map = map_support(losses, q, mu, args.p)
        qv = None if name == 'vanilla' else q
        S_beam, pool = fit_fr(X, y_signed, args.k, mu, qv, args.parent_size)
        mu0, S_loss = mu0_threshold(losses, q)
        eps_star, radius_holds, radius_tight = thm1_radius(losses, q, mu, S_map)
        inter = len(S_map & S_beam)
        rows.append({
            'name': name, 'mu_rel': round(mu / mu_scale if mu_scale else 0.0, 4),
            'k_map': len(S_map), 'k_beam': len(S_beam),
            'exact_is_Sstar': int(S_map == S_star),
            # theory check on the integer objective (no FasterRisk):
            'eps_star_rel': round(eps_star / mu_scale, 4) if np.isfinite(eps_star) and mu_scale else eps_star,
            'radius_holds': radius_holds, 'radius_tight': radius_tight,
            'mu0_rel': round(mu0 / mu_scale, 4) if np.isfinite(mu0) and mu_scale else mu0,
            # beam-gap measurement vs the exact integer MAP:
            'beam_match': int(S_beam == S_map), 'map_in_pool': int(S_map in pool),
            'beam_recovers_Sstar': int(S_star <= S_beam),
            'jaccard': round(inter / len(S_map | S_beam), 3) if (S_map | S_beam) else 1.0,
            'S_map': json.dumps(sorted(S_map)), 'S_beam': json.dumps(sorted(S_beam)),
        })

    df = pd.DataFrame(rows)
    out = new_run_dir(OUT_DIR / (f'int_p{args.p}_k{args.k_star}_C{args.C}_seed{args.seed}'
                                 + ('_smoke' if args.smoke else '')), vars(args))
    df.to_csv(out / 'integer_radii.csv', index=False)
    print('\n[theory] radii on the exact integer MAP (no FasterRisk):', flush=True)
    print(df[['name', 'mu_rel', 'exact_is_Sstar', 'eps_star_rel', 'radius_holds',
              'radius_tight', 'mu0_rel']].to_string(index=False), flush=True)
    print('\n[beam gap] FasterRisk vs the exact integer MAP:', flush=True)
    show = df[['name', 'mu_rel', 'beam_match', 'map_in_pool', 'beam_recovers_Sstar', 'jaccard']]
    print(show.to_string(index=False), flush=True)
    print(f'\nS* = {sorted(S_star)} ; loss-optimal support S_loss = {sorted(min(losses, key=losses.get))}', flush=True)
    print(f'Done. Results in {out}/', flush=True)


if __name__ == '__main__':
    main()
