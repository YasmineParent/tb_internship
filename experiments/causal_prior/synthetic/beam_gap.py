"""Does FasterRisk's beam search find the exact MAP support?

Brute-forces the exact continuous MAP support at the p=30, K=2k* anchor, which is
out of reach for the p=12 enumeration in exact_radii.py but tractable on many
cores: ~5.3e7 supports at K=10, ~10 min/seed on 48 cores with the compact Newton
fit below (~0.55 ms/fit vs ~7.7 ms for sklearn, matched to ~1e-2).

For each support S with |S| <= K we compute the restricted continuous logistic
optimum ell(S) (L2 lambda=1, intercept unpenalised; the C_REG=1 convention of
exact_radii.py, a stand-in for FasterRisk's box). The exact MAP support for a
given (q, mu) is argmin_S [ell(S) - mu Q(S)], found by a streaming reduce over all
supports. We then fit FasterRisk on the same cell and compare its beam-selected
support to the exact MAP support:

  beam_match  : S_beam == S_exact
  jaccard     : |S_beam & S_exact| / |S_beam | S_exact|
  obj_gap     : F_q(S_beam) - F_q(S_exact) >= 0, how suboptimal the beam support is

ell(S) does not depend on q or mu, so all (q, mu) conditions share the one
expensive enumeration. Conditions: vanilla (mu=0), oracle and ges at each mu_rel.
mu is absolute = mu_rel * mu_scale, the same unit passed to FasterRisk.

Usage:
    python experiments/causal_prior/synthetic/beam_gap.py
    python experiments/causal_prior/synthetic/beam_gap.py --smoke
    python experiments/causal_prior/synthetic/beam_gap.py --p 30 --k 10 --n-seeds 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.synthetic_lingauss import LinGaussSyntheticData      # noqa: E402
from src.causal_prior.q_sources import oracle_q                    # noqa: E402
from src.causal_prior.priors import ges_stability_q                # noqa: E402
from experiments._io import new_run_dir                            # noqa: E402

OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'beam_gap'
LAM = 1.0                                  # L2 (matches exact_radii C_REG=1)
CSV_FIELDS = [
    'seed', 'p', 'n', 'k_star', 'p_edge', 'K', 'parent_size', 'q_name', 'mu_rel', 'mu',
    'k_exact', 'k_beam', 'beam_match', 'jaccard', 'map_in_pool', 'pool_size', 'obj_gap',
    'exact_is_Sstar', 'beam_recovers_Sstar', 'S_exact', 'S_beam',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--p', type=int, default=30, help='feature count')
    p.add_argument('--n', type=int, default=300, help='sample size')
    p.add_argument('--k-star', type=int, default=5, help='true sparsity')
    p.add_argument('--p-edge', type=float, default=0.2, help='DAG edge probability')
    p.add_argument('--k', type=int, default=8, help='sparsity budget')
    p.add_argument('--n-seeds', type=int, default=5)
    p.add_argument('--seed0', type=int, default=0, help='first seed')
    p.add_argument('--n-jobs', type=int, default=-1, help='joblib parallelism over chunks')
    p.add_argument('--chunk', type=int, default=60000, help='supports per enumeration chunk')
    p.add_argument('--b-disc', type=int, default=50, help='GES stability subsamples')
    p.add_argument('--mu-rel', type=str, default='0.5,2.0',
                   help='comma-separated mu_rel values for the prior conditions')
    p.add_argument('--parent-sizes', type=str, default='10',
                   help='comma-separated FasterRisk beam widths (parent_size) to sweep; '
                        'wider beam should close the gap to the exact MAP if it is a '
                        'search artifact (e.g. "10,25,50,100")')
    p.add_argument('--smoke', action='store_true',
                   help='shrink the enumeration for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.p, args.n, args.k, args.k_star = 16, 100, 4, 3
        args.n_seeds, args.b_disc = 1, 10
    args.mu_rel = tuple(float(x) for x in args.mu_rel.split(','))
    args.parent_sizes = tuple(int(x) for x in args.parent_sizes.split(','))
    return args


def fit_loss(Xs, y, lam=LAM, iters=30, tol=1e-10):
    """Restricted continuous logistic min: sum NLL + 0.5 lam ||w||^2, intercept free."""
    n, k = Xs.shape
    Z = np.empty((n, k + 1)); Z[:, 0] = 1.0; Z[:, 1:] = Xs
    b = np.zeros(k + 1)
    pen = np.full(k + 1, lam); pen[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ b)))
        g = Z.T @ (p - y) + pen * b
        W = p * (1.0 - p)
        H = (Z * W[:, None]).T @ Z
        H[np.diag_indices_from(H)] += pen
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        b -= step
        if np.max(np.abs(step)) < tol:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-(Z @ b))), 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def _chunks(p, K, size):
    it = itertools.chain.from_iterable(itertools.combinations(range(p), r)
                                       for r in range(K + 1))
    while True:
        block = list(itertools.islice(it, size))
        if not block:
            break
        yield block


def _process_chunk(block, X, y, conditions):
    """Per condition (name, q, mu): the (F, support) minimiser within this chunk."""
    best = {name: (np.inf, ()) for name, _, _ in conditions}
    for S in block:
        ell = fit_loss(X[:, S], y) if S else fit_loss(X[:, :0], y)
        for name, q, mu in conditions:
            Q = float(q[list(S)].sum()) if S else 0.0
            F = ell - mu * Q
            if F < best[name][0]:
                best[name] = (F, S)
    return best


def _fit_fr(X, y_signed, k, mu, q, parent_size):
    """fit fasterrisk and return (top-1 support, list of all pool-member supports).
    parent_size is the beam width; the pool is collectsparsediversepool's output."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
        fr = FasterRisk(k=k, mu=float(mu), parent_size=parent_size,
                        freq=None if q is None else q.astype(float))
        fr.fit(X, y_signed)
    pool = [frozenset(int(j) for j in np.where(np.abs(b) > 0)[0]) for b in fr.betas_]
    return pool[0], pool


def jaccard(a, b):
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def run_seed(seed, args):
    d = LinGaussSyntheticData(p=args.p, n_samples=args.n, k_star=args.k_star,
                              p_edge=args.p_edge, seed=seed)
    X = d.X
    y01 = (d.y > 0).astype(float)
    y_signed = d.y
    S_star = frozenset(int(j) for j in d.S_star)
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y_signed)))
    q_or = oracle_q(args.p, sorted(S_star))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q_ge, _ = ges_stability_q(X, d.y_continuous, B=args.b_disc,
                                  rng=np.random.default_rng(seed))

    # conditions: (name, q_vector, mu_absolute). vanilla = mu 0 (q inert).
    conditions = [('vanilla', q_or, 0.0)]
    for mr in args.mu_rel:
        conditions.append((f'oracle@{mr}', q_or, mr * mu_scale))
        conditions.append((f'ges@{mr}', q_ge, mr * mu_scale))

    t = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_process_chunk)(blk, X, y01, conditions)
        for blk in _chunks(args.p, args.k, args.chunk))
    best = {name: (np.inf, ()) for name, _, _ in conditions}
    for r in results:
        for name, (F, S) in r.items():
            if F < best[name][0]:
                best[name] = (F, S)
    enum_s = time.time() - t

    qmap = {'vanilla': (None, 0.0)}
    for mr in args.mu_rel:
        qmap[f'oracle@{mr}'] = (q_or, mr * mu_scale)
        qmap[f'ges@{mr}'] = (q_ge, mr * mu_scale)

    rows = []
    for name, q, mu in conditions:
        S_exact = frozenset(best[name][1])
        qv, _ = qmap[name]

        def F_of(S):
            ell = fit_loss(X[:, sorted(S)], y01) if S else fit_loss(X[:, :0], y01)
            Q = float(q[list(S)].sum()) if S else 0.0
            return ell - mu * Q
        mr = mu / mu_scale if mu_scale > 0 else 0.0
        for ps in args.parent_sizes:
            S_beam, pool = _fit_fr(X, y_signed, args.k, mu, qv, ps)
            map_in_pool = int(S_exact in pool)            # move 2: is the exact MAP in the pool at all?
            rows.append({
                'seed': seed, 'p': args.p, 'n': args.n, 'k_star': args.k_star,
                'p_edge': args.p_edge, 'K': args.k, 'parent_size': ps,
                'q_name': name, 'mu_rel': round(mr, 4), 'mu': mu,
                'k_exact': len(S_exact), 'k_beam': len(S_beam),
                'beam_match': int(S_beam == S_exact), 'jaccard': jaccard(S_beam, S_exact),
                'map_in_pool': map_in_pool, 'pool_size': len(set(pool)),
                'obj_gap': F_of(S_beam) - F_of(S_exact), 'exact_is_Sstar': int(S_exact == S_star),
                'beam_recovers_Sstar': int(S_star <= S_beam),
                'S_exact': json.dumps(sorted(S_exact)), 'S_beam': json.dumps(sorted(S_beam)),
            })
    print(f'  seed {seed}: enum {enum_s:.0f}s  '
          + '  '.join(f"{r['q_name']}@ps{r['parent_size']}:"
                      f"{'=' if r['beam_match'] else ('p' if r['map_in_pool'] else 'x')}"
                      for r in rows),
          flush=True)
    return rows


def main():
    args = parse_args()
    last = args.seed0 + args.n_seeds - 1
    print(f'beam_gap: p={args.p} n={args.n} k*={args.k_star} p_edge={args.p_edge} '
          f'K={args.k} seeds={args.seed0}..{last} mu_rel={args.mu_rel} '
          f'n_jobs={args.n_jobs}', flush=True)
    all_rows = []
    for seed in range(args.seed0, args.seed0 + args.n_seeds):
        all_rows.extend(run_seed(seed, args))
    df = pd.DataFrame(all_rows, columns=CSV_FIELDS)

    suffix = (f'beam_gap_p{args.p}_n{args.n}_k{args.k_star}_pedge{args.p_edge}'
              f'_K{args.k}_seeds{args.seed0}-{last}' + ('_smoke' if args.smoke else ''))
    output_dir = new_run_dir(OUT_DIR / suffix, vars(args))
    df.to_csv(output_dir / 'beam_gap.csv', index=False)

    summary = (df.groupby(['q_name', 'parent_size'], sort=False)
               .agg(match_pct=('beam_match', lambda s: 100 * s.mean()),
                    in_pool_pct=('map_in_pool', lambda s: 100 * s.mean()),
                    mean_jacc=('jaccard', 'mean'),
                    mean_gap=('obj_gap', 'mean'),
                    exact_is_Sstar_pct=('exact_is_Sstar', lambda s: 100 * s.mean())))
    summary.to_csv(output_dir / 'summary.csv')
    print('\nbeam (top-1) vs exact MAP support, plus is-MAP-in-pool, by condition:', flush=True)
    print(summary.to_string(), flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
