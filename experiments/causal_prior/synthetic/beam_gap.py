"""Caveat 1(i): does FasterRisk's beam search find the exact MAP support?

Brute-forces the exact continuous MAP support at the §6.1 anchor (p=30, K=2k*),
which is out of reach for the p=12 enumeration in exact_radii.py but tractable on
many cores: ~5.3e7 supports at K=10, ~10 min/seed on 48 cores with the compact
Newton fit below (~0.55 ms/fit vs ~7.7 ms for sklearn, matched to ~1e-2).

For each support S with |S| <= K we compute the restricted continuous logistic
optimum ell(S) (L2 lambda=1, intercept unpenalised; this is the C_REG=1 convention
of exact_radii.py, a stand-in for FasterRisk's box). The exact MAP support for a
given (q, mu) is argmin_S [ell(S) - mu Q(S)], found by a streaming reduce over all
supports. We then fit FasterRisk on the same cell and compare its beam-selected
support to the exact MAP support:

  beam_match  : S_beam == S_exact
  jaccard     : |S_beam & S_exact| / |S_beam | S_exact|
  obj_gap     : F_q(S_beam) - F_q(S_exact) >= 0, how suboptimal the beam support is

ell(S) does not depend on q or mu, so all (q, mu) conditions share the one
expensive enumeration. Conditions: vanilla (mu=0), oracle and ges at mu_rel in
{0.5, 2.0}. mu is absolute = mu_rel * mu_scale, the same unit passed to FasterRisk.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.synthetic_lingauss import LinGaussSyntheticData      # noqa: E402
from src.causal_prior.q_sources import oracle_q                    # noqa: E402
from src.causal_prior.priors import ges_stability_q                # noqa: E402

P = int(os.environ.get('P', '30'))
N = int(os.environ.get('N', '300'))
K_STAR = int(os.environ.get('K_STAR', '5'))
P_EDGE = float(os.environ.get('P_EDGE', '0.2'))
K = int(os.environ.get('K', '8'))
LAM = 1.0                                  # L2 (matches exact_radii C_REG=1)
N_SEEDS = int(os.environ.get('N_SEEDS', '5'))
SEED0 = int(os.environ.get('SEED0', '0'))
N_JOBS = int(os.environ.get('N_JOBS', '-1'))
CHUNK = int(os.environ.get('CHUNK', '60000'))
B_DISC = int(os.environ.get('B_DISC', '50'))
MU_REL = tuple(float(x) for x in os.environ.get('MU_REL', '0.5,2.0').split(','))

OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'beam_gap'
CSV_FIELDS = [
    'seed', 'p', 'n', 'k_star', 'p_edge', 'K', 'q_name', 'mu_rel', 'mu',
    'k_exact', 'k_beam', 'beam_match', 'jaccard', 'obj_gap',
    'exact_is_Sstar', 'beam_recovers_Sstar', 'S_exact', 'S_beam',
]


def fit_loss(Xs: np.ndarray, y: np.ndarray, lam: float = LAM,
             iters: int = 30, tol: float = 1e-10) -> float:
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


def _chunks(p: int, K: int, size: int):
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


def _fit_fr_support(X, y_signed, k, mu, q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
        fr = FasterRisk(k=k, mu=float(mu), freq=None if q is None else q.astype(float))
        fr.fit(X, y_signed)
    b = fr.betas_[0]
    return frozenset(int(j) for j in np.where(np.abs(b) > 0)[0])


def jaccard(a, b):
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def run_seed(seed: int) -> list[dict]:
    d = LinGaussSyntheticData(p=P, n_samples=N, k_star=K_STAR, p_edge=P_EDGE, seed=seed)
    X = d.X
    y01 = (d.y > 0).astype(float)
    y_signed = d.y
    S_star = frozenset(int(j) for j in d.S_star)
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y_signed)))
    q_or = oracle_q(P, sorted(S_star))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q_ge, _ = ges_stability_q(X, d.y_continuous, B=B_DISC,
                                  rng=np.random.default_rng(seed))

    # conditions: (name, q_vector, mu_absolute). vanilla = mu 0 (q inert).
    conditions = [('vanilla', q_or, 0.0)]
    for mr in MU_REL:
        conditions.append((f'oracle@{mr}', q_or, mr * mu_scale))
        conditions.append((f'ges@{mr}', q_ge, mr * mu_scale))

    t = time.time()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_process_chunk)(blk, X, y01, conditions) for blk in _chunks(P, K, CHUNK))
    best = {name: (np.inf, ()) for name, _, _ in conditions}
    for r in results:
        for name, (F, S) in r.items():
            if F < best[name][0]:
                best[name] = (F, S)
    enum_s = time.time() - t

    qmap = {'vanilla': (None, 0.0)}
    for mr in MU_REL:
        qmap[f'oracle@{mr}'] = (q_or, mr * mu_scale)
        qmap[f'ges@{mr}'] = (q_ge, mr * mu_scale)

    rows = []
    for name, q, mu in conditions:
        S_exact = frozenset(best[name][1])
        qv, _ = qmap[name]
        S_beam = _fit_fr_support(X, y_signed, K, mu, qv)
        # objective gap of the beam support under the same ell + q
        def F_of(S):
            ell = fit_loss(X[:, sorted(S)], y01) if S else fit_loss(X[:, :0], y01)
            Q = float(q[list(S)].sum()) if S else 0.0
            return ell - mu * Q
        gap = F_of(S_beam) - F_of(S_exact)
        mr = mu / mu_scale if mu_scale > 0 else 0.0
        rows.append({
            'seed': seed, 'p': P, 'n': N, 'k_star': K_STAR, 'p_edge': P_EDGE, 'K': K,
            'q_name': name, 'mu_rel': round(mr, 4), 'mu': mu,
            'k_exact': len(S_exact), 'k_beam': len(S_beam),
            'beam_match': int(S_beam == S_exact), 'jaccard': jaccard(S_beam, S_exact),
            'obj_gap': gap, 'exact_is_Sstar': int(S_exact == S_star),
            'beam_recovers_Sstar': int(S_star <= S_beam),
            'S_exact': json.dumps(sorted(S_exact)), 'S_beam': json.dumps(sorted(S_beam)),
        })
    print(f'  seed {seed}: enum {enum_s:.0f}s  '
          + '  '.join(f"{r['q_name']}:{'=' if r['beam_match'] else 'x'}" for r in rows),
          flush=True)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'beam_gap: p={P} n={N} k*={K_STAR} p_edge={P_EDGE} K={K} '
          f'seeds={SEED0}..{SEED0+N_SEEDS-1} mu_rel={MU_REL} n_jobs={N_JOBS}', flush=True)
    all_rows = []
    for seed in range(SEED0, SEED0 + N_SEEDS):
        all_rows.extend(run_seed(seed))

    out = OUT_DIR / f'beam_gap_p{P}_n{N}_k{K_STAR}_pedge{P_EDGE}_K{K}_seeds{SEED0}-{SEED0+N_SEEDS-1}.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(all_rows)
    print(f'wrote {out}', flush=True)

    names = []
    for r in all_rows:
        if r['q_name'] not in names:
            names.append(r['q_name'])
    print('\n=== beam vs exact MAP support, by condition ===', flush=True)
    print(f'{"condition":>12s} | {"match%":>6s} {"mean_jacc":>9s} {"mean_gap":>9s} '
          f'{"exact=S*":>8s}', flush=True)
    for nm in names:
        rs = [r for r in all_rows if r['q_name'] == nm]
        mt = 100 * np.mean([r['beam_match'] for r in rs])
        jc = np.mean([r['jaccard'] for r in rs])
        gp = np.mean([r['obj_gap'] for r in rs])
        es = 100 * np.mean([r['exact_is_Sstar'] for r in rs])
        print(f'{nm:>12s} | {mt:>5.0f}% {jc:>9.3f} {gp:>9.3f} {es:>7.0f}%', flush=True)


if __name__ == '__main__':
    main()
