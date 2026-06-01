"""Phase A of the §6.1 mechanism test: precompute stability-selection q vectors.

For each cell (seed, p_edge, p, n, k_star) defined by config.SWEEPS, generate
the linear-Gaussian-DAG synthetic and compute the slow bootstrap-stability q
sources (PC, GES, bootstrap-L1). Save as .npz files in cache/. Crash-safe:
skips cells whose output already exists; pass --force to recompute.

The fast q sources (oracle, uniform, adversarial) are derived in Phase B from
S_star and the confounded set, both saved here.

Usage:
    python experiments/causal_prior/synthetic/stability_selection.py            # all sweeps
    python experiments/causal_prior/synthetic/stability_selection.py --sweep p_edge
    python experiments/causal_prior/synthetic/stability_selection.py --sweep n --B 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.causal_prior.synthetic.config import (  # noqa: E402
    SWEEPS, DEFAULT_N_SEEDS, Cell, build_cells,
)
from src_tb.data.synthetic_lingauss import LinGaussSyntheticData  # noqa: E402
from src_tb.causal_discovery.priors import (  # noqa: E402
    pc_stability_q, ges_stability_q, bootstrap_l1_q,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / 'results' / 'causal_prior' / 'synthetic' / 'cache'


def compute_mu_scale(X: np.ndarray, y: np.ndarray) -> float:
    """median |grad_j L| at w=0 for the logistic loss with y in {-1, +1}.

    At w=0 the gradient simplifies to -0.5 * (X^T y); we report its median
    absolute value as the data-relative scale for mu (pipeline.ipynb §2.4).
    """
    return float(np.median(0.5 * np.abs(X.T @ y)))


SOURCES = ('pc', 'ges', 'bootstrap_l1')


def process_cell(cell: Cell, B: int, cache_dir: Path, force: bool,
                 sources: tuple[str, ...]) -> None:
    out_path = cache_dir / cell.filename
    if out_path.exists() and not force:
        print(f'  skip {cell.filename} (exists)', flush=True)
        return

    t_total = time.time()
    data = LinGaussSyntheticData(
        p=cell.p, n_samples=cell.n, k_star=cell.k_star,
        p_edge=cell.p_edge, seed=cell.seed,
    )
    mu_scale = compute_mu_scale(data.X, data.y)

    # derive per-source RNGs from the cell seed so reruns are reproducible
    rng_pc  = np.random.default_rng((cell.seed, 0))
    rng_ges = np.random.default_rng((cell.seed, 1))
    rng_l1  = np.random.default_rng((cell.seed, 2))

    qs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    ges_timeouts = 0
    if 'pc' in sources:
        t = time.time(); qs['q_pc']  = pc_stability_q(data.X, data.y_continuous, B=B, rng=rng_pc); timings['pc']  = time.time() - t
    if 'ges' in sources:
        t = time.time()
        qs['q_ges'], ges_timeouts = ges_stability_q(data.X, data.y_continuous, B=B, rng=rng_ges)
        timings['ges'] = time.time() - t
    if 'bootstrap_l1' in sources:
        t = time.time(); qs['q_bootstrap_l1'] = bootstrap_l1_q(data.X, data.y, B=B, rng=rng_l1); timings['l1']  = time.time() - t

    np.savez(
        out_path,
        seed=cell.seed,
        p_edge=cell.p_edge,
        p=cell.p,
        n=cell.n,
        k_star=cell.k_star,
        X=data.X,
        y=data.y,
        y_continuous=data.y_continuous,
        S_star=np.array(sorted(data.S_star), dtype=int),
        confounded=np.array(sorted(data.confounded), dtype=int),
        mu_scale=mu_scale,
        ges_n_timeouts=ges_timeouts,
        **qs,
    )
    timing_str = ', '.join(f'{name} {t:.1f}s' for name, t in timings.items())
    extra = f' [ges timeouts: {ges_timeouts}/{B}]' if 'ges' in sources and ges_timeouts else ''
    print(f'  saved {cell.filename}: total {time.time()-t_total:.1f}s '
          f'({timing_str}){extra}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', choices=list(SWEEPS) + ['all'], default='all',
                        help='which sweep to populate (default: all)')
    parser.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument('--cache-dir', type=Path, default=CACHE_DIR)
    parser.add_argument('--B', type=int, default=100,
                        help='bootstrap count for stability sources')
    parser.add_argument('--sources', type=str, default=','.join(SOURCES),
                        help=('comma-separated subset of stability sources to compute. '
                              f'choices: {SOURCES}; default: all'))
    parser.add_argument('--force', action='store_true',
                        help='recompute even if cache file already exists')
    parser.add_argument('--p', type=int, help='override DEFAULTS p (feature count)')
    parser.add_argument('--n', type=int, help='override DEFAULTS n (sample size)')
    parser.add_argument('--k-star', dest='k_star', type=int,
                        help='override DEFAULTS k_star (true sparsity)')
    args = parser.parse_args()

    sources = tuple(s.strip() for s in args.sources.split(',') if s.strip())
    unknown = set(sources) - set(SOURCES)
    if unknown:
        parser.error(f'unknown source(s) {unknown}; choose from {SOURCES}')

    overrides = {k: v for k, v in {'p': args.p, 'n': args.n, 'k_star': args.k_star}.items()
                 if v is not None}

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells(args.sweep, n_seeds=args.n_seeds, overrides=overrides)
    print(f'sweep={args.sweep}, overrides={overrides or "{}"}, '
          f'{len(cells)} cells, B={args.B}, '
          f'sources={sources}, cache={args.cache_dir}', flush=True)

    for i, cell in enumerate(cells, 1):
        print(f'[{i}/{len(cells)}] {cell.filename}', flush=True)
        process_cell(cell, B=args.B, cache_dir=args.cache_dir, force=args.force,
                     sources=sources)


if __name__ == '__main__':
    main()
