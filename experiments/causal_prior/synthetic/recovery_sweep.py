"""Recovery sweep: fit FasterRisk over (q source, mu) per Phase A cell.

For each cell in the Phase A cache, derive the fast q sources (oracle,
uniform, adversarial) from S_star and confounded, take q_pc / q_ges /
q_bootstrap_l1 from the cell file, and fit FasterRisk(K, mu, freq=q) for
each (q source, mu) in the grid. Record support and S_star recovery
metrics. One CSV per cell; one row per fit. Crash-safe: skips cells
whose output already exists.

K is set per-cell as K = 2 * k_star to give vanilla room to fail (at
K = k_star vanilla saturates at recall=1.0 in the regime where causal
discovery is reliable). mu is reported both in raw units and as
mu_relative = mu / mu_scale, where mu_scale = median |grad_j L| at w=0
(computed in Phase A); cross-cell plots use mu_relative.

Sources: oracle (sigma=0), uniform (0.5), adversarial, the three
causal-discovery backends pc, ges, iamb (the source axis), and bootstrap_l1
(the predictive contrast for the §6.1 mediation story). All read from the
cell file as q_pc / q_ges / q_iamb / q_bootstrap_l1; missing ones are
skipped. No hard-pre-selection baseline: soft-vs-hard is a downstream
(real-data) point, not a synthetic-recovery one, since the soft prior's
purpose is to keep FasterRisk a single integrated step rather than win
recovery precision.

Usage:
    python experiments/causal_prior/synthetic/recovery_sweep.py            # default cache, parallel
    # --cache-dir / --out-dir take full paths, not bare names:
    python experiments/causal_prior/synthetic/recovery_sweep.py \
        --cache-dir results/causal_prior/synthetic/cache_p30_headline \
        --out-dir   results/causal_prior/synthetic/recovery_p30_headline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.causal_prior.q_sources import (  # noqa: E402
    oracle_q, uniform_q, adversarial_q,
)
from src.causal_prior.metrics import support_recovery_metrics  # noqa: E402
from src.causal_prior.loading import causal_partition  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / 'results' / 'causal_prior' / 'synthetic' / 'cache'
OUT_DIR = REPO_ROOT / 'results' / 'causal_prior' / 'synthetic' / 'recovery'
N_MU_LOG = 12          # log-spaced points in (mu_scale * 10^-2, mu_scale * 10^1)
K_MULTIPLIER = 2       # K = K_MULTIPLIER * k_star
SOURCES = ('oracle', 'uniform', 'adversarial', 'pc', 'ges', 'iamb', 'bootstrap_l1')

CSV_FIELDS = [
    'seed', 'p', 'n', 'k_star', 'p_edge', 'noise_scale', 'mu_scale',
    'q_source', 'mu', 'mu_relative', 'K',
    'support', 'k_actual',
    'S_recall', 'S_precision', 'C_inclusion',
    'causal_precision', 'correlate_inclusion',
    'fit_seconds',
]


def import_fasterrisk():
    """Defer FR import: it warns about R bindings on every fresh worker."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
    return FasterRisk


def build_q_sources(cell, p: int, S_star: list[int],
                    confounded: list[int]) -> dict[str, np.ndarray]:
    """Catalog of q vectors for the cell. Missing-in-file sources are dropped."""
    sources: dict[str, np.ndarray] = {
        'oracle':      oracle_q(p, S_star, sigma=0.0),
        'uniform':     uniform_q(p, 0.5),
        'adversarial': adversarial_q(p, confounded),
    }
    for key in ('q_pc', 'q_ges', 'q_iamb', 'q_bootstrap_l1'):
        if key in cell.files:
            sources[key.removeprefix('q_')] = cell[key]
    return sources


def fit_and_score(FasterRisk, X: np.ndarray, y: np.ndarray, K: int,
                  mu: float, q: np.ndarray | None) -> tuple[list[int], float]:
    # default select_top_m=50: smaller pool sizes change beam-search exploration and
    # produce different supports at mu=0; keep the pool full for consistent results
    t = time.time()
    fr = FasterRisk(k=K, mu=float(mu),
                    freq=q.astype(float) if q is not None else None)
    fr.fit(X, y)
    betas = fr.betas_[0]
    support = sorted(int(j) for j in np.where(np.abs(betas) > 0)[0])
    return support, time.time() - t


def process_cell(cell_path: Path, out_path: Path, n_mu_log: int) -> str:
    cell = np.load(cell_path, allow_pickle=False)
    p = int(cell['p'])
    k_star = int(cell['k_star'])
    K = K_MULTIPLIER * k_star
    mu_scale = float(cell['mu_scale'])
    # dense in [0.01, 1] where synthetic recovery peaks (mu_rel ~ 0.04), coarse
    # tail to 10x where the curve is flat. real-data runs use a wider grid; here
    # the action is all at low mu so the resolution goes there.
    mu_relative_grid = np.unique(np.concatenate([
        [0.0],
        np.logspace(-2, 0, n_mu_log),   # 0.01 .. 1, the peak region
        np.logspace(0, 1, 4),           # 1 .. 10, flat tail
    ]))
    mu_grid = mu_relative_grid * mu_scale

    X = cell['X']
    y = cell['y']
    S_star_list = sorted(int(j) for j in cell['S_star'])
    confounded_list = sorted(int(j) for j in cell['confounded'])
    S_set = set(S_star_list)
    C_set = set(confounded_list)
    # causal partition (all_causes = Anc(Y), correlates = non-causal subset of C)
    # for the cause-aware metrics; regenerated deterministically from cell params.
    part = causal_partition(int(cell['seed']), p, float(cell['p_edge']), k_star)
    causes, correlates = part['all_causes'], part['correlates']

    FasterRisk = import_fasterrisk()
    sources = build_q_sources(cell, p, S_star_list, confounded_list)

    rows: list[dict] = []
    base_id = {
        'seed': int(cell['seed']),
        'p': p,
        'n': int(cell['n']),
        'k_star': k_star,
        'p_edge': float(cell['p_edge']),
        'noise_scale': float(cell['noise_scale']) if 'noise_scale' in cell.files else 1.0,
        'mu_scale': mu_scale,
    }
    for q_source, q in sources.items():
        for mu, mu_rel in zip(mu_grid, mu_relative_grid):
            support, fit_seconds = fit_and_score(FasterRisk, X, y, K, mu, q)
            m = support_recovery_metrics(support, S_set, C_set,
                                         causes=causes, correlates=correlates)
            rows.append({**base_id,
                'q_source': q_source,
                'mu': float(mu), 'mu_relative': float(mu_rel),
                'K': K,
                'support': json.dumps(support), 'k_actual': m['k_actual'],
                'S_recall': m['S_recall'], 'S_precision': m['S_precision'],
                'C_inclusion': m['C_inclusion'],
                'causal_precision': m['causal_precision'],
                'correlate_inclusion': m['correlate_inclusion'],
                'fit_seconds': fit_seconds,
            })

    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return f'{cell_path.name}: {len(rows)} rows'


def _process_one_safe(cell_path: Path, out_dir: Path, force: bool,
                      n_mu_log: int) -> str:
    out_path = out_dir / (cell_path.stem + '.csv')
    if out_path.exists() and not force:
        return f'{cell_path.name}: skip (exists)'
    t = time.time()
    summary = process_cell(cell_path, out_path, n_mu_log)
    return f'{summary} in {time.time()-t:.1f}s'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, default=CACHE_DIR,
                        help='Phase A cache directory of .npz cells')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR,
                        help='output dir for per-cell recovery CSVs')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='joblib parallelism across cells (default: all cores)')
    parser.add_argument('--force', action='store_true',
                        help='recompute even if output CSV already exists')
    parser.add_argument('--n-mu-log', type=int, default=N_MU_LOG,
                        help=f'log-spaced mu grid points (default: {N_MU_LOG}); '
                             f'fewer = faster local iteration')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cell_paths = sorted(args.cache_dir.glob('*.npz'))
    if not cell_paths:
        parser.error(f'no .npz cells in {args.cache_dir}')
    print(f'cache={args.cache_dir}, {len(cell_paths)} cells, '
          f'n_jobs={args.n_jobs}, out={args.out_dir}', flush=True)

    # return_as='generator' streams completed results so we get per-cell progress;
    # default tuple-return buffers everything until the whole sweep finishes
    for m in Parallel(n_jobs=args.n_jobs, return_as='generator')(
        delayed(_process_one_safe)(p, args.out_dir, args.force, args.n_mu_log)
        for p in cell_paths
    ):
        print(f'  {m}', flush=True)
    print(f'Done. Results in {args.out_dir}/', flush=True)


if __name__ == '__main__':
    main()
