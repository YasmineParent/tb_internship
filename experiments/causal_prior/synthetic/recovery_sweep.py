"""§6.1 recovery sweep: fit FasterRisk over (q source, mu) per Phase A cell.

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

Sources: oracle (sigma=0), uniform (0.5), adversarial, pc, ges,
bootstrap_l1. Sources missing from the cell file (e.g. q_ges on dense
legacy cells) are skipped. In addition, the §6.4 hard-pre-selection
baseline is run for the causal/predictive sources (pc, ges,
bootstrap_l1) at thresholds {0.3, 0.5, 0.7}; these rows have
q_source = '<source>_hard_t<threshold>', mu = 0, K reduced to the
post-threshold feature count.

Usage:
    python experiments/causal_prior/synthetic/recovery_sweep.py            # default cache, parallel
    python experiments/causal_prior/synthetic/recovery_sweep.py --cache-dir cache_p30_headline --out-dir recovery_p30
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

from src_tb.support_recovery.q_sources import (  # noqa: E402
    oracle_q, uniform_q, adversarial_q,
)
from src_tb.support_recovery.metrics import support_recovery_metrics  # noqa: E402


CACHE_DIR = Path(__file__).parent / 'cache'
OUT_DIR = Path(__file__).parent / 'recovery'
N_MU_LOG = 12          # log-spaced points in (mu_scale * 10^-2, mu_scale * 10^1)
K_MULTIPLIER = 2       # K = K_MULTIPLIER * k_star
SOURCES = ('oracle', 'uniform', 'adversarial', 'pc', 'ges', 'bootstrap_l1')
HARD_THRESHOLDS = (0.3, 0.5, 0.7)  # MB-standard thresholds for the §6.4 hard-pre-selection baseline
HARD_BASELINE_SOURCES = ('pc', 'ges', 'bootstrap_l1')  # only causal/predictive sources, not the synthetic q's

CSV_FIELDS = [
    'seed', 'p', 'n', 'k_star', 'p_edge', 'mu_scale',
    'q_source', 'mu', 'mu_relative', 'K',
    'support', 'k_actual',
    'S_recall', 'S_precision', 'C_inclusion',
    'fit_seconds',
]


def _import_fasterrisk():
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
    for key in ('q_pc', 'q_ges', 'q_bootstrap_l1'):
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


def fit_hard_threshold(FasterRisk, X: np.ndarray, y: np.ndarray, K: int,
                       q: np.ndarray, t: float) -> tuple[list[int], int, float]:
    """Hard pre-selection baseline: restrict to features with q_j >= t, vanilla FR.

    Returns (support_global, K_effective, fit_seconds). K is capped at the
    pre-selected feature count; if the threshold leaves no features, returns
    an empty support immediately.
    """
    mask = q >= t
    if not mask.any():
        return [], 0, 0.0
    K_eff = min(K, int(mask.sum()))
    fr = FasterRisk(k=K_eff, mu=0.0, freq=None)
    start = time.time()
    fr.fit(X[:, mask], y)
    elapsed = time.time() - start
    betas = fr.betas_[0]
    selected_local = np.where(np.abs(betas) > 0)[0]
    global_indices = np.where(mask)[0][selected_local]
    return sorted(int(j) for j in global_indices), K_eff, elapsed


def process_cell(cell_path: Path, out_path: Path, n_mu_log: int) -> str:
    cell = np.load(cell_path, allow_pickle=False)
    p = int(cell['p'])
    k_star = int(cell['k_star'])
    K = K_MULTIPLIER * k_star
    mu_scale = float(cell['mu_scale'])
    mu_relative_grid = np.concatenate([[0.0], np.logspace(-2, 1, n_mu_log)])
    mu_grid = mu_relative_grid * mu_scale

    X = cell['X']
    y = cell['y']
    S_star_list = sorted(int(j) for j in cell['S_star'])
    confounded_list = sorted(int(j) for j in cell['confounded'])
    S_set = set(S_star_list)
    C_set = set(confounded_list)

    FasterRisk = _import_fasterrisk()
    sources = build_q_sources(cell, p, S_star_list, confounded_list)

    rows: list[dict] = []
    base_id = {
        'seed': int(cell['seed']),
        'p': p,
        'n': int(cell['n']),
        'k_star': k_star,
        'p_edge': float(cell['p_edge']),
        'mu_scale': mu_scale,
    }
    for q_source, q in sources.items():
        for mu, mu_rel in zip(mu_grid, mu_relative_grid):
            support, fit_seconds = fit_and_score(FasterRisk, X, y, K, mu, q)
            m = support_recovery_metrics(support, S_set, C_set)
            rows.append({**base_id,
                'q_source': q_source,
                'mu': float(mu), 'mu_relative': float(mu_rel),
                'K': K,
                'support': json.dumps(support), 'k_actual': m['k_actual'],
                'S_recall': m['S_recall'], 'S_precision': m['S_precision'],
                'C_inclusion': m['C_inclusion'],
                'fit_seconds': fit_seconds,
            })

    # §6.4 baseline: hard pre-selection by q-threshold + vanilla FasterRisk
    for q_name in HARD_BASELINE_SOURCES:
        if q_name not in sources:
            continue
        for t_thresh in HARD_THRESHOLDS:
            support, K_eff, fit_seconds = fit_hard_threshold(
                FasterRisk, X, y, K, sources[q_name], t_thresh,
            )
            m = support_recovery_metrics(support, S_set, C_set)
            rows.append({**base_id,
                'q_source': f'{q_name}_hard_t{t_thresh}',
                'mu': 0.0, 'mu_relative': 0.0,
                'K': K_eff,
                'support': json.dumps(support), 'k_actual': m['k_actual'],
                'S_recall': m['S_recall'], 'S_precision': m['S_precision'],
                'C_inclusion': m['C_inclusion'],
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


if __name__ == '__main__':
    main()
