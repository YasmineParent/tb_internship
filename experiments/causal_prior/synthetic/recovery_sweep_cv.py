"""§6.1 recovery sweep with CV-on-log_loss mu selection.

For each Phase A cell and each q source, pick mu by 5-fold CV-on-log_loss
across the same log-spaced mu grid used by recovery_sweep.py, then refit
on full data at the chosen mu_star. Record the resulting support and S*
recovery metrics. This is the principled headline path (mu chosen by a
procedure a practitioner could run); the existing recovery_sweep.py
covers the oracle-mu baseline for the gap figure (post-hoc argmax over
its full mu x metric grid). AUC is also recorded for diagnostic use but
is not the selection criterion (AUC is essentially mu-invariant for FR's
integer betas; log_loss has real but small sensitivity to mu).

K-ablation: pass --K-multipliers '1.0,1.5,2.0,3.0' to evaluate the prior at
several cardinality budgets per cell (one row per K_multiplier per cell per
q_source). Default is just K_multiplier=2.0 (the headline choice).

Crash-safe: skips cells whose output already exists; pass --force to
recompute.
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
from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / 'results' / 'causal_prior' / 'synthetic' / 'cache'
OUT_DIR = REPO_ROOT / 'results' / 'causal_prior' / 'synthetic' / 'recovery_cv'
N_MU_LOG = 12
DEFAULT_K_MULTIPLIERS = (2.0,)
N_SPLITS = 5
SOURCES = ('oracle', 'uniform', 'adversarial', 'pc', 'ges', 'bootstrap_l1')

CSV_FIELDS = [
    'seed', 'p', 'n', 'k_star', 'p_edge', 'noise_scale', 'mu_scale',
    'q_source', 'K_multiplier', 'K',
    'mu_star', 'mu_star_relative', 'log_loss_star', 'auc_star',
    'support', 'k_actual',
    'S_recall', 'S_precision', 'C_inclusion',
    'fit_seconds',
]


def build_q_sources(cell, p: int, S_star: list[int],
                    confounded: list[int]) -> dict[str, np.ndarray]:
    """Same catalog as recovery_sweep.build_q_sources; missing sources dropped."""
    sources: dict[str, np.ndarray] = {
        'oracle':      oracle_q(p, S_star, sigma=0.0),
        'uniform':     uniform_q(p, 0.5),
        'adversarial': adversarial_q(p, confounded),
    }
    for key in ('q_pc', 'q_ges', 'q_bootstrap_l1'):
        if key in cell.files:
            sources[key.removeprefix('q_')] = cell[key]
    return sources


def process_cell(cell_path: Path, out_path: Path, n_mu_log: int,
                 k_multipliers: tuple[float, ...], n_splits: int) -> str:
    cell = np.load(cell_path, allow_pickle=False)
    p = int(cell['p'])
    k_star = int(cell['k_star'])
    mu_scale = float(cell['mu_scale'])
    mu_relative_grid = np.concatenate([[0.0], np.logspace(-2, 1, n_mu_log)])
    mu_grid = mu_relative_grid * mu_scale

    X = cell['X']
    y = cell['y']
    S_star_list = sorted(int(j) for j in cell['S_star'])
    confounded_list = sorted(int(j) for j in cell['confounded'])
    S_set = set(S_star_list)
    C_set = set(confounded_list)

    sources = build_q_sources(cell, p, S_star_list, confounded_list)

    # one rng per cell drives the CV fold-shuffle seed (deterministic given cell)
    rng = np.random.default_rng(int(cell['seed']))

    rows: list[dict] = []
    base_id = {
        'seed': int(cell['seed']), 'p': p, 'n': int(cell['n']),
        'k_star': k_star, 'p_edge': float(cell['p_edge']),
        'noise_scale': float(cell['noise_scale']) if 'noise_scale' in cell.files else 1.0,
        'mu_scale': mu_scale,
    }
    for K_mult in k_multipliers:
        K = max(1, int(round(K_mult * k_star)))
        for q_source, q in sources.items():
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cv = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=q,
                                n_splits=n_splits, rng=rng)
            m = support_recovery_metrics(cv.support, S_set, C_set)
            mu_rel = cv.mu_star / mu_scale if mu_scale > 0 else 0.0
            rows.append({**base_id,
                'q_source': q_source,
                'K_multiplier': K_mult, 'K': K,
                'mu_star': cv.mu_star, 'mu_star_relative': mu_rel,
                'log_loss_star': cv.log_loss_star, 'auc_star': cv.auc_star,
                'support': json.dumps(cv.support),
                'k_actual': m['k_actual'],
                'S_recall': m['S_recall'], 'S_precision': m['S_precision'],
                'C_inclusion': m['C_inclusion'],
                'fit_seconds': time.time() - t,
            })

    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return f'{cell_path.name}: {len(rows)} rows'


def _process_one_safe(cell_path: Path, out_dir: Path, force: bool,
                      n_mu_log: int, k_multipliers: tuple[float, ...],
                      n_splits: int) -> str:
    out_path = out_dir / (cell_path.stem + '.csv')
    if out_path.exists() and not force:
        return f'{cell_path.name}: skip (exists)'
    t = time.time()
    summary = process_cell(cell_path, out_path, n_mu_log, k_multipliers, n_splits)
    return f'{summary} in {time.time()-t:.1f}s'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, default=CACHE_DIR)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--n-mu-log', type=int, default=N_MU_LOG,
                        help='log-spaced mu grid points (default 12)')
    parser.add_argument('--n-splits', type=int, default=N_SPLITS,
                        help='CV folds (default 5)')
    parser.add_argument('--K-multipliers', type=str,
                        default=','.join(str(k) for k in DEFAULT_K_MULTIPLIERS),
                        help='comma-separated K = K_mult * k_star values '
                             "(e.g. '2.0' or '1.0,1.5,2.0,3.0' for K-ablation)")
    parser.add_argument('--cell-filter', type=str, default=None,
                        help='optional glob matched against cell filenames '
                             "(e.g. 'seed*_p30_n300_k5_pedge0.2.npz' for the anchor only)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_multipliers = tuple(float(x) for x in args.K_multipliers.split(','))
    pattern = args.cell_filter or '*.npz'
    cell_paths = sorted(args.cache_dir.glob(pattern))
    if not cell_paths:
        parser.error(f'no .npz cells matching {pattern!r} in {args.cache_dir}')
    print(f'cache={args.cache_dir}, {len(cell_paths)} cells, '
          f'K_mults={k_multipliers}, n_splits={args.n_splits}, '
          f'n_jobs={args.n_jobs}, out={args.out_dir}', flush=True)

    for m in Parallel(n_jobs=args.n_jobs, return_as='generator')(
        delayed(_process_one_safe)(
            p, args.out_dir, args.force, args.n_mu_log, k_multipliers, args.n_splits,
        )
        for p in cell_paths
    ):
        print(f'  {m}', flush=True)


if __name__ == '__main__':
    main()
