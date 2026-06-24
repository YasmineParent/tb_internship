"""q-corruption graceful-degradation curve: the do-no-harm theorem as a figure.

interpolate q from a soft accurate prior (q_hi on S*, q_lo elsewhere) to per-feature
noise, q(a) = (1-a)*q_soft + a*noise, and at each corruption level a pick mu by cv,
refit, and score. the soft (not hard 0/1) start avoids the degenerate case where cv
distrusts an all-or-nothing prior and picks mu~0. the claim: as q degrades, support
recovery declines smoothly toward the no-information floor while held-out auc stays
flat throughout. so a bad prior costs accuracy nothing and recovery degrades
gracefully, never catastrophically.

Usage:
    python experiments/causal_prior/synthetic/q_corruption.py --smoke
    python experiments/causal_prior/synthetic/q_corruption.py --n-seeds 20 --n-levels 11
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.cv_mu import cv_pick_mu                               # noqa: E402
from src.causal_prior.metrics import support_recovery_metrics             # noqa: E402
from src.causal_prior.loading import causal_partition                     # noqa: E402
from experiments._io import new_run_dir                                    # noqa: E402

CACHE = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'cache_p30_headline'
OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'q_corruption'


def cell_rows(path, levels, n_mu, n_splits, soft_hi, soft_lo, start_source):
    cell = np.load(path, allow_pickle=False)
    p, k_star, seed = int(cell['p']), int(cell['k_star']), int(cell['seed'])
    X, y = cell['X'], cell['y']
    mu_scale = float(cell['mu_scale'])
    S_star = sorted(int(j) for j in cell['S_star'])
    confounded = sorted(int(j) for j in cell['confounded'])
    part = causal_partition(seed, p, float(cell['p_edge']), k_star)
    causes, correlates = part['all_causes'], part['correlates']
    # fixed mu_rel=1 (recovery is not a predictive quantity on balanced synthetic
    # data, so a CV-picked mu is arbitrary for it). a single-value grid makes
    # cv_pick_mu evaluate held-out AUC at the fixed mu without selecting it.
    mu_grid = np.array([1.0]) * mu_scale
    K = 2 * k_star

    # start q: an operational discovered prior (ges) degrades monotonically; the
    # soft-oracle start does not, because exact-S* over-constrains vs cv-log-loss.
    if start_source == 'ges' and 'q_ges' in cell.files:
        q_start = np.asarray(cell['q_ges'], float)
    else:
        q_start = np.full(p, soft_lo)
        q_start[S_star] = soft_hi
    noise = np.random.default_rng((seed, 99)).uniform(0, 1, size=p)  # fixed per seed
    rng = np.random.default_rng(seed)

    rows = []
    for a in levels:
        q = np.clip((1 - a) * q_start + a * noise, 0.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=q,
                            n_splits=n_splits, criterion='log_loss', rng=rng)
        m = support_recovery_metrics(cv.support, S_star, confounded,
                                     causes=causes, correlates=correlates)
        rows.append({'seed': seed, 'corruption': round(float(a), 3),
                     'mu_star_rel': round(cv.mu_star / mu_scale, 4) if mu_scale else 0.0,
                     'S_precision': m['S_precision'], 'S_recall': m['S_recall'],
                     'correlate_inclusion': m.get('correlate_inclusion', float('nan')),
                     'auc_star': cv.auc_star})
    print(f'  {path.name}: {len(rows)} levels', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=Path, default=CACHE)
    ap.add_argument('--cell-glob', default='seed*_p30_n300_k5_pedge0.2.npz')
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--n-levels', type=int, default=11)
    ap.add_argument('--n-mu', type=int, default=12)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--soft-hi', type=float, default=0.9, help='q on S* at corruption 0')
    ap.add_argument('--soft-lo', type=float, default=0.1, help='q off S* at corruption 0')
    ap.add_argument('--start-source', choices=['ges', 'soft_oracle'], default='ges',
                    help='prior at corruption 0 (ges = operational, degrades cleanly)')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.n_seeds, args.n_levels, args.n_mu, args.n_splits, args.n_jobs = 2, 5, 5, 3, 1

    import re
    def _seed(p):
        m = re.match(r'seed(\d+)_', p.name)
        return int(m.group(1)) if m else 1 << 30
    # numeric seed filter (seeds 0..n_seeds-1), matching q_robustness; a plain
    # sorted()[:n] truncates lexicographically (seed0, seed1, seed10, ...) and
    # picks a different seed pool than the robustness panel.
    cells = sorted((c for c in args.cache_dir.glob(args.cell_glob)
                    if _seed(c) < args.n_seeds), key=_seed)
    if not cells:
        ap.error(f'no cells matching {args.cell_glob!r}')
    levels = np.linspace(0.0, 1.0, args.n_levels)
    print(f'q-corruption: {len(cells)} seeds, {args.n_levels} levels', flush=True)
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(cell_rows)(p, levels, args.n_mu, args.n_splits,
                           args.soft_hi, args.soft_lo, args.start_source)
        for p in cells)
    df = pd.DataFrame([r for rows in results for r in rows])

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    out = new_run_dir(OUT_DIR / ('headline' + ('_smoke' if args.smoke else '')), cfg)
    df.to_csv(out / 'q_corruption.csv', index=False)

    g = df.groupby('corruption').agg(
        S_precision=('S_precision', 'mean'), S_prec_sem=('S_precision', 'sem'),
        recall=('S_recall', 'mean'),
        auc=('auc_star', 'mean'), auc_sem=('auc_star', 'sem'),
        mu_rel=('mu_star_rel', 'median')).reset_index()
    g.to_csv(out / 'summary.csv', index=False)

    # vanilla floor (uniform q == no prior) for reference
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(g['corruption'], g['S_precision'], yerr=g['S_prec_sem'],
                 marker='o', color='C0', label='S_precision (recovery)')
    ax1.set_xlabel('q corruption (0 = discovered prior, 1 = noise)')
    ax1.set_ylabel('S_precision', color='C0')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.errorbar(g['corruption'], g['auc'], yerr=g['auc_sem'],
                 marker='s', color='C3', label='held-out AUC')
    ax2.set_ylabel('AUC', color='C3')
    ax2.set_ylim(0.5, 1.0)
    ax1.set_title('Graceful degradation: recovery slides to the floor, AUC stays flat')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'q_corruption.png', dpi=130, bbox_inches='tight')

    print('\ncorruption summary:', flush=True)
    print(g.round(3).to_string(index=False), flush=True)
    print(f'AUC spread over corruption: {g["auc"].std():.4f} (flat = do-no-harm)', flush=True)
    print(f'Done. Results in {out}/', flush=True)


if __name__ == '__main__':
    main()
