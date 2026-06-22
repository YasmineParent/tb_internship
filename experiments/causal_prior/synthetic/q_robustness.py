"""q-source robustness: does the soft prior work regardless of which discovery
method supplies q?

q sources span the well-known paradigms: controls (oracle, uniform, adversarial),
pc (constraint), ges (score), bootstrap_l1 (stability), and the markov-blanket
family q actually targets, iamb (classic), hiton_mb (canonical), fbed (modern).
per source we pick mu by the recovery-sweep cv, refit, and score the support.
pc/ges/bootstrap_l1/iamb are read from the cached cell; hiton_mb/fbed are computed
here. the claim: auc_star stays flat across sources (do-no-harm) while support
quality tracks q quality, and no source breaks the solver.

Usage:
    python experiments/causal_prior/synthetic/q_robustness.py --smoke
    python experiments/causal_prior/synthetic/q_robustness.py --n-seeds 20 --B 50
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.q_sources import oracle_q, uniform_q, adversarial_q   # noqa: E402
from src.causal_prior.priors import (pc_stability_q, dagma_stability_q,     # noqa: E402
                                     iamb_stability_q)
from src.causal_prior.cv_mu import cv_pick_mu                               # noqa: E402
from src.causal_prior.metrics import support_recovery_metrics              # noqa: E402
from src.causal_prior.loading import causal_partition                      # noqa: E402
from experiments._io import new_run_dir                                     # noqa: E402

CACHE = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'cache_p30_headline'
OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'q_robustness'

# read from the cached cell (computed once for the recovery sweep)
CACHED = {'ges': 'q_ges', 'bootstrap_l1': 'q_bootstrap_l1', 'iamb': 'q_iamb'}
# extra bnlearn markov-blanket learners computed here (continuous y, fisher-z)
BNLEARN_MB = {'gs': 'gs', 'inter_iamb': 'inter.iamb'}

CSV_FIELDS = ['seed', 'n', 'p_edge', 'q_source', 'paradigm', 'mu_star_rel',
              'k_actual', 'S_recall', 'S_precision', 'correlate_inclusion',
              'causal_precision', 'auc_star', 'stability_star', 'log_loss_star']
PARADIGM = {'oracle': 'control', 'uniform': 'control', 'adversarial': 'control',
            'pc_stable': 'constraint-global', 'iamb': 'constraint-local',
            'gs': 'constraint-local', 'inter_iamb': 'constraint-local',
            'ges': 'score', 'dagma': 'continuous-opt', 'bootstrap_l1': 'stability'}


def build_sources(cell, p, S_star, confounded, B, alpha, rng):
    """all q vectors for one cell."""
    q = {'oracle': oracle_q(p, S_star), 'uniform': uniform_q(p, 0.5),
         'adversarial': adversarial_q(p, confounded)}
    for name, key in CACHED.items():
        if key in cell.files:
            q[name] = cell[key]
    X, yc = cell['X'], cell['y_continuous']
    q['pc_stable'] = pc_stability_q(X, yc, B=B, stable=True,
                                    rng=np.random.default_rng(rng))
    q['dagma'] = dagma_stability_q(X, yc, B=B, rng=np.random.default_rng(rng))
    for name, method in BNLEARN_MB.items():
        q[name] = iamb_stability_q(X, yc, B=B, method=method,
                                   rng=np.random.default_rng(rng))
    return q


def process_cell(path, B, alpha, n_mu, n_splits):
    cell = np.load(path, allow_pickle=False)
    p, k_star = int(cell['p']), int(cell['k_star'])
    seed, n, p_edge = int(cell['seed']), int(cell['n']), float(cell['p_edge'])
    X, y = cell['X'], cell['y']
    mu_scale = float(cell['mu_scale'])
    S_star = sorted(int(j) for j in cell['S_star'])
    confounded = sorted(int(j) for j in cell['confounded'])
    part = causal_partition(seed, p, float(cell['p_edge']), k_star)
    causes, correlates = part['all_causes'], part['correlates']

    mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, n_mu)]) * mu_scale
    K = 2 * k_star
    rng = np.random.default_rng(seed)
    sources = build_sources(cell, p, S_star, confounded, B, alpha, seed)

    rows = []
    for name, q in sources.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=np.asarray(q, float),
                            n_splits=n_splits, criterion='log_loss', rng=rng)
        m = support_recovery_metrics(cv.support, S_star, confounded,
                                     causes=causes, correlates=correlates)
        rows.append({
            'seed': seed, 'n': n, 'p_edge': p_edge,
            'q_source': name, 'paradigm': PARADIGM.get(name, '?'),
            'mu_star_rel': round(cv.mu_star / mu_scale, 4) if mu_scale else 0.0,
            'k_actual': m['k_actual'], 'S_recall': m['S_recall'],
            'S_precision': m['S_precision'],
            'correlate_inclusion': m.get('correlate_inclusion', float('nan')),
            'causal_precision': m.get('causal_precision', float('nan')),
            'auc_star': cv.auc_star, 'stability_star': cv.stability_star,
            'log_loss_star': cv.log_loss_star,
        })
    print(f'  {path.name}: {len(rows)} sources', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=Path, default=CACHE)
    ap.add_argument('--cell-glob', default='seed*_p30_n300_k5_pedge0.2.npz',
                    help='comma-separated cell globs; n_seeds caps seeds per config')
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--B', type=int, default=50, help='pyCausalFS stability subsamples')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--n-mu', type=int, default=12)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.n_seeds, args.B, args.n_jobs = 1, 10, 1

    import re
    matched = set()
    for g in args.cell_glob.split(','):
        matched.update(args.cache_dir.glob(g.strip()))
    def _seed(p):
        m = re.match(r'seed(\d+)_', p.name)
        return int(m.group(1)) if m else 1 << 30
    cells = sorted(c for c in matched if _seed(c) < args.n_seeds)
    if not cells:
        ap.error(f'no cells matching {args.cell_glob!r} in {args.cache_dir}')
    print(f'q-robustness: {len(cells)} cells, B={args.B}, sources='
          f'{len(CACHED)+3+2+len(BNLEARN_MB)}', flush=True)

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_cell)(p, args.B, args.alpha, args.n_mu, args.n_splits)
        for p in cells)
    df = pd.DataFrame([r for rows in results for r in rows], columns=CSV_FIELDS)

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    out = new_run_dir(OUT_DIR / ('headline' + ('_smoke' if args.smoke else '')), cfg)
    df.to_csv(out / 'q_robustness.csv', index=False)

    r, pr = df['S_recall'], df['S_precision']
    df['F1'] = np.where((r + pr) > 0, 2 * r * pr / (r + pr), 0.0)
    summary = (df.groupby(['paradigm', 'q_source'], sort=False)
               .agg(F1=('F1', 'mean'), recall=('S_recall', 'mean'),
                    corr_incl=('correlate_inclusion', 'mean'),
                    auc=('auc_star', 'mean'), stab=('stability_star', 'mean'),
                    n=('F1', 'count')).round(3))
    summary.to_csv(out / 'summary.csv')
    print('\nq-source robustness on the headline cell:', flush=True)
    print(summary.to_string(), flush=True)
    print(f'\nAUC spread across sources: {df.groupby("q_source")["auc_star"].mean().std():.4f} '
          f'(flat = do-no-harm)', flush=True)
    print(f'Done. Results in {out}/', flush=True)


if __name__ == '__main__':
    main()
