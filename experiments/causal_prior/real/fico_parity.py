"""FICO HELOC parity: causal-prior FasterRisk vs vanilla.

No causal ground truth here, so no support-recovery claim; the bar is parity:
causal AUC within noise of vanilla at matched sparsity. Self-contained from one
raw CSV: causal-discovery stability on the original (continuous) features -> q;
quantile-binarize each feature into indicator columns (q propagated to the
children); vanilla (mu=0) vs causal (mu by inner-CV log-loss) at matched K over
the outer splits. Writes q, per-split metrics, and the parity table to results/.

q source (--qsrc): PC or GES subsample-stability on the original features. The
synthetic experiments found GES is the selective source (PC near-noise), so GES
is the stronger parity test: it lets the prior engage and still tie vanilla.

DATA: raw 23-feature heloc_dataset_v1.csv (FICO Community Explainable ML
Challenge) at the path below; target 'RiskFlag', positive class 'Bad'.

Caveats (flag in writeup): PC runs Fisher Z with the binary target as continuous
(point-biserial; the spec prefers a mixed CI test); quantile binarization leaves
FICO's special codes (-9/-8/-7) in the numeric column.

Usage:
    python experiments/causal_prior/real/fico_parity.py
    python experiments/causal_prior/real/fico_parity.py --qsrc ges
    python experiments/causal_prior/real/fico_parity.py --smoke
    python experiments/causal_prior/real/fico_parity.py --k 8 --splits 20
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from src.causal_prior.priors import pc_stability_q, ges_stability_q  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402

DATA = REPO_ROOT / 'data/real/raw/HelocData.csv'
TARGET = 'RiskFlag'
POS_LABEL = 'Bad'          # the credit-risk event = positive class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--qsrc', choices=['pc', 'ges'], default='pc',
                   help='causal-discovery source for q')
    p.add_argument('--k', type=int, default=10, help='matched sparsity for both arms')
    p.add_argument('--splits', type=int, default=10, help='outer train/test splits')
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--n_cv', type=int, default=5, help='inner CV folds for mu selection')
    p.add_argument('--b', type=int, default=100, help='discovery stability subsamples')
    p.add_argument('--n_thresholds', type=int, default=4,
                   help='interior quantiles per feature for binarization')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true',
                   help='shrink the run for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.b, args.splits = 10, 2
    return args


def _import_fasterrisk():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
    return FasterRisk


def binarize(X_orig, names, n_thresholds):
    """Threshold each original feature at interior quantiles into `<= t` indicator
    columns. Returns (X_bin, col_names, parent_idx) where parent_idx[c] is the
    index into `names` of the original feature column c was derived from."""
    qlevels = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    cols, col_names, parent = [], [], []
    for j, name in enumerate(names):
        x = X_orig[:, j]
        for t in np.unique(np.quantile(x, qlevels)):
            cols.append((x <= t).astype(float))
            col_names.append(f'{name}_le_{t:g}')
            parent.append(j)
    return np.column_stack(cols), col_names, np.asarray(parent, dtype=int)


def ece(y01, p, n_bins=10):
    """Expected calibration error: weighted mean |accuracy - confidence| over bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y01[m].mean() - p[m].mean())
    return float(e)


def fit_eval(FasterRisk, X_tr, y_tr, X_te, y_te, mu, q, k):
    fr = FasterRisk(k=k, mu=float(mu), freq=q.astype(float) if q is not None else None)
    fr.fit(X_tr, y_tr)
    p = np.clip(fr.predict_proba(X_te), 1e-7, 1 - 1e-7)
    y01 = (y_te > 0).astype(int)
    return {
        'auc': roc_auc_score(y01, p),
        'brier': brier_score_loss(y01, p),
        'ece': ece(y01, p),
        'nfeat': int(np.count_nonzero(fr.betas_[0])),
    }


def discover_q(qsrc, X, y, b, seed):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if qsrc == 'ges':
            q, _ = ges_stability_q(X, y, B=b, subsample_fraction=0.5,
                                   rng=np.random.default_rng(seed))
        else:
            q = pc_stability_q(X, y, B=b, subsample_fraction=0.5,
                               alpha=0.1, m_max=5, rng=np.random.default_rng(seed))
    return q


def main():
    args = parse_args()
    if not DATA.exists():
        sys.exit(f'FICO CSV not found at {DATA}\n'
                 'Download heloc_dataset_v1.csv (FICO Community Explainable ML '
                 'Challenge) and place it there, then rerun.')

    df = pd.read_csv(DATA)
    y = np.where(df[TARGET].astype(str).str.strip() == POS_LABEL, 1, -1).astype(int)
    names = [c for c in df.columns if c != TARGET]
    X_orig = df[names].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(float)
    n, p_orig = X_orig.shape
    print(f'FICO: n={n}  features={p_orig}  positive ({POS_LABEL})={(y == 1).mean():.0%}',
          flush=True)

    print(f'{args.qsrc.upper()} subsample-stability (B={args.b}) on original features...',
          flush=True)
    q_orig = discover_q(args.qsrc, X_orig, y.astype(float), args.b, args.seed)

    X_bin, _, parent = binarize(X_orig, names, args.n_thresholds)
    q_bin = q_orig[parent]
    print(f'binarized: {X_bin.shape[1]} columns; '
          f'q_bin nonzero on {int((q_bin > 0).sum())} columns', flush=True)

    FasterRisk = _import_fasterrisk()
    mu_scale = float(np.median(0.5 * np.abs(X_bin.T @ y)))
    mu_rel = np.logspace(-2, 1, 3 if args.smoke else 12)
    mu_grid = np.concatenate([[0.0], mu_rel]) * mu_scale

    sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size,
                                 random_state=args.seed)
    records = []
    for s, (tr, te) in enumerate(sss.split(X_bin, (y > 0).astype(int))):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            van = fit_eval(FasterRisk, X_bin[tr], y[tr], X_bin[te], y[te], 0.0, None, args.k)
            cv = cv_pick_mu(X_bin[tr], y[tr], K=args.k, mu_grid=mu_grid, q=q_bin,
                            n_splits=args.n_cv, criterion='log_loss',
                            rng=np.random.default_rng(args.seed + s))
            cau = fit_eval(FasterRisk, X_bin[tr], y[tr], X_bin[te], y[te],
                           cv.mu_star, q_bin, args.k)
        mu_hat_rel = cv.mu_star / mu_scale if mu_scale else 0.0
        records.append({'split': s, 'arm': 'vanilla', 'mu_hat_rel': 0.0, **van})
        records.append({'split': s, 'arm': 'causal', 'mu_hat_rel': mu_hat_rel, **cau})
        print(f'  split {s + 1}/{args.splits} done (mu_hat_rel={mu_hat_rel:.2f})', flush=True)

    df_splits = pd.DataFrame(records)
    metrics = ['auc', 'brier', 'ece', 'nfeat']
    df_parity = (df_splits.groupby('arm')[metrics]
                 .agg(['mean', 'std']).loc[['vanilla', 'causal']])
    mu_hats = df_splits.loc[df_splits['arm'] == 'causal', 'mu_hat_rel'].to_numpy()

    suffix = f'fico_{args.qsrc}_k{args.k}' + ('_smoke' if args.smoke else '')
    config = {**vars(args), 'data': str(DATA), 'target': TARGET, 'pos_label': POS_LABEL,
              'n': int(n), 'p_orig': int(p_orig), 'mu_scale': mu_scale,
              'mu_hat_rel_mean': float(mu_hats.mean()),
              'mu_hat_nonzero_splits': int((mu_hats > 0).sum())}
    output_dir = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'fico_parity' / suffix, config)

    (pd.DataFrame({'feature': names, 'q': q_orig})
     .sort_values('q', ascending=False)
     .to_csv(output_dir / 'q.csv', index=False))
    df_splits.to_csv(output_dir / 'splits.csv', index=False)
    df_parity.to_csv(output_dir / 'parity.csv')

    print(df_parity.to_string(), flush=True)
    print(f'mu_hat_rel mean {mu_hats.mean():.3f}, '
          f'nonzero in {int((mu_hats > 0).sum())}/{args.splits} splits', flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
