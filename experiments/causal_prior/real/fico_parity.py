"""FICO HELOC parity: causal-prior FasterRisk vs vanilla.

No causal ground truth here, so no support-recovery claim; the bar is parity:
causal AUC within noise of vanilla at matched sparsity. Leakage-free: q is
discovered once on a held-out discovery set (disjoint from all eval rows), and
per split the quantile-binarizer and mu scale are fit on train rows only and
applied to test. vanilla (mu=0) vs causal (mu by inner-CV log-loss) at matched K
over the outer splits. Writes q, per-split metrics, and the parity table to
results/. (For the test-AUC-vs-k curve use fico_ksweep.py.)

q source (--qsrc): pc/ges are Gaussian (pcalg, Fisher-Z / Gaussian-BIC); pc_cg/
ges_cg are conditional-Gaussian (bnlearn mi-cg / bic-cg), appropriate for FICO's
mixed shape (continuous features, binary target). The CG variants are the
principled choice here; combine with --sentinel-nan to also handle the -7/-8/-9
missingness codes.

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
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from src.causal_prior.priors import (  # noqa: E402
    pc_stability_q, ges_stability_q, bnlearn_stability_q)
from experiments._io import new_run_dir  # noqa: E402

DATA = REPO_ROOT / 'data/real/raw/HelocData.csv'
TARGET = 'RiskFlag'
POS_LABEL = 'Bad'
SENTINELS = (-7, -8, -9)   # fico codes: no record / no usable trades / condition not met


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--qsrc', choices=['pc', 'ges', 'pc_cg', 'ges_cg'], default='pc',
                   help='q source: pc/ges = Gaussian (pcalg); pc_cg/ges_cg = '
                        'conditional-Gaussian for mixed data (bnlearn)')
    p.add_argument('--k', type=int, default=10, help='matched sparsity for both arms')
    p.add_argument('--splits', type=int, default=10, help='outer train/test splits')
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--discovery-frac', type=float, default=0.3,
                   help='fraction held out as the q-discovery set (disjoint from eval)')
    p.add_argument('--n_cv', type=int, default=5, help='inner CV folds for mu selection')
    p.add_argument('--b', type=int, default=100, help='discovery stability subsamples')
    p.add_argument('--n_thresholds', type=int, default=4,
                   help='interior quantiles per feature for binarization')
    p.add_argument('--sentinel-nan', action='store_true',
                   help="treat FICO codes -7/-8/-9 as missing (median-impute over "
                        "valid values + add '_missing' indicators) not as magnitudes")
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


def load_features(df, sentinel_nan):
    """Numeric feature matrix and names. With sentinel_nan, FICO's special codes
    (-7/-8/-9) are treated as missing rather than magnitudes: each affected column
    is median-imputed over its valid values, and a binary `<feat>_missing` indicator
    is appended so informative missingness stays available to the graph and model."""
    names = [c for c in df.columns if c != TARGET]
    raw = df[names].apply(pd.to_numeric, errors='coerce')
    if not sentinel_nan:
        return raw.fillna(0.0).to_numpy(float), names
    cols, out_names = [], []
    for name in names:
        s = raw[name].astype(float)
        miss = s.isin(SENTINELS)
        valid = s.mask(miss)
        cols.append(valid.fillna(valid.median()).fillna(0.0).to_numpy(float))
        out_names.append(name)
        if miss.any():
            cols.append(miss.astype(float).to_numpy())
            out_names.append(f'{name}_missing')
    return np.column_stack(cols), out_names


def fit_binarizer(X_orig, names, n_thresholds):
    """Fit binarization thresholds on the given rows (use train-only rows to avoid
    leakage). Returns (spec, col_names, parent_idx): spec is a list of (j, t) per
    output column, t=None for a passed-through already-binary column or a threshold
    for an `x_j <= t` indicator; parent_idx[c] is the index into `names` of the
    original feature column c was derived from. Transform with apply_binarizer."""
    qlevels = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    spec, col_names, parent = [], [], []
    for j, name in enumerate(names):
        x = X_orig[:, j]
        if set(np.unique(x).tolist()) <= {0.0, 1.0}:
            spec.append((j, None))
            col_names.append(name)
            parent.append(j)
            continue
        for t in np.unique(np.quantile(x, qlevels)):
            spec.append((j, float(t)))
            col_names.append(f'{name}_le_{t:g}')
            parent.append(j)
    return spec, col_names, np.asarray(parent, dtype=int)


def apply_binarizer(X_orig, spec):
    """Transform original features into the binarized matrix using a fitted spec."""
    cols = [X_orig[:, j] if t is None else (X_orig[:, j] <= t).astype(float)
            for j, t in spec]
    return np.column_stack(cols)


def binarize(X_orig, names, n_thresholds):
    """Convenience wrapper: fit thresholds and transform on the same rows. For a
    leakage-free protocol fit on train rows and apply to test rows separately."""
    spec, col_names, parent = fit_binarizer(X_orig, names, n_thresholds)
    return apply_binarizer(X_orig, spec), col_names, parent


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


def fit_eval(FasterRisk, X_tr, y_tr, X_te, y_te, mu, q, k, return_card=False):
    fr = FasterRisk(k=k, mu=float(mu), freq=q.astype(float) if q is not None else None)
    fr.fit(X_tr, y_tr)
    p = np.clip(fr.predict_proba(X_te), 1e-7, 1 - 1e-7)
    y01 = (y_te > 0).astype(int)
    out = {
        'auc': roc_auc_score(y01, p),
        'brier': brier_score_loss(y01, p),
        'ece': ece(y01, p),
        'nfeat': int(np.count_nonzero(fr.betas_[0])),
    }
    if return_card:  # full fitted scorecard for recording: betas, intercept, multiplier
        out['card'] = {'betas': np.asarray(fr.betas_[0]).tolist(),
                       'intercept': float(fr.beta0_[0]),
                       'multiplier': float(fr.multipliers_[0])}
    return out


def discover_q(qsrc, X, y, b, seed):
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if qsrc == 'ges':
            q, _ = ges_stability_q(X, y, B=b, subsample_fraction=0.5, rng=rng)
        elif qsrc == 'pc_cg':
            q = bnlearn_stability_q(X, y, method='mi-cg', B=b,
                                    subsample_fraction=0.5, rng=rng)
        elif qsrc == 'ges_cg':
            q = bnlearn_stability_q(X, y, method='bic-cg', B=b,
                                    subsample_fraction=0.5, rng=rng)
        else:
            q = pc_stability_q(X, y, B=b, subsample_fraction=0.5,
                               alpha=0.1, m_max=5, rng=rng)
    return q


def main():
    args = parse_args()
    if not DATA.exists():
        sys.exit(f'FICO CSV not found at {DATA}\n'
                 'Download heloc_dataset_v1.csv (FICO Community Explainable ML '
                 'Challenge) and place it there, then rerun.')

    df = pd.read_csv(DATA)
    y = np.where(df[TARGET].astype(str).str.strip() == POS_LABEL, 1, -1).astype(int)
    X_orig, names = load_features(df, args.sentinel_nan)
    n, p_orig = X_orig.shape
    n_ind = sum(nm.endswith('_missing') for nm in names)
    extra = f' (+{n_ind} missingness indicators)' if args.sentinel_nan else ''
    print(f'FICO: n={n}  features={p_orig}{extra}  '
          f'positive ({POS_LABEL})={(y == 1).mean():.0%}', flush=True)

    # leakage-free: discover q once on a held-out set disjoint from all eval rows
    pool_idx, disc_idx = train_test_split(
        np.arange(n), test_size=args.discovery_frac,
        stratify=(y > 0), random_state=args.seed)
    print(f'discovery n={len(disc_idx)} (held out), eval pool n={len(pool_idx)}', flush=True)
    print(f'{args.qsrc.upper()} discovery (B={args.b}) on held-out set...', flush=True)
    q_orig = discover_q(args.qsrc, X_orig[disc_idx], y[disc_idx].astype(float), args.b, args.seed)

    FasterRisk = _import_fasterrisk()
    sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size,
                                 random_state=args.seed)
    records = []
    for s, (trp, tep) in enumerate(sss.split(pool_idx, (y[pool_idx] > 0).astype(int))):
        tr, te = pool_idx[trp], pool_idx[tep]
        # binarizer + mu scale fit on train rows only, applied to test
        spec, _, parent = fit_binarizer(X_orig[tr], names, args.n_thresholds)
        q_bin = q_orig[parent]
        Xtr, Xte = apply_binarizer(X_orig[tr], spec), apply_binarizer(X_orig[te], spec)
        ytr, yte = y[tr], y[te]
        mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ytr)))
        mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, 3 if args.smoke else 12)]) * mu_scale
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            van = fit_eval(FasterRisk, Xtr, ytr, Xte, yte, 0.0, None, args.k)
            cv = cv_pick_mu(Xtr, ytr, K=args.k, mu_grid=mu_grid, q=q_bin,
                            n_splits=args.n_cv, criterion='log_loss',
                            rng=np.random.default_rng(args.seed + s))
            cau = fit_eval(FasterRisk, Xtr, ytr, Xte, yte, cv.mu_star, q_bin, args.k)
        mu_hat_rel = cv.mu_star / mu_scale if mu_scale else 0.0
        records.append({'split': s, 'arm': 'vanilla', 'mu_hat_rel': 0.0, **van})
        records.append({'split': s, 'arm': 'causal', 'mu_hat_rel': mu_hat_rel, **cau})
        print(f'  split {s + 1}/{args.splits} done (mu_hat_rel={mu_hat_rel:.2f})', flush=True)

    df_splits = pd.DataFrame(records)
    metrics = ['auc', 'brier', 'ece', 'nfeat']
    df_parity = (df_splits.groupby('arm')[metrics]
                 .agg(['mean', 'std']).loc[['vanilla', 'causal']])
    mu_hats = df_splits.loc[df_splits['arm'] == 'causal', 'mu_hat_rel'].to_numpy()

    suffix = (f'fico_{args.qsrc}_k{args.k}' + ('_sent' if args.sentinel_nan else '')
              + ('_smoke' if args.smoke else ''))
    config = {**vars(args), 'data': str(DATA), 'target': TARGET, 'pos_label': POS_LABEL,
              'n': int(n), 'p_orig': int(p_orig), 'discovery_n': int(len(disc_idx)),
              'eval_pool_n': int(len(pool_idx)), 'leakage_free': True,
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
    print(f'Done (leakage-free). Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
