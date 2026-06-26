"""compute for fig 1a: fit vanilla + causal scorecards, write cards.json.

the refit (openml load, ges discovery, fasterrisk + auc-cv) runs here once; the
builder (figures/scorecard.py) only reads the json, so rendering never refits.

    python experiments/causal_prior/real/scorecard_fit.py --dataset diabetes130
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.binarize import fit_binarizer, apply_binarizer   # noqa: E402
from src.causal_prior.priors import discover_q                          # noqa: E402
from src.causal_prior.cv_mu import cv_pick_mu, make_mu_grid             # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk                # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset         # noqa: E402

OUT_DIR = ROOT / 'results/causal_prior/real'


def _load_diabetes130(seed=0, n_sub=20000):
    """load and binarize diabetes 130-US hospitals; subsample for speed."""
    from sklearn.datasets import fetch_openml
    DROP = ['encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3', 'weight',
            'examide', 'citoglipton', 'troglitazone', 'acetohexamide', 'tolazamide',
            'glimepiride.pioglitazone', 'metformin.rosiglitazone',
            'metformin.pioglitazone', 'glipizide.metformin', 'glyburide.metformin']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d = fetch_openml('diabetes130us', version=1, as_frame=True, parser='auto')
    df = d.data.drop(columns=[c for c in DROP if c in d.data.columns])
    out, names = [], []
    for c in df.columns:
        s = df[c]
        if str(s.dtype) in ('object', 'category'):
            s2 = s.astype('object').fillna('nan').astype(str)
            for lv in sorted(s2.unique()):
                if lv in ('nan', '?'):
                    continue
                out.append((s2 == lv).to_numpy(float))
                names.append(f'{c}={lv}')
        else:
            col = s.apply(lambda x: float(x) if str(x) not in ('?', 'nan') else np.nan)
            out.append(col.fillna(col.median()).to_numpy(float))
            names.append(str(c))
    X = np.column_stack(out)
    y = np.where(d.target.astype(str) == '<30', 1, -1).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), min(n_sub, len(y)), replace=False)
    return X[idx], np.asarray(names), y[idx]


def fit_cards(dataset='diabetes130', k=None, b=100, n_mu=12, n_cv=5, n_thresholds=4, seed=0):
    """return top-1 card dict (rows, intercept, auc) for vanilla and causal."""
    if dataset == 'diabetes130':
        X_orig, names, y = _load_diabetes130(seed=seed)
        k = k or 8
    else:
        class _A:
            sentinel_nan = False
        X_orig, names, y = load_dataset(dataset, _A())
        k = k or 10
    qsrc = 'ges_cg'

    names = np.asarray(names)
    rest, disc_idx = train_test_split(np.arange(len(y)), test_size=0.3,
                                      stratify=(y > 0), random_state=seed)
    tr_idx, te_idx = train_test_split(rest, test_size=0.3,
                                      stratify=(y[rest] > 0), random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = discover_q(qsrc, X_orig[disc_idx], y[disc_idx].astype(float), b, seed)
    spec, bin_names_raw, parent = fit_binarizer(X_orig[tr_idx], names, n_thresholds)
    bin_names = [str(b) for b in bin_names_raw]
    q_bin = q[parent]
    Xtr = apply_binarizer(X_orig[tr_idx], spec)
    Xte = apply_binarizer(X_orig[te_idx], spec)
    ytr, yte = y[tr_idx], (y[te_idx] > 0).astype(int)
    mu_scale, mu_grid = make_mu_grid(Xtr, ytr, n_mu)
    FR = import_fasterrisk()
    # auc cv on real data: log_loss and auc diverge on imbalanced clinical outcomes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = FR(k=k, mu=0.0, freq=None)
        van.fit(Xtr, ytr)
        mu_star = cv_pick_mu(Xtr, ytr, K=k, mu_grid=mu_grid, q=q_bin,
                             n_splits=n_cv, criterion='auc',
                             rng=np.random.default_rng(seed)).mu_star
        cau = FR(k=k, mu=float(mu_star), freq=q_bin.astype(float))
        cau.fit(Xtr, ytr)

    def top_card(fr):
        b0 = float(fr.beta0_[0])
        betas = np.asarray(fr.betas_[0])
        nz = np.nonzero(betas)[0]
        p = np.clip(fr.predict_proba(Xte), 1e-7, 1 - 1e-7)
        auc = float(roc_auc_score(yte, p))
        rows = [{'feature': bin_names[j], 'points': int(betas[j]), 'q': float(q_bin[j])}
                for j in sorted(nz, key=lambda i: abs(betas[i]), reverse=True)]
        return {'rows': rows, 'intercept': b0, 'auc': auc}

    return top_card(van), top_card(cau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='diabetes130')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    van, cau = fit_cards(dataset=args.dataset, seed=args.seed)
    out = OUT_DIR / f'scorecard_{args.dataset}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cards.json').write_text(json.dumps(
        {'dataset': args.dataset, 'seed': args.seed, 'vanilla': van, 'causal': cau}, indent=2))
    print(f"vanilla AUC={van['auc']:.3f}  causal AUC={cau['auc']:.3f}", flush=True)
    print(f'wrote {out}/cards.json', flush=True)


if __name__ == '__main__':
    main()
