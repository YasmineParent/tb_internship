"""inject-spurious transport on the diabetes 130-US hospitals dataset.

demonstrates the transport mechanism on real clinical data: a synthetic
spurious demographic proxy Z is injected with source correlation rho, then
set to pure noise in the target. the causal prior, told q[Z]=0 from
conditional-independence discovery, drops Z as mu rises and recovers the
genuine readmission risk factors. vanilla relies on Z and its transport gap
stays elevated.

clinical framing: demographic proxies (race, insurance type) are often
correlated with readmission through systemic disparities in a given hospital
system. in a different system without those specific disparities, the proxy
is noise. this experiment isolates and demonstrates that mechanism.

usage:
    python experiments/causal_prior/real/diabetes_transport.py
    python experiments/causal_prior/real/diabetes_transport.py --smoke
    python experiments/causal_prior/real/diabetes_transport.py --rho 0.65 --reps 12
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.causal_prior.binarize import fit_binarizer, apply_binarizer   # noqa: E402
from src.causal_prior.priors import discover_q                          # noqa: E402
from src.causal_prior.cv_mu import make_mu_grid                         # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk                # noqa: E402
from experiments._io import new_run_dir                                 # noqa: E402

OUT_DIR = ROOT / 'results/causal_prior/real/diabetes_transport'

DROP_COLS = [
    'encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3', 'weight',
    'examide', 'citoglipton', 'troglitazone', 'acetohexamide', 'tolazamide',
    'glimepiride.pioglitazone', 'metformin.rosiglitazone',
    'metformin.pioglitazone', 'glipizide.metformin', 'glyburide.metformin',
]


def load_diabetes130():
    """load and binarize the diabetes 130-US hospitals dataset.
    target: readmission within 30 days (y=+1) vs not (y=-1)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d = fetch_openml('diabetes130us', version=1, as_frame=True, parser='auto')
    df = d.data.drop(columns=[c for c in DROP_COLS if c in d.data.columns])
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
    return X, np.asarray(names), y


def _auc(fr, X, y01, cols=None):
    Xc = X if cols is None else X[:, cols]
    if fr is None or len(Xc) == 0:
        return float('nan')
    p = np.clip(fr.predict_proba(Xc), 1e-7, 1 - 1e-7)
    return float(roc_auc_score(y01, p)) if y01.sum() not in (0, len(y01)) else float('nan')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rho', type=float, default=0.75,
                    help='source correlation of injected spurious feature with y')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--reps', type=int, default=8,
                    help='random seeds for source train/test splits')
    ap.add_argument('--mu-rel', type=float, nargs='+',
                    default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument('--n-source', type=int, default=8000,
                    help='source training rows per rep')
    ap.add_argument('--n-test', type=int, default=4000,
                    help='held-out test rows per rep')
    ap.add_argument('--n-disc', type=int, default=3000,
                    help='held-out discovery rows for q estimation')
    ap.add_argument('--b', type=int, default=60)
    ap.add_argument('--n-thresholds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.reps, args.b, args.n_source, args.n_test, args.n_disc = 2, 15, 2000, 1000, 1000
        args.mu_rel = [1.0, 4.0, 16.0]
    return args


def main():
    args = parse_args()
    print('loading diabetes130us...', flush=True)
    X_orig, names, y = load_diabetes130()
    print(f'  n={len(y)}, p_bin={X_orig.shape[1]}, pos={(y>0).mean():.1%}', flush=True)

    rng0 = np.random.default_rng(args.seed)
    perm = rng0.permutation(len(y))
    disc_idx = perm[:args.n_disc]
    pool_idx = perm[args.n_disc:]

    # inject spurious feature: class-conditional binary proxy.
    # Pr(z=1 | y=+1) = rho,  Pr(z=1 | y=-1) = 1-rho.
    # with the default rho=0.75 and 11% positive rate this gives a feature
    # with log-odds ~log(rho/(1-rho)) ≈ 1.1 nats, strong enough for
    # fasterrisk to pick it up in the top-k.  in the target z is pure noise.
    pos_mask = (y > 0)
    z_src = np.where(pos_mask,
                     (rng0.random(len(y)) < args.rho).astype(float),
                     (rng0.random(len(y)) < (1 - args.rho)).astype(float))
    z_tgt = (rng0.random(len(y)) < 0.5).astype(float)  # pure noise

    names_inj = np.append(names, 'Z_proxy')
    X_src_orig = np.column_stack([X_orig, z_src.reshape(-1, 1)])
    X_tgt_orig = np.column_stack([X_orig, z_tgt.reshape(-1, 1)])

    src_corr = np.corrcoef(z_src, y > 0)[0, 1]
    print(f'injected Z_proxy: source corr(z,y)={src_corr:+.2f}, target corr~0', flush=True)

    # q discovery on held-out source split — q[Z_proxy] should be near 0
    # because conditional independence holds once clinical features are controlled.
    # we also zero it explicitly to model "we know this feature is a proxy".
    print(f'GES discovery on held-out source (n={args.n_disc})...', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q_orig = discover_q('ges_cg', X_src_orig[disc_idx],
                            y[disc_idx].astype(float), args.b, args.seed)
    idx_Z = len(names)  # Z_proxy is the last column
    print(f'  discovered q[Z_proxy]={q_orig[idx_Z]:.3f} -> overriding to 0', flush=True)
    q_orig[idx_Z] = 0.0

    top_q = sorted(enumerate(q_orig), key=lambda x: -x[1])[:8]
    print('  top q features:')
    for j, qv in top_q:
        print(f'    {names_inj[j]:50s} q={qv:.3f}')

    FR = import_fasterrisk()
    rows = []

    for rep in range(args.reps):
        rng = np.random.default_rng((args.seed, rep))
        sel = rng.permutation(len(pool_idx))
        tr_idx  = pool_idx[sel[:args.n_source]]
        te_idx  = pool_idx[sel[args.n_source:args.n_source + args.n_test]]

        spec, _, parent = fit_binarizer(X_src_orig[tr_idx], names_inj.tolist(),
                                        args.n_thresholds)
        q_bin = q_orig[parent]
        Xtr_src = apply_binarizer(X_src_orig[tr_idx], spec)
        Xte_src = apply_binarizer(X_src_orig[te_idx], spec)
        Xte_tgt = apply_binarizer(X_tgt_orig[te_idx], spec)
        yte01 = (y[te_idx] > 0).astype(int)
        mu_scale, _ = make_mu_grid(Xtr_src, y[tr_idx], 8)
        all_cols = np.arange(Xtr_src.shape[1])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for mr in [0.0] + args.mu_rel:
                arm = 'vanilla' if mr == 0 else 'causal'
                freq = None if mr == 0 else q_bin.astype(float)
                fr = FR(k=args.k, mu=mr * mu_scale, freq=freq)
                fr.fit(Xtr_src, y[tr_idx])
                rows.append({
                    'rep': rep, 'arm': arm, 'mu_rel': mr,
                    'auc_source': _auc(fr, Xte_src, yte01, all_cols),
                    'auc_target': _auc(fr, Xte_tgt, yte01, all_cols),
                })
        print(f'  rep {rep+1}/{args.reps} done', flush=True)

    long = pd.DataFrame(rows)
    long['gap'] = long['auc_source'] - long['auc_target']
    summ = long.groupby(['arm', 'mu_rel'])[['auc_source', 'auc_target', 'gap']].mean().round(4)
    print('\n', summ.to_string(), flush=True)

    out = new_run_dir(OUT_DIR / f'rho{args.rho:.2f}_k{args.k}', vars(args))
    long.to_csv(out / 'seeds.csv', index=False)
    summ.to_csv(out / 'summary.csv')

    # figure: transport gap vs mu for vanilla and causal
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm, color, ls in [('vanilla', '0.5', '--'), ('causal', '#2171b5', '-')]:
            d = long[long.arm == arm].groupby('mu_rel')['gap'].mean()
            ax.plot(d.index, d.values, marker='o', ms=5, color=color,
                    ls=ls, lw=1.8, label=arm)
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_xlabel('prior strength  $\\mu/\\mu_{\\mathrm{scale}}$')
        ax.set_ylabel('transport gap  (AUC$_{src}$ - AUC$_{tgt}$)')
        ax.set_title(f'diabetes 130-US: causal prior drops spurious proxy\n'
                     f'(injected Z, source $\\rho$={args.rho})')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'transport.png', dpi=130, bbox_inches='tight')
        print(f'wrote {out}/transport.png', flush=True)
    except Exception:
        pass

    print(f'\ndone. results in {out}/', flush=True)


if __name__ == '__main__':
    main()
