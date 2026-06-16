"""Folktables transport: does the causal prior buy out-of-state transport that
predictive feature selection does not?

Real-data version of the §6.4 synthetic transport story, and the axis the FICO
comparison cannot test (FICO is one stationary distribution). Train on a source
state, test in-distribution (held-out source rows) and on shifted target states.

Two design choices learned the hard way:
  - the feature set must contain non-causal *predictive* correlates for the prior
    to have anything to prune. we one-hot POBP (place of birth): it predicts income
    within a state via state-specific composition but should not transport. without
    such features (and with more scorecard slots than features) the prior is inert.
  - mu is SWEPT, not cv-picked. cv on the source rewards the spurious correlates
    (they help in-distribution), so it zeroes the prior. §6.4 sweeps mu_rel and
    reads transport as a function of prior strength; we do the same.

Arms: vanilla (= causal at mu=0), causal at a grid of mu_rel, and the cfs
baselines (cfs_cg = valid mi-cg; cfs_iamb/hiton = naive fisher-z, invalid on mixed
data). Selection (q, cfs blankets) is done once on a held-out source discovery
split; seeds resample the source train/test. Discriminating quantity per arm:
    transport_gap(target) = auc(source in-distribution) - auc(target).
Win signature: as mu_rel grows the causal source auc dips slightly but the target
auc holds or rises, so its transport gap falls below vanilla's.

Run under `caffeinate -i`; first run downloads the ACS PUMS files (~per state).

usage:
    python experiments/causal_prior/real/folktables_transport.py --source CA --targets PR,SD
    python experiments/causal_prior/real/folktables_transport.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.causal_prior.priors import bnlearn_mb  # noqa: E402
from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.scorecard import discover_q, _import_fasterrisk  # noqa: E402
from src.causal_prior.baselines import _cfs_fisherz  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402

# continuous/ordinal kept numeric (thresholded later); the rest one-hot to {0,1}
# indicators. POBP (birthplace) is the spurious state-correlate we want present.
CONT = ['AGEP', 'WKHP', 'SCHL']
CAT = ['SEX', 'RAC1P']
HIGHCARD = ['POBP']
DATA_ROOT = REPO_ROOT / 'data' / 'real' / 'folktables'


def load_raw(state, year):
    from folktables import ACSDataSource, ACSIncome
    ds = ACSDataSource(survey_year=str(year), horizon='1-Year', survey='person',
                       root_dir=str(DATA_ROOT))
    raw = ACSIncome._preprocess(ds.get_data(states=[state], download=True))
    cols = ACSIncome.features + [ACSIncome.target]
    return raw.dropna(subset=cols)[cols].reset_index(drop=True)


def build_encoder(df_src, topk):
    """fix one-hot levels from the SOURCE so all states share identical columns."""
    enc = {'cat': {c: sorted(df_src[c].dropna().unique().tolist()) for c in CAT}}
    enc['high'] = {c: df_src[c].value_counts().index[:topk].tolist() for c in HIGHCARD}
    return enc


def encode(df, enc):
    """raw ACS df -> (X_orig float matrix, names, y in {-1,1}) under a fixed encoder."""
    from folktables import ACSIncome
    cols, names = [], []
    for c in CONT:
        cols.append(df[c].to_numpy(float)); names.append(c)
    for c in CAT:
        for lv in enc['cat'][c]:
            cols.append((df[c] == lv).to_numpy(float)); names.append(f'{c}={lv}')
    for c in HIGHCARD:
        keep = enc['high'][c]
        for lv in keep:
            cols.append((df[c] == lv).to_numpy(float)); names.append(f'{c}={lv}')
        cols.append((~df[c].isin(keep)).to_numpy(float)); names.append(f'{c}=other')
    y = np.where(df[ACSIncome.target].to_numpy(float) > 50000, 1, -1).astype(int)
    return np.column_stack(cols), names, y


def _auc(fr, cols, X, y01):
    if fr is None:
        return float('nan')
    return float(roc_auc_score(y01, np.clip(fr.predict_proba(X[:, cols]), 1e-7, 1 - 1e-7)))


def main():
    args = parse_args()
    targets = args.targets.split(',')
    print(f'loading source={args.source}, targets={targets} (year {args.year})...', flush=True)
    src_raw = load_raw(args.source, args.year)
    enc = build_encoder(src_raw, args.topk)
    Xsrc, names, ysrc = encode(src_raw, enc)
    names = np.asarray(names)
    tgt = {}
    for t in targets:
        Xt, _, yt = encode(load_raw(t, args.year), enc)
        tgt[t] = (Xt, yt)
    print(f'features={len(names)} (>=2 cont + indicators); source n={len(ysrc)}; ' +
          '; '.join(f'{t} n={len(v[1])}' for t, v in tgt.items()), flush=True)

    FR = _import_fasterrisk()
    rng0 = np.random.default_rng(args.seed)
    perm = rng0.permutation(len(ysrc))
    disc = perm[:args.n_disc]                      # held-out selection split
    pool = perm[args.n_disc:]                      # seeds resample train/test from here

    # selection once on the held-out source discovery split
    print(f'selection on held-out source discovery (n={len(disc)})...', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = discover_q('ges_cg', Xsrc[disc], ysrc[disc].astype(float), args.b, args.seed)
        mbs = {'cfs_iamb': _cfs_fisherz('iamb', Xsrc[disc], ysrc[disc], args.alpha),
               'cfs_hiton_mb': _cfs_fisherz('hiton_mb', Xsrc[disc], ysrc[disc], args.alpha),
               'cfs_cg': bnlearn_mb(Xsrc[disc], ysrc[disc], method='iamb', test='mi-cg',
                                    alpha=args.alpha)}
    for a, mb in mbs.items():
        print(f'  {a}: |MB|={len(mb)}', flush=True)

    mu_grid = [0.0] + args.mu_rel
    rows = []
    for s in range(args.reps):
        rng = np.random.default_rng((args.seed, s))
        sel = rng.permutation(len(pool))
        tr = pool[sel[:args.n_train]]
        te = pool[sel[args.n_train:args.n_train + args.n_test]]
        spec, _, parent = fit_binarizer(Xsrc[tr], names.tolist(), args.n_thresholds)
        q_bin = q[parent]
        Xtr = apply_binarizer(Xsrc[tr], spec)
        all_cols = np.arange(Xtr.shape[1])
        Xte_src = apply_binarizer(Xsrc[te], spec)
        yte01 = (ysrc[te] > 0).astype(int)
        tgt_bin = {}  # one target eval subsample per seed, binarized with the source spec
        for t, (Xt, yt) in tgt.items():
            idx = rng.choice(len(yt), min(args.n_test, len(yt)), replace=False)
            tgt_bin[t] = (apply_binarizer(Xt[idx], spec), (yt[idx] > 0).astype(int))

        mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ysrc[tr])))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for mr in mu_grid:  # causal prior swept (mr=0 is vanilla)
                fr = FR(k=args.k, mu=mr * mu_scale, freq=q_bin.astype(float) if mr > 0 else None)
                fr.fit(Xtr, ysrc[tr])
                rec = {'seed': s, 'arm': 'vanilla' if mr == 0 else 'causal', 'mu_rel': mr,
                       'auc_source': _auc(fr, all_cols, Xte_src, yte01)}
                for t, (Xtb, ytb) in tgt_bin.items():
                    rec[f'auc_{t}'] = _auc(fr, all_cols, Xtb, ytb)
                rows.append(rec)
            for arm, mb in mbs.items():
                cols = all_cols[np.isin(parent, mb)] if mb else all_cols[:0]
                fr = None if len(cols) == 0 else FR(k=args.k, mu=0.0, freq=None)
                if fr is not None:
                    fr.fit(Xtr[:, cols], ysrc[tr])
                rec = {'seed': s, 'arm': arm, 'mu_rel': np.nan,
                       'auc_source': _auc(fr, cols, Xte_src, yte01)}
                for t, (Xtb, ytb) in tgt_bin.items():
                    rec[f'auc_{t}'] = _auc(fr, cols, Xtb, ytb)
                rows.append(rec)
        print(f'  seed {s + 1}/{args.reps} done', flush=True)

    long = pd.DataFrame(rows)
    env_cols = ['auc_source'] + [f'auc_{t}' for t in targets]
    summ = long.groupby(['arm', 'mu_rel'], dropna=False)[env_cols].mean().round(4)
    for t in targets:
        summ[f'gap_{t}'] = (summ['auc_source'] - summ[f'auc_{t}']).round(4)

    out = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'folktables_transport'
                      / f'{args.source}_to_{"-".join(targets)}_{args.year}_musweep',
                      {**vars(args), 'features': names.tolist()})
    long.to_csv(out / 'seeds.csv', index=False)
    summ.to_csv(out / 'summary.csv')
    plot_transport(long, targets, out / 'transport.png')
    print(summ.to_string(), flush=True)
    print(f'Done. Results in {out}/', flush=True)


def plot_transport(long, targets, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = targets[-1]  # the larger-shift target
    cau = long[long['arm'] == 'causal'].groupby('mu_rel')[['auc_source', f'auc_{t}']].mean()
    van = long[long['arm'] == 'vanilla'][f'auc_{t}'].mean()
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot(cau.index, cau[f'auc_{t}'], marker='o', color='C1', label=f'causal -> {t}')
    ax.plot(cau.index, cau['auc_source'], marker='o', color='C0', ls='--',
            label='causal -> source (in-dist)')
    ax.axhline(van, color='0.5', ls=':', label=f'vanilla -> {t} (mu=0)')
    for arm, c in (('cfs_cg', 'C2'), ('cfs_iamb', '0.6')):
        v = long[long['arm'] == arm][f'auc_{t}'].mean()
        ax.axhline(v, color=c, ls=':', lw=1, label=f'{arm} -> {t}')
    ax.set(xlabel='prior strength mu_rel', ylabel='test AUC',
           title=f'transport under shift ({t})')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='CA')
    p.add_argument('--targets', default='PR,SD')
    p.add_argument('--year', type=int, default=2018)
    p.add_argument('--reps', type=int, default=6)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--mu-rel', type=float, nargs='+', default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument('--topk', type=int, default=8, help='POBP levels kept before "other"')
    p.add_argument('--n-disc', type=int, default=1500)
    p.add_argument('--n-train', type=int, default=2500)
    p.add_argument('--n-test', type=int, default=4000)
    p.add_argument('--b', type=int, default=40)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.reps, args.b, args.n_disc, args.n_train, args.n_test, args.targets, args.mu_rel = \
            2, 10, 800, 1200, 1500, 'SD', [1.0, 4.0]
    return args


if __name__ == '__main__':
    main()
