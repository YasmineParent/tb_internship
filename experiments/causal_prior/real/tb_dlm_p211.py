"""§6.3 Experiment 3.1: causal prior on the raw (p >> n) TB-DLM mutation set.

Both arms run on the full mutation matrix (all genes, constant columns dropped).
The causal prior q_j is nonzero only on the features CMM evaluated (edges into
dlm_mic from the with_lineage_forbid_to_mic stability run); q_j = 0 elsewhere
encodes 'no causal evidence'. This is the regime the pipeline hypothesises the
prior helps most: at p >> n vanilla can grab any of ~200 spurious mutations and
its support thrashes across folds, while the prior should pin selection to the
causally-supported features and stabilise it.

vanilla (mu=0) vs causal (mu by log-loss CV) at matched K, over several seeds.
Headline = 10-fold support stability (Jaccard); AUC for reference.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402

DATA = ROOT / 'data/real/processed/tb_pheno_geno_clean.csv'
QCSV = ROOT / ('results/mixed_cmm/subsampling/tb_subsampling_dlm_mp4_k6_mcc4/'
               'with_lineage_forbid_to_mic/edge_stability.csv')
MIC_NODE = 'dlm_mic'
TARGET = 'interp_dlm016'
K = 5
N_SPLITS = 10
SEEDS = range(6)
MU_REL_GRID = np.array([0.0, 0.5, 1.5, 5.0])
import os
MIN_POS = int(os.environ.get('MIN_POS', '2'))  # 2: drop singletons; 1: keep them (true p>>n)


def build():
    df = pd.read_csv(DATA)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = np.where(df[TARGET].to_numpy() == 2.0, 1, -1).astype(int)

    meta = {'isolate', 'glims', 'type', 'lineage'}
    mut_cols = [c for c in df.columns
                if c not in meta
                and not c.startswith(('interp_', 'prop_mutants'))
                and 'mic' not in c]
    M = df[mut_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0)
    pos = (M > 0).sum()
    keep = sorted(pos[(pos >= MIN_POS) & (pos <= len(df) - MIN_POS)].index)
    lin = {f'lineage_{lv}': (df['lineage'].astype('Int64') == lv).astype(float).to_numpy()
           for lv in (2, 4)}
    feat_names = keep + list(lin)
    X = np.column_stack([M[keep].to_numpy()] + [lin[c] for c in lin])

    edges = pd.read_csv(QCSV)
    into_mic = edges[edges['target'] == MIC_NODE].set_index('source')['frequency']
    q = np.array([float(into_mic.get(n, 0.0)) for n in feat_names])
    return X, y, q, feat_names


def main():
    X, y, q, feat_names = build()
    print(f'n={len(y)}  resistant={int((y==1).sum())}  p={len(feat_names)}  '
          f'nonzero_q={int((q>0).sum())}  K={K}', flush=True)
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_grid = MU_REL_GRID * mu_scale

    print(f'{"seed":>4s} | {"stab_van":>8s} {"stab_cau":>8s} {"d":>7s} | '
          f'{"mu_hat":>6s} | {"auc_v":>5s} {"auc_c":>5s}', flush=True)
    rows = []
    for s in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            van = cv_pick_mu(X, y, K=K, mu_grid=np.array([0.0]), q=None,
                             n_splits=N_SPLITS, rng=np.random.default_rng(s))
            cau = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=q,
                             n_splits=N_SPLITS, rng=np.random.default_rng(s))
        mu_rel = cau.mu_star / mu_scale if mu_scale else 0.0
        rows.append((van.stability_star, cau.stability_star, mu_rel,
                     van.auc_star, cau.auc_star, tuple(van.support), tuple(cau.support)))
        print(f'{s:>4d} | {van.stability_star:>8.3f} {cau.stability_star:>8.3f} '
              f'{cau.stability_star-van.stability_star:>+7.3f} | {mu_rel:>6.2f} | '
              f'{van.auc_star:>5.3f} {cau.auc_star:>5.3f}', flush=True)

    sv = np.array([r[0] for r in rows]); sc = np.array([r[1] for r in rows])
    av = np.array([r[3] for r in rows]); ac = np.array([r[4] for r in rows])
    print(f'\nMEAN | stab_van={sv.mean():.3f}  stab_cau={sc.mean():.3f}  '
          f'd={sc.mean()-sv.mean():+.3f}  causal>=vanilla {np.mean(sc>=sv):.0%}  '
          f'| AUC {av.mean():.3f}->{ac.mean():.3f}', flush=True)

    names = lambda sup: [feat_names[j] for j in sup]
    print('\nseed 0 vanilla support:', names(rows[0][5]), flush=True)
    print('seed 0 causal  support:', names(rows[0][6]), flush=True)
    pepq_in = ['pepq_Ala87Gly' in names(r[6]) for r in rows]
    print(f'pepq_Ala87Gly in causal support: {sum(pepq_in)}/{len(rows)} seeds', flush=True)


if __name__ == '__main__':
    main()
