"""§6.3 first pass: causal-prior FasterRisk vs vanilla on the real TB-DLM data.

Experiment 3.2 (prevalence-filtered). Target = agar delamanid R/S call
(interp_dlm016); features = prevalence-feasible [5%, 98%] mutations in the
delamanid-pathway genes + lineage indicators; causal prior q_j = stability
frequency of the edge (j -> dlm_mic) from the CMM stability-selection run
(with_lineage_forbid_to_mic spec, the §7.3 type-forbid decision spec).

Compares vanilla FasterRisk (mu=0) against causal FasterRisk (mu picked by
log-loss CV) at matched sparsity K, using the validated cv_pick_mu. Reports
the three pre-registered success criteria:
  1. mu_hat > 0   (the prior earns its place under log-loss CV)
  2. cross-fold support stability(causal) >= stability(vanilla)
  3. selected support includes the §7.3 candidate(s) / is biologically defensible
AUC is reported for reference only (wide CIs at this n; not the headline).
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
DLM_GENES = ('rv0678', 'mmpl5', 'mmps5', 'atpe', 'pepq', 'rv1979c', 'fgd1', 'ddn')
PREV_LO, PREV_HI = 0.05, 0.98
K = 5
N_SPLITS = 10
SEED = 0


def main() -> None:
    df = pd.read_csv(DATA)

    # --- target: agar R/S call, drop missing ---
    y_raw = df['interp_dlm016']
    keep = y_raw.notna()
    df = df[keep].reset_index(drop=True)
    y = np.where(df['interp_dlm016'].to_numpy() == 2.0, 1, -1).astype(int)

    # --- features: prevalence-feasible dlm-pathway mutations + lineage indicators ---
    mut_cols = [c for c in df.columns if any(c.startswith(g + '_') for g in DLM_GENES)]
    M = df[mut_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0)
    prev = M.mean()
    feasible = sorted(prev[(prev >= PREV_LO) & (prev <= PREV_HI)].index)

    lineage = df['lineage'].astype('Int64')
    lin_feats = {}
    for lv in (2, 4):  # non-reference lineages present as CMM nodes; 1/3 merged into reference
        col = f'lineage_{lv}'
        lin_feats[col] = (lineage == lv).astype(float).to_numpy()

    feat_names = feasible + list(lin_feats.keys())
    X = np.column_stack([M[feasible].to_numpy()] + [lin_feats[c] for c in lin_feats])

    # --- causal prior q: freq(feature -> dlm_mic) from CMM stability ---
    edges = pd.read_csv(QCSV)
    into_mic = edges[edges['target'] == MIC_NODE].set_index('source')['frequency']
    q = np.array([float(into_mic.get(name, 0.0)) for name in feat_names])

    n_pos = int((y == 1).sum())
    print(f'n={len(y)}  resistant={n_pos} ({n_pos/len(y):.0%})  p={len(feat_names)}  K={K}')
    print('features (prevalence, q):')
    for name in feat_names:
        pv = float(M[name].mean()) if name in M.columns else float(np.mean(dict(zip(feat_names, X.T))[name]))
        print(f'  {name:22s} prev={pv:.2f}  q={q[feat_names.index(name)]:.2f}')
    print(f'nonzero q on {int((q>0).sum())} features; max q={q.max():.2f}\n')

    # --- mu grid (same construction as §6.1) ---
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_rel_grid = np.concatenate([[0.0], np.logspace(-2, 1, 12)])
    mu_grid = mu_rel_grid * mu_scale

    rng = np.random.default_rng(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vanilla = cv_pick_mu(X, y, K=K, mu_grid=np.array([0.0]), q=None,
                             n_splits=N_SPLITS, criterion='log_loss',
                             rng=np.random.default_rng(SEED))
        causal = cv_pick_mu(X, y, K=K, mu_grid=mu_grid, q=q,
                            n_splits=N_SPLITS, criterion='log_loss',
                            rng=np.random.default_rng(SEED))

    def names(support):
        return [feat_names[j] for j in support]

    print('=== VANILLA (mu=0) ===')
    print(f'  support: {names(vanilla.support)}')
    print(f'  10-fold support stability (Jaccard): {vanilla.stability_star:.3f}')
    print(f'  CV AUC: {vanilla.auc_star:.3f}   log-loss: {vanilla.log_loss_star:.3f}')
    print('=== CAUSAL (log-loss CV) ===')
    print(f'  mu_hat_relative: {causal.mu_star / mu_scale if mu_scale else 0:.3f}')
    print(f'  support: {names(causal.support)}')
    print(f'  10-fold support stability (Jaccard): {causal.stability_star:.3f}')
    print(f'  CV AUC: {causal.auc_star:.3f}   log-loss: {causal.log_loss_star:.3f}')

    mu_hat_rel = causal.mu_star / mu_scale if mu_scale else 0.0
    print('\n=== success criteria ===')
    print(f'  1. mu_hat > 0:                       {"PASS" if mu_hat_rel > 0 else "FAIL"} (mu_hat_rel={mu_hat_rel:.3f})')
    print(f'  2. stability(causal) >= vanilla:     {"PASS" if causal.stability_star >= vanilla.stability_star else "FAIL"} '
          f'({causal.stability_star:.3f} vs {vanilla.stability_star:.3f})')
    has_pepq = 'pepq_Ala87Gly' in names(causal.support)
    print(f'  3. support includes pepq_Ala87Gly:   {"PASS" if has_pepq else "FAIL"}')


if __name__ == '__main__':
    main()
