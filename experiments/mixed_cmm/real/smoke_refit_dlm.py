"""Smoke test: refit CMM on full delamanid data with stable graph from with_lineage run.

Verifies that refit_with_stable_graph runs to completion and exposes betas, gammas,
cluster assignments for the MIC node. Prints shapes and a small summary.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.causal_discovery.cmm_utils import refit_with_stable_graph
from src.data.load_tb import load_tb_data, prevalence_filter, lineage_dummies


DATA_PATH = REPO_ROOT / 'data' / 'real' / 'processed' / 'tb_pheno_geno_clean.csv'
STABILITY_CSV = REPO_ROOT / 'results' / 'mixed_cmm' / 'subsampling' / 'tb_subsampling_dlm_mp4_k6_mcc4' / 'with_lineage' / 'edge_stability.csv'
MIC_COL = 'dlm_mic'
THRESHOLD = 0.5


def main():
    df, mutation_cols, _, _, _ = load_tb_data(str(DATA_PATH))
    df = df.dropna(subset=[MIC_COL]).reset_index(drop=True)
    df[MIC_COL] = np.log2(df[MIC_COL])
    keep = prevalence_filter(df, mutation_cols, min_prev=0.05, max_prev=0.98)
    df_lin = lineage_dummies(df, drop_first=True, merge_below=5)
    lin_cols = list(df_lin.columns)
    features = [MIC_COL] + keep + lin_cols
    X = pd.concat([df[[MIC_COL] + keep], df_lin], axis=1).values
    name_to_idx = {f: i for i, f in enumerate(features)}
    print(f"X shape: {X.shape}, features: {len(features)}", flush=True)

    stab = pd.read_csv(STABILITY_CSV)
    stable = stab[stab['frequency'] >= THRESHOLD]
    edges = [(name_to_idx[s], name_to_idx[t]) for s, t in zip(stable['source'], stable['target'])
             if s in name_to_idx and t in name_to_idx]
    print(f"stable edges at threshold {THRESHOLD}: {len(edges)}", flush=True)
    for s, t in stable[['source', 'target']].values:
        if s in name_to_idx and t in name_to_idx:
            print(f"  {s} -> {t}", flush=True)

    cmm, kept = refit_with_stable_graph(
        X, edges, use_logistic=True, k_max=6, max_parents=4,
        min_cluster_count=4, seed=0,
    )
    kept_features = [features[i] for i in kept]
    print(f"kept after sparse drop: {len(kept)} features", flush=True)
    print(f"dropped: {[features[i] for i in range(len(features)) if i not in kept]}", flush=True)

    mic_new_idx = kept_features.index(MIC_COL)
    betas_mic = cmm.betas.get(mic_new_idx)
    gammas_mic = cmm.gammas.get(mic_new_idx)
    idls_mic = cmm.idls.get(mic_new_idx)
    pprobas_mic = cmm.pprobas.get(mic_new_idx)

    print()
    print("=== MIC node mixture summary ===", flush=True)
    print(f"MIC parents (kept-indexed): {[(p, kept_features[p]) for p in cmm.dag.predecessors(mic_new_idx)]}", flush=True)
    print(f"betas len: {len(betas_mic)} (K clusters for MIC)", flush=True)
    for k, entry in enumerate(betas_mic):
        if isinstance(entry, tuple) and len(entry) == 2:
            beta_k, sigma_k = entry
            print(f"  cluster {k}: beta={np.array(beta_k).ravel().round(3).tolist()}, sigma={sigma_k}", flush=True)
        else:
            print(f"  cluster {k}: {entry}", flush=True)
    print(f"gammas (mixing weights): {np.array(gammas_mic).round(3).tolist()}", flush=True)
    print(f"idls len: {len(idls_mic)}, unique: {np.unique(idls_mic).tolist()}", flush=True)
    unique, counts = np.unique(idls_mic, return_counts=True)
    print(f"cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}", flush=True)
    pp = np.asarray(pprobas_mic)
    print(f"pprobas shape: {pp.shape}, row-sum check: min={pp.sum(axis=1).min():.3f} max={pp.sum(axis=1).max():.3f}", flush=True)


if __name__ == '__main__':
    main()
