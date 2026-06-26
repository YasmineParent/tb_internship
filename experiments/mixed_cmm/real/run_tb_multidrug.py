"""Joint multi-drug causal graph for cross-resistance on real TB data.

Puts several drug MICs into ONE graph as outcome nodes, alongside common mutations, rare-variant
pathway burdens (collapse_burden), lineage and type. mic<->mic edges are forbidden, so two drug
mics can only correlate through a shared mutation/burden parent: a feature that parents several
mics is the cross-resistance driver. lineage uses the corrected orientation (downstream of
mutations, can cause mic) from run_tb_subsample_dlm.build_forbidden.

linezolid is excluded by default (3 mic levels, no resistant isolates -> near-constant continuous
node that destabilises the mixture fit). Run sharded for speed (see run_tb_ablation_dlm.py pattern).

Usage:
    python experiments/mixed_cmm/real/run_tb_multidrug.py
    python experiments/mixed_cmm/real/run_tb_multidrug.py --drugs dlm,ptm,bdq,cfz --n_runs 100
    python experiments/mixed_cmm/real/run_tb_multidrug.py --no_burden --include_type
"""
import sys
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from src.data.load_tb import (load_tb_data, collapse_burden, lineage_dummies, type_beyond_MDR,
                              RESISTANCE_PATHWAYS)
from src.causal_discovery.cmm_utils import subsample_cmm, edge_stability, per_node_k_summary
from experiments._io import new_run_dir
from experiments.mixed_cmm.real.run_tb_subsample_dlm import build_forbidden

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / 'data' / 'real' / 'processed' / 'tb_pheno_geno_clean.csv'
# which pathway burden is relevant to which drug (drives the linezolid-burden drop when lnz absent)
PATHWAY_DRUGS = {'f420_activation': {'dlm', 'ptm'}, 'efflux': {'bdq', 'cfz'}, 'linezolid': {'lnz'}}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--drugs', type=str, default='dlm,ptm,bdq,cfz', help='comma-separated drug prefixes')
    p.add_argument('--n_runs', type=int, default=100)
    p.add_argument('--subsample_frac', type=float, default=0.8)
    p.add_argument('--max_parents', type=int, default=4)
    p.add_argument('--k_max', type=int, default=6)
    p.add_argument('--min_cluster_count', type=int, default=4)
    p.add_argument('--min_prev', type=float, default=0.05)
    p.add_argument('--max_prev', type=float, default=0.98)
    p.add_argument('--presence_threshold', type=float, default=0.0,
                   help='value above which a mutation is "present" (use 5 for the 0-100 freq data)')
    p.add_argument('--no_burden', action='store_true', help='drop the rare-variant pathway burdens')
    p.add_argument('--include_lineage', action='store_true', default=True)
    p.add_argument('--no_lineage', dest='include_lineage', action='store_false')
    p.add_argument('--lineage_merge_below', type=int, default=5)
    p.add_argument('--include_type', action='store_true')
    p.add_argument('--lineage_exogenous', action='store_true', help='legacy lineage orientation')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    drugs = [d.strip() for d in args.drugs.split(',')]
    mic_cols = [f'{d}_mic' for d in drugs]

    df, mutation_cols, _, _, _ = load_tb_data(str(DATA_PATH))
    df = df.dropna(subset=mic_cols).reset_index(drop=True)
    print(f"isolates with all of {mic_cols}: {len(df)}", flush=True)
    # mics are 2-fold dilution series; log2 puts dilution steps on a uniform scale
    for c in mic_cols:
        df[c] = np.log2(df[c])

    single_cols, burden_df = collapse_burden(
        df, mutation_cols, min_prev=args.min_prev, max_prev=args.max_prev,
        presence_threshold=args.presence_threshold)
    if args.no_burden:
        burden_cols, burden_df = [], pd.DataFrame(index=df.index)
    else:
        # keep only burdens whose pathway is relevant to a selected drug
        keep_burden = [p for p in burden_df.columns if PATHWAY_DRUGS.get(p, set()) & set(drugs)]
        burden_df = burden_df[keep_burden].add_prefix('burden_')
        burden_cols = list(burden_df.columns)
    mut_cols = single_cols + burden_cols

    if args.include_lineage:
        df_lin = lineage_dummies(df, drop_first=True, merge_below=args.lineage_merge_below)
        lin_cols = list(df_lin.columns)
    else:
        df_lin, lin_cols = pd.DataFrame(index=df.index), []
    if args.include_type:
        type_df = type_beyond_MDR(df).rename('type_beyond_MDR').to_frame()
        type_cols = ['type_beyond_MDR']
    else:
        type_df, type_cols = pd.DataFrame(index=df.index), []

    features = mic_cols + mut_cols + lin_cols + type_cols
    X = pd.concat([df[mic_cols], df[single_cols], burden_df, df_lin, type_df], axis=1).values

    nmic, nmut, nlin = len(mic_cols), len(mut_cols), len(lin_cols)
    mic_idx = list(range(nmic))
    mut_idx = list(range(nmic, nmic + nmut))
    lin_idx = list(range(nmic + nmut, nmic + nmut + nlin))
    type_idx = list(range(nmic + nmut + nlin, nmic + nmut + nlin + len(type_cols)))

    forbidden = build_forbidden(
        mic_idx, mut_idx, lin_idx, type_idx,
        include_lineage=args.include_lineage, include_type=args.include_type,
        forbid_lineage_to_mic=False, forbid_type_to_mic=False,
        lineage_exogenous=args.lineage_exogenous)

    print(f"mics={mic_cols}\nsingle mutations ({len(single_cols)}): {single_cols}", flush=True)
    print(f"pathway burdens: {burden_cols}", flush=True)
    print(f"lineage={lin_cols} type={type_cols}  X shape={X.shape}", flush=True)

    base = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / 'results' / 'mixed_cmm' / 'subsampling' / f'tb_multidrug_{"_".join(drugs)}')
    output_dir = new_run_dir(base, {**vars(args), 'mic_cols': mic_cols, 'features': features})

    cmm_list, feats = subsample_cmm(
        X, forbidden, n_runs=args.n_runs, use_logistic=True, max_parents=args.max_parents,
        k_max=args.k_max, min_cluster_count=args.min_cluster_count,
        subsample_frac=args.subsample_frac, features=features, seed=args.seed)
    edge_stability(cmm_list, feats).sort_values('frequency', ascending=False).to_csv(
        output_dir / 'edge_stability.csv', index=False)
    per_node_k_summary(cmm_list, feats).to_csv(output_dir / 'per_node_k.csv', index=False)
    print(f"Done. Results in {output_dir}/", flush=True)


if __name__ == '__main__':
    main()
