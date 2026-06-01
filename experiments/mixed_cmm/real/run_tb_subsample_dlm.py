"""
Stability selection (subsampling) for CMM-logistic on real TB data (delamanid only).
Builds [dlm_mic, ...mutations] (prevalence-filtered), runs subsample_cmm with logistic
driver, and saves per-edge selection frequencies. Threshold and stable-graph extraction
are downstream concerns.

Usage:
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --n_runs 200 --min_prev 0.1
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --max_parents 6 --k_max 7
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --include_lineage --lineage_merge_below 5
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --include_lineage --forbid_lineage_to_mic
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --include_type
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --include_type --forbid_type_to_mic
"""
import sys
import argparse
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from src_tb.data.load_tb import load_tb_data, prevalence_filter, lineage_dummies, type_beyond_MDR
from src_tb.causal_discovery.cmm_utils import subsample_cmm, edge_stability, per_node_k_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / 'data' / 'real' / 'processed' / 'tb_pheno_geno_clean.csv'
MIC_COL = 'dlm_mic'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=100, help='stability-selection iterations')
    parser.add_argument('--subsample_frac', type=float, default=0.8, help='fraction of rows per run')
    parser.add_argument('--max_parents', type=int, default=4, help='cap on in-degree per node')
    parser.add_argument('--k_max', type=int, default=5, help='max mixture components per node')
    parser.add_argument('--min_cluster_count', type=int, default=5, help='min positives per cluster for binary cols')
    parser.add_argument('--min_prev', type=float, default=0.05)
    parser.add_argument('--max_prev', type=float, default=0.98)
    parser.add_argument('--include_lineage', action='store_true', help='one-hot lineage as covariate')
    parser.add_argument('--lineage_merge_below', type=int, default=None,
                        help='pool lineages with count < N into the reference category (e.g. 5 merges L1+L3)')
    parser.add_argument('--forbid_lineage_to_mic', action='store_true',
                        help='forbid lineage->MIC edges (default: allow, lets lineage absorb its direct effect on MIC)')
    parser.add_argument('--include_type', action='store_true',
                        help='add binary type_beyond_MDR (preXDR/XDR vs MDR) as exogenous covariate')
    parser.add_argument('--forbid_type_to_mic', action='store_true',
                        help='forbid type->MIC edges (default: allow, lets type absorb its direct effect on MIC)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    df, mutation_cols, _, _, _ = load_tb_data(str(DATA_PATH))
    n_before = len(df)
    df = df.dropna(subset=[MIC_COL]).reset_index(drop=True)
    print(f"isolates with {MIC_COL}: {len(df)} / {n_before}", flush=True)
    # MIC is measured on a 2-fold dilution series; log2 puts dilution steps on a uniform scale.
    df[MIC_COL] = np.log2(df[MIC_COL])
    keep = prevalence_filter(df, mutation_cols, min_prev=args.min_prev, max_prev=args.max_prev)

    if args.include_lineage:
        df_lin = lineage_dummies(df, drop_first=True, merge_below=args.lineage_merge_below)
        lin_cols = list(df_lin.columns)
    else:
        df_lin = None
        lin_cols = []

    if args.include_type:
        type_series = type_beyond_MDR(df).rename('type_beyond_MDR')
        type_cols = ['type_beyond_MDR']
    else:
        type_series = None
        type_cols = []

    features = [MIC_COL] + keep + lin_cols + type_cols
    pieces = [df[[MIC_COL] + keep]]
    if df_lin is not None:
        pieces.append(df_lin)
    if type_series is not None:
        pieces.append(type_series.to_frame())
    X = pd.concat(pieces, axis=1).values

    mic_idx = 0
    mut_idx = list(range(1, 1 + len(keep)))
    lin_idx = list(range(1 + len(keep), 1 + len(keep) + len(lin_cols)))
    type_idx = list(range(1 + len(keep) + len(lin_cols),
                          1 + len(keep) + len(lin_cols) + len(type_cols)))

    forbidden = set()
    # MIC -> mutation: mutations cause MIC, never the reverse
    for j in mut_idx:
        forbidden.add((mic_idx, j))
    if args.include_lineage:
        # lineage is exogenous: forbid edges INTO lineage from anything (incl. other lineage dummies)
        for source in [mic_idx] + mut_idx + lin_idx + type_idx:
            for target in lin_idx:
                if source != target:
                    forbidden.add((source, target))
        if args.forbid_lineage_to_mic:
            for source in lin_idx:
                forbidden.add((source, mic_idx))
    if args.include_type:
        # type is a clinical classification set before/independently of MIC: forbid edges INTO it
        for source in [mic_idx] + mut_idx + lin_idx + type_idx:
            for target in type_idx:
                if source != target:
                    forbidden.add((source, target))
        if args.forbid_type_to_mic:
            for source in type_idx:
                forbidden.add((source, mic_idx))

    col_threshold = args.min_cluster_count * args.k_max
    print(f"mutations after prevalence filter: {len(keep)}, X shape: {X.shape}", flush=True)
    print(f"include_lineage={args.include_lineage}, lineage cols: {lin_cols}", flush=True)
    print(f"include_type={args.include_type}, type cols: {type_cols}", flush=True)
    print(f"k_max={args.k_max}, min_cluster_count={args.min_cluster_count}, col_threshold={col_threshold}", flush=True)
    sparse = [f for f in keep if df[f].sum() < col_threshold]
    print(f"mutations below col_threshold ({col_threshold}): {sparse}", flush=True)

    suffix_parts = []
    if args.include_lineage:
        suffix_parts.append('lineage')
        if args.lineage_merge_below is not None:
            suffix_parts.append(f'mb{args.lineage_merge_below}')
        if args.forbid_lineage_to_mic:
            suffix_parts.append('fbl')
    if args.include_type:
        suffix_parts.append('type')
        if args.forbid_type_to_mic:
            suffix_parts.append('fbt')
    suffix_parts.extend([f'mp{args.max_parents}', f'k{args.k_max}', f'mcc{args.min_cluster_count}'])
    suffix = '_' + '_'.join(suffix_parts)
    base = REPO_ROOT / 'results' / 'subsampling' / f'tb_subsample_dlm{suffix}'
    output_dir = base
    i = 2
    while output_dir.exists():
        output_dir = base.with_name(base.name + f'_{i}')
        i += 1
    output_dir.mkdir(parents=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**vars(args), 'mic_col': MIC_COL, 'features': features}, f, indent=2)

    cmm_list, features_per_run = subsample_cmm(X, forbidden, n_runs=args.n_runs, use_logistic=True,
                                               max_parents=args.max_parents, k_max=args.k_max,
                                               min_cluster_count=args.min_cluster_count,
                                               subsample_frac=args.subsample_frac,
                                               features=features, seed=args.seed)
    df_stability = edge_stability(cmm_list, features_per_run).sort_values('frequency', ascending=False)
    df_stability.to_csv(output_dir / 'edge_stability.csv', index=False)

    df_k = per_node_k_summary(cmm_list, features_per_run)
    df_k.to_csv(output_dir / 'per_node_k.csv', index=False)

    print(f"Done. Results in {output_dir}/", flush=True)


if __name__ == '__main__':
    main()
