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
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from src.data.load_tb import load_tb_data, prevalence_filter, lineage_dummies, type_beyond_MDR
from src.causal_discovery.cmm_utils import subsample_cmm, edge_stability, per_node_k_summary
from experiments._io import new_run_dir

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
    parser.add_argument('--lineage_exogenous', action='store_true',
                        help='legacy orientation: treat lineage as an exogenous common cause (forbid '
                             'every edge into it). default is the corrected biology where lineage is '
                             'determined by mutations (mutation->lineage allowed, lineage->mutation forbidden)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='explicit output directory (overrides the auto-named results path)')
    parser.add_argument('--data', type=str, default=str(DATA_PATH),
                        help='dataset csv (default: binary clean set; pass tb_freq_clean.csv for freq data)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_shards', type=int, default=1,
                        help='>1 self-shards this run across that many cores (needs --out_dir)')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def build_forbidden(mic_idx, mut_idx, lin_idx, type_idx, *, include_lineage, include_type,
                    forbid_lineage_to_mic, forbid_type_to_mic, lineage_exogenous):
    """forbidden-edge mask encoding the causal assumptions for the (single- or multi-drug) graph.

    mic_idx is one index or an iterable of them. every mic is a sink among the modelled variables:
    mutations/lineage/type cause mic, never the reverse, and mic<->mic is forbidden in both
    directions so two drug mics can only correlate through a shared mutation parent (this is how
    cross-resistance reads out). lineage orientation is switchable:
      - default (corrected biology, bio team): lineage is determined by mutations, so allow
        mutation->lineage and lineage->mic, but forbid lineage->mutation, lineage<->type,
        mic->lineage and lineage->lineage.
      - lineage_exogenous=True (legacy): lineage is an exogenous common cause; forbid every edge
        into it. kept only to reproduce the earlier ablation runs.
    """
    mics = [mic_idx] if isinstance(mic_idx, int) else list(mic_idx)
    forbidden = set()
    for m in mics:
        for j in mut_idx:
            forbidden.add((m, j))  # mutations cause mic, never the reverse
    for a in mics:
        for b in mics:
            if a != b:
                forbidden.add((a, b))  # mics do not cause each other (cross-resistance via shared parents)

    if include_lineage:
        if lineage_exogenous:
            for source in mics + mut_idx + lin_idx + type_idx:
                for target in lin_idx:
                    if source != target:
                        forbidden.add((source, target))
        else:
            # lineage is downstream of mutations: it cannot be a parent of mutations, of type,
            # or of another lineage dummy
            for source in lin_idx:
                for target in mut_idx + type_idx + lin_idx:
                    if source != target:
                        forbidden.add((source, target))
            # nothing downstream sets lineage: mic and type cannot point into it
            # (mutation->lineage stays allowed)
            for source in mics + type_idx:
                for target in lin_idx:
                    forbidden.add((source, target))
        if forbid_lineage_to_mic:
            for source in lin_idx:
                for m in mics:
                    forbidden.add((source, m))

    if include_type:
        # type is a clinical classification set before/independently of mic: forbid edges into it
        for source in mics + mut_idx + lin_idx + type_idx:
            for target in type_idx:
                if source != target:
                    forbidden.add((source, target))
        if forbid_type_to_mic:
            for source in type_idx:
                for m in mics:
                    forbidden.add((source, m))
    return forbidden


def main():
    args = parse_args()
    if args.n_shards > 1 and not args._worker:  # orchestrate: shard this run across cores
        from experiments._io import self_shard
        self_shard(args.out_dir, args.n_runs, args.n_shards)
        return
    df, mutation_cols, _, _, _ = load_tb_data(args.data)
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

    forbidden = build_forbidden(
        mic_idx, mut_idx, lin_idx, type_idx,
        include_lineage=args.include_lineage, include_type=args.include_type,
        forbid_lineage_to_mic=args.forbid_lineage_to_mic,
        forbid_type_to_mic=args.forbid_type_to_mic,
        lineage_exogenous=args.lineage_exogenous)

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
    if args.out_dir is not None:
        base = Path(args.out_dir)
    else:
        base = REPO_ROOT / 'results' / 'mixed_cmm' / 'subsampling' / f'tb_subsample_dlm{suffix}'
    output_dir = new_run_dir(base, {**vars(args), 'mic_col': MIC_COL, 'features': features})

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
