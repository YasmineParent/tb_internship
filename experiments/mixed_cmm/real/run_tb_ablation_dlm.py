"""Run the corrected delamanid lineage ablation, sharded across cores.

cmm is ~80s per subsample fit, so 100 subsamples x several conditions runs into hours
serially. this driver splits each condition's n_runs into shards (distinct seeds) run in
parallel with single-threaded blas, then aggregates the per-edge and per-node-k tables.

sharded stability selection is statistically equivalent to one long run: the subsamples are
iid draws, so the combined selection frequency is exactly total_selections / total_eligible
(meinshausen-buhlmann). per-node k is aggregated from the per-shard count histograms.

corrected orientation (bio team, jun 2026): lineage is determined by mutations, so the
default mask allows mutation->lineage and lineage->mic but forbids lineage->mutation. only
the lineage-bearing conditions change; baseline is lineage-free and is reused as committed.

usage:
    python experiments/mixed_cmm/real/run_tb_ablation_dlm.py
    python experiments/mixed_cmm/real/run_tb_ablation_dlm.py --n_runs 100 --n_shards 10
    python experiments/mixed_cmm/real/run_tb_ablation_dlm.py --smoke
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'experiments' / 'mixed_cmm' / 'real' / 'run_tb_subsample_dlm.py'
COMMITTED = ROOT / 'results' / 'mixed_cmm' / 'subsampling' / 'tb_subsampling_dlm_mp4_k6_mcc4'
OUT_PARENT = ROOT / 'results' / 'mixed_cmm' / 'subsampling' / 'tb_subsampling_dlm_mp4_k6_mcc4_linfix'

COMMON = ['--max_parents', '4', '--k_max', '6', '--min_cluster_count', '4']
# corrected orientation is the script default; these are the two conditions the fix changes.
CONDITIONS = {
    'with_lineage': ['--include_lineage', '--lineage_merge_below', '5'],
    'with_lineage_and_type': ['--include_lineage', '--lineage_merge_below', '5', '--include_type'],
}
SINGLE_THREAD = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                 'VECLIB_MAXIMUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_runs', type=int, default=100, help='total subsamples per condition')
    p.add_argument('--n_shards', type=int, default=10, help='parallel shards per condition')
    p.add_argument('--smoke', action='store_true', help='tiny run to check the harness end to end')
    args = p.parse_args()
    if args.smoke:
        args.n_runs, args.n_shards = 4, 2
    return args


def run_condition(name: str, flags: list[str], n_runs: int, n_shards: int) -> Path:
    """Launch n_shards parallel subsample jobs for one condition, wait, and aggregate."""
    per = n_runs // n_shards
    extra = n_runs - per * n_shards  # spread any remainder over the first shards
    cond_dir = OUT_PARENT / name
    cond_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **SINGLE_THREAD}

    procs, shard_dirs = [], []
    for s in range(n_shards):
        runs_s = per + (1 if s < extra else 0)
        if runs_s == 0:
            continue
        shard_dir = cond_dir / f'shard_{s}'
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        cmd = [sys.executable, str(SCRIPT), *flags, *COMMON,
               '--n_runs', str(runs_s), '--seed', str(s), '--out_dir', str(shard_dir)]
        log = open(cond_dir / f'shard_{s}.log', 'w')
        procs.append((s, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        shard_dirs.append(shard_dir)

    t0 = time.perf_counter()
    failed = []
    for s, proc, log in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            failed.append(s)
    dt = time.perf_counter() - t0
    if failed:
        raise RuntimeError(f"{name}: shards {failed} failed; see {cond_dir}/shard_*.log")

    aggregate(shard_dirs, cond_dir)
    print(f"[{name}] {n_runs} subsamples over {len(shard_dirs)} shards in {dt/60:.1f} min "
          f"-> {cond_dir}/edge_stability.csv", flush=True)
    return cond_dir


def aggregate(shard_dirs: list[Path], cond_dir: Path):
    """Combine per-shard edge_stability and per_node_k tables into the condition root."""
    edges = pd.concat([pd.read_csv(d / 'edge_stability.csv') for d in shard_dirs], ignore_index=True)
    agg = edges.groupby(['source', 'target'], as_index=False)[['count', 'n_eligible']].sum()
    agg['frequency'] = agg['count'] / agg['n_eligible']
    agg.sort_values('frequency', ascending=False).to_csv(cond_dir / 'edge_stability.csv', index=False)

    ks = pd.concat([pd.read_csv(d / 'per_node_k.csv') for d in shard_dirs], ignore_index=True)
    kcols = sorted([c for c in ks.columns if c.startswith('k') and c.endswith('_count')],
                   key=lambda c: int(c[1:-6]))
    ks[kcols] = ks[kcols].fillna(0)
    g = ks.groupby('feature', as_index=False)[['n_runs_present'] + kcols].sum()
    kvals = np.array([int(c[1:-6]) for c in kcols])
    counts = g[kcols].to_numpy()
    totals = counts.sum(axis=1)
    g['mean_k'] = (counts * kvals).sum(axis=1) / np.where(totals == 0, 1, totals)
    g['mode_k'] = kvals[counts.argmax(axis=1)]
    g['median_k'] = [_hist_median(kvals, row) for row in counts]
    g.sort_values('mean_k', ascending=False).to_csv(cond_dir / 'per_node_k.csv', index=False)


def _hist_median(kvals: np.ndarray, counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float('nan')
    cum = np.cumsum(counts)
    return float(kvals[np.searchsorted(cum, total / 2.0)])


def main():
    args = parse_args()
    OUT_PARENT.mkdir(parents=True, exist_ok=True)

    # baseline is lineage-free: the orientation fix does not touch it, so reuse the committed run.
    base_src = COMMITTED / 'baseline'
    base_dst = OUT_PARENT / 'baseline'
    if base_src.exists() and not base_dst.exists():
        shutil.copytree(base_src, base_dst)
        print(f"[baseline] reused committed run (lineage-free, unchanged) -> {base_dst}", flush=True)

    for name, flags in CONDITIONS.items():
        run_condition(name, flags, args.n_runs, args.n_shards)

    print("\n=== dlm_mic parents (corrected orientation) ===", flush=True)
    for name in CONDITIONS:
        df = pd.read_csv(OUT_PARENT / name / 'edge_stability.csv')
        into_mic = df[df['target'] == 'dlm_mic'].sort_values('frequency', ascending=False)
        print(f"\n[{name}]", flush=True)
        print(into_mic[['source', 'frequency', 'count', 'n_eligible']].to_string(index=False), flush=True)
    print(f"\nDone. Results in {OUT_PARENT}/", flush=True)


if __name__ == '__main__':
    main()
