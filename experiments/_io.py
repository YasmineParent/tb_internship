"""Shared I/O and sharding helpers for experiment scripts."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# pin BLAS to one thread per shard so n_shards processes saturate n_shards cores cleanly
SINGLE_THREAD = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                 'VECLIB_MAXIMUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def new_run_dir(base: Path, config: dict | None = None) -> Path:
    """Create a fresh results directory that never overwrites an existing one.

    If `base` already exists, appends `_2`, `_3`, ... until an unused name is
    found. When `config` is given, dumps it to `config.json` inside the new
    directory. Returns the created directory.
    """
    out, i = base, 2
    while out.exists():
        out = base.with_name(base.name + f'_{i}')
        i += 1
    out.mkdir(parents=True)
    if config is not None:
        (out / 'config.json').write_text(json.dumps(config, indent=2))
    return out


def self_shard(out_dir: str | Path, n_runs: int, n_shards: int):
    """Re-invoke the calling script as n_shards parallel workers and aggregate.

    A CMM run is ~80-140s per subsample, so this splits n_runs into shards (distinct seeds,
    single-threaded BLAS) launched in parallel, then aggregates; equivalent to one long run.
    Call from a script's main when n_shards > 1; the worker branch (when '--_worker' is in argv)
    should run one sequential job into --out_dir. The script's own --n_runs/--seed/--out_dir/
    --n_shards are stripped and replaced per shard; all other args pass through unchanged."""
    ctrl, keep, drop_next = {'--n_shards', '--n_runs', '--seed', '--out_dir'}, [], False
    for a in sys.argv[1:]:
        if drop_next:
            drop_next = False
        elif a in ctrl:
            drop_next = True
        else:
            keep.append(a)
    run_sharded(Path(sys.argv[0]), keep + ['--_worker'], Path(out_dir), n_runs, n_shards)


def run_sharded(script: Path, flags: list[str], out_dir: Path, n_runs: int, n_shards: int,
                label: str | None = None) -> Path:
    """Launch n_shards parallel subsample jobs of `script` into out_dir, wait, and aggregate.

    flags must already include every script option except --n_runs/--seed/--out_dir, which this
    adds per shard. `script` must accept those three options and emit edge_stability.csv and
    per_node_k.csv."""
    label = label or out_dir.name
    per, extra = divmod(n_runs, n_shards)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **SINGLE_THREAD}

    procs, shard_dirs = [], []
    for s in range(n_shards):
        runs_s = per + (1 if s < extra else 0)
        if runs_s == 0:
            continue
        shard_dir = out_dir / f'shard_{s}'
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        cmd = [sys.executable, str(script), *flags,
               '--n_runs', str(runs_s), '--seed', str(s), '--out_dir', str(shard_dir)]
        log = open(out_dir / f'shard_{s}.log', 'w')
        procs.append((s, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        shard_dirs.append(shard_dir)

    t0, failed = time.perf_counter(), []
    for s, proc, log in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            failed.append(s)
    if failed:
        raise RuntimeError(f"{label}: shards {failed} failed; see {out_dir}/shard_*.log")

    _aggregate(shard_dirs, out_dir)
    print(f"[{label}] {n_runs} subsamples over {len(shard_dirs)} shards in "
          f"{(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}/edge_stability.csv", flush=True)
    return out_dir


def _aggregate(shard_dirs: list[Path], out_dir: Path):
    """Combine per-shard edge_stability and per_node_k tables into out_dir."""
    edges = pd.concat([pd.read_csv(d / 'edge_stability.csv') for d in shard_dirs], ignore_index=True)
    agg = edges.groupby(['source', 'target'], as_index=False)[['count', 'n_eligible']].sum()
    agg['frequency'] = agg['count'] / agg['n_eligible']
    agg.sort_values('frequency', ascending=False).to_csv(out_dir / 'edge_stability.csv', index=False)

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
    cum = counts.cumsum(axis=1)
    g['median_k'] = [float(kvals[np.searchsorted(cum[i], totals[i] / 2.0)]) if totals[i] else float('nan')
                     for i in range(len(g))]
    g.sort_values('mean_k', ascending=False).to_csv(out_dir / 'per_node_k.csv', index=False)
