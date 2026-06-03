"""Load recovery_sweep CSVs and Phase A npz cells; compute the selectivity table.

The recovery CSVs (one per cell, written by recovery_sweep.py) hold one row per
fit and don't have the q vectors. To compute sel(q) and per-source q quality
we have to read q from the matching Phase A npz cells.

The two loaders are independent; selectivity_per_cell joins them via cell ids.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import selectivity
from .q_sources import oracle_q, uniform_q, adversarial_q


CELL_ID_COLS = ['seed', 'p', 'n', 'k_star', 'p_edge']


def load_recovery_csvs(out_dir: Path | str) -> pd.DataFrame:
    """Concatenate every recovery_sweep CSV in out_dir into one long DataFrame.

    The 'support' column is parsed from JSON back to list[int]; everything else
    keeps its CSV dtype.
    """
    out_dir = Path(out_dir)
    paths = sorted(out_dir.glob('seed*.csv'))
    if not paths:
        raise FileNotFoundError(f'no recovery CSVs in {out_dir}')
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df['support'] = df['support'].apply(json.loads)
    # legacy CSVs (pre-noise_scale commit) lack the column; they were all run at
    # the default noise level, so backfill rather than re-run them.
    if 'noise_scale' in df.columns:
        df['noise_scale'] = df['noise_scale'].fillna(1.0)
    else:
        df['noise_scale'] = 1.0
    return df


def load_phase_a_npz(cache_dir: Path | str) -> pd.DataFrame:
    """One row per cell with S_star, confounded, q vectors, mu_scale, ges_timeouts.

    Missing q sources are stored as None.
    """
    cache_dir = Path(cache_dir)
    paths = sorted(cache_dir.glob('seed*.npz'))
    if not paths:
        raise FileNotFoundError(f'no Phase A cells in {cache_dir}')
    rows: list[dict] = []
    for path in paths:
        cell = np.load(path, allow_pickle=False)
        row = {
            'seed':           int(cell['seed']),
            'p':              int(cell['p']),
            'n':              int(cell['n']),
            'k_star':         int(cell['k_star']),
            'p_edge':         float(cell['p_edge']),
            'mu_scale':       float(cell['mu_scale']),
            'ges_n_timeouts': int(cell['ges_n_timeouts']) if 'ges_n_timeouts' in cell.files else 0,
            'S_star':         sorted(int(j) for j in cell['S_star']),
            'confounded':     sorted(int(j) for j in cell['confounded']),
        }
        for key in ('q_pc', 'q_ges', 'q_bootstrap_l1'):
            row[key] = cell[key] if key in cell.files else None
        rows.append(row)
    return pd.DataFrame(rows)


def selectivity_per_cell(phase_a: pd.DataFrame) -> pd.DataFrame:
    """For each (cell, q_source) compute sel(q), bar_q_S, bar_q_C.

    Synthetic q sources (oracle sigma=0, uniform 0.5, adversarial) are
    reconstructed from S_star and confounded. Discovery sources (pc, ges,
    bootstrap_l1) come from the npz q columns; missing sources are dropped.
    """
    rows: list[dict] = []
    for _, cell in phase_a.iterrows():
        p = cell['p']
        S = cell['S_star']
        C = cell['confounded']
        sources: dict[str, np.ndarray] = {
            'oracle':      oracle_q(p, S, sigma=0.0),
            'uniform':     uniform_q(p, 0.5),
            'adversarial': adversarial_q(p, C),
        }
        for key, label in (('q_pc', 'pc'), ('q_ges', 'ges'),
                           ('q_bootstrap_l1', 'bootstrap_l1')):
            if cell[key] is not None:
                sources[label] = cell[key]

        for src, q in sources.items():
            q = np.asarray(q)
            rows.append({
                **{col: cell[col] for col in CELL_ID_COLS},
                'q_source': src,
                'sel_q':   selectivity(q, S, C),
                'bar_q_S': float(q[S].mean()) if S else float('nan'),
                'bar_q_C': float(q[C].mean()) if C else float('nan'),
            })
    return pd.DataFrame(rows)


def selectivity_summary(
    merged: pd.DataFrame,
    mu_relative: float = 1.0,
    metric: str = 'S_precision',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two pivot tables for the selectivity reframe: sel(q) and prior gain.

    Returns (sel_table, gain_table), each indexed by q_source with one column
    per p_edge value present in merged. Values are averaged across seeds.

    sel_table[q_source, p_edge]   = mean sel(q) = mean(q on C) / mean(q on S*)
    gain_table[q_source, p_edge]  = mean metric at mu_relative
                                    minus mean metric at mu_relative = 0
                                    (positive = the prior helps over vanilla)

    Hard-threshold baseline rows (q_source = '<src>_hard_t<t>') are excluded;
    use them via plot_soft_vs_hard for the §6.4 comparison.
    """
    soft = merged[~merged['q_source'].astype(str).str.contains('_hard_', na=False)]
    mu_vals = sorted(soft['mu_relative'].unique())
    mu_pick = min(mu_vals, key=lambda v: abs(v - mu_relative))

    sel_table = (soft.groupby(['q_source', 'p_edge'])['sel_q']
                     .mean().unstack().round(2))

    at_mu  = soft[soft['mu_relative'] == mu_pick].groupby(['q_source', 'p_edge'])[metric].mean()
    at_van = soft[soft['mu_relative'] == 0.0   ].groupby(['q_source', 'p_edge'])[metric].mean()
    gain_table = (at_mu - at_van).unstack().round(2)
    gain_table.name = f'{metric} gain (mu_rel={mu_pick:.2f} minus vanilla)'

    return sel_table, gain_table


def selectivity_ordered(
    merged: pd.DataFrame,
    mu_relative: float = 1.0,
    metric: str = 'S_precision',
    drop_undefined: bool = True,
) -> pd.DataFrame:
    """Long-form table sorted by sel(q) for the selectivity-reframe headline.

    Returns one row per (q_source, p_edge) with columns sel_q, vanilla,
    at_mu, gain. Sorted ascending by sel_q so the eye scans monotone-
    decreasing gain top to bottom: the operationalised form of the §6.1
    selectivity claim.

    drop_undefined=True drops rows where sel(q) is NaN (e.g. GES missing
    at dense cells) or inf (e.g. q on S* is zero, PC at moderate p_edge,
    adversarial by construction). Set False to inspect those edge cases.
    """
    soft = merged[~merged['q_source'].astype(str).str.contains('_hard_', na=False)]
    mu_vals = sorted(soft['mu_relative'].unique())
    mu_pick = min(mu_vals, key=lambda v: abs(v - mu_relative))

    sel    = soft.groupby(['q_source', 'p_edge'])['sel_q'].mean()
    at_mu  = soft[soft['mu_relative'] == mu_pick].groupby(['q_source', 'p_edge'])[metric].mean()
    at_van = soft[soft['mu_relative'] == 0.0   ].groupby(['q_source', 'p_edge'])[metric].mean()

    table = pd.DataFrame({
        'sel_q':   sel,
        'vanilla': at_van,
        'at_mu':   at_mu,
        'gain':    at_mu - at_van,
    }).reset_index()

    if drop_undefined:
        table = table[np.isfinite(table['sel_q'])]
    return table.sort_values('sel_q').round(2).reset_index(drop=True)


def aggregate_seeds(
    df: pd.DataFrame,
    metrics: Sequence[str] = ('S_recall', 'S_precision', 'C_inclusion'),
    by: Sequence[str] = ('q_source', 'mu_relative', 'p_edge'),
) -> pd.DataFrame:
    """Group-and-aggregate: mean and SEM for each metric across seeds.

    Returned column naming: '<metric>_mean' and '<metric>_sem' alongside the
    groupby keys. Cells with a single seed produce sem = NaN.
    """
    agg = df.groupby(list(by), dropna=False)[list(metrics)].agg(['mean', 'sem'])
    agg.columns = [f'{metric}_{stat}' for metric, stat in agg.columns]
    return agg.reset_index()
