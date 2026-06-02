"""
Re-score a sweep results.csv from its saved 'edges' column. Use after fixing a metric
bug, adding a new metric, or just to re-verify. Regenerates the SyntheticData per row
(deterministic from param/value/seed) and runs score_recovered against the parsed edges.

Usage:
    python experiments/mixed_cmm/synthetic/recompute_metrics.py results/baselines_20260430_1451/results.csv
    python experiments/mixed_cmm/synthetic/recompute_metrics.py path/to/results.csv --output recomputed.csv
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings('ignore')

import pandas as pd

from experiments.mixed_cmm.synthetic.config import DEFAULTS, GRAPH_METRICS
from src.data.synthetic import SyntheticData
from src.causal_discovery.evaluation import parse_edges, score_recovered

CSV_METRIC_KEYS = list(GRAPH_METRICS) + [
    f'{stat}_{kind}'
    for stat in ('f1', 'precision', 'recall') for kind in ('bin_bin', 'bin_cont')
]


def coerce(param: str, value):
    """Cast the swept value back to the type the SyntheticData kwarg expects."""
    return type(DEFAULTS[param])(value)


def recompute_row(row: pd.Series) -> dict:
    kwargs = {**DEFAULTS, row['param']: coerce(row['param'], row['value'])}
    data = SyntheticData(**kwargs, seed=int(row['seed']))
    recovered = parse_edges(row.get('edges', ''))
    full = score_recovered(recovered, data)
    return {k: full.get(k, float('nan')) for k in CSV_METRIC_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--output', help='output path (default: <input>_recomputed.csv)')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if 'edges' not in df.columns:
        sys.exit(f"error: {args.input_csv} has no 'edges' column. Was it produced by the new sweep scripts?")

    print(f"Recomputing {len(df)} rows from {args.input_csv}", flush=True)
    new_metrics = pd.DataFrame([recompute_row(row) for _, row in df.iterrows()])

    id_cols = [c for c in ('param', 'value', 'seed', 'model', 'method', 'edges') if c in df.columns]
    out = pd.concat([df[id_cols].reset_index(drop=True), new_metrics.reset_index(drop=True)], axis=1)

    out_path = args.output or args.input_csv.replace('.csv', '_recomputed.csv')
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
