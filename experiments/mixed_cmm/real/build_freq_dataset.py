"""Build a pipeline-compatible dataset from the allele-frequency matrix.

matrix_muts_freq.csv has mutations as allele frequencies (0-100) over an expanded gene panel but
no continuous MIC columns. This merges the continuous MICs from the existing clean dataset (same
164 isolates, join on isolate) and binarizes the mutations at a frequency threshold, yielding the
same schema as tb_pheno_geno_clean.csv (meta + continuous MICs + 0/1 mutations) so load_tb and the
existing runners work unchanged.

Usage:
    python experiments/mixed_cmm/real/build_freq_dataset.py            # threshold 5 (>=5% present)
    python experiments/mixed_cmm/real/build_freq_dataset.py --threshold 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from src.data.load_tb import MIC_COLS

ROOT = Path(__file__).resolve().parents[3]
FREQ = ROOT / 'data' / 'real' / 'raw' / 'matrix_muts_freq.csv'
CLEAN = ROOT / 'data' / 'real' / 'processed' / 'tb_pheno_geno_clean.csv'
META = ['isolate', 'glims', 'type', 'lineage']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=5.0, help='allele %% at/above which a variant is present')
    ap.add_argument('--out', type=str, default=str(ROOT / 'data' / 'real' / 'processed' / 'tb_freq_clean.csv'))
    args = ap.parse_args()

    freq = pd.read_csv(FREQ)
    # exclude meta, the proportion-method/interp phenotype cols, and the freq file's own MIC
    # columns (mic_<drug>): those are outcomes, not mutations, and would leak into the target.
    mut_cols = [c for c in freq.columns
                if c not in META and not c.startswith(('pm_', 'interp_', 'mic_'))]
    clean = pd.read_csv(CLEAN)
    mics = [c for c in MIC_COLS if c in clean.columns]

    # binarize mutations at the threshold; the clean MICs are complete (the freq file's mic_* are
    # sparse), so take the continuous MICs from clean, joined on isolate.
    binmut = (freq[mut_cols].apply(pd.to_numeric, errors='coerce').fillna(0) >= args.threshold).astype(int)
    out = pd.concat([freq[['isolate', 'glims', 'type', 'lineage']], binmut], axis=1)
    out = out.merge(clean[['isolate'] + mics], on='isolate', how='inner')
    out = out[['isolate', 'glims', 'type', 'lineage'] + mics + mut_cols]

    out.to_csv(args.out, index=False)

    # diagnostics: gain over the old 14-gene binary panel
    n = len(out)
    bar = int(round(0.05 * n))
    genes = sorted({c.split('_', 1)[0] for c in mut_cols})
    carr = out[mut_cols].sum()
    print(f"wrote {args.out}", flush=True)
    print(f"  {n} isolates, {len(mut_cols)} variants over {len(genes)} genes, threshold >= {args.threshold}%")
    print(f"  MIC merge non-null: " + ", ".join(f"{m}={out[m].notna().sum()}" for m in mics))
    print(f"  variants clearing the 5% bar (>= {bar} carriers): {(carr >= bar).sum()}")
    print(f"  genes: {genes}")
    print(f"\n  carriers clearing 5% bar, by threshold choice (tradeoff):")
    for t in [1, 5, 10, 25]:
        b = (freq[mut_cols].apply(pd.to_numeric, errors='coerce').fillna(0) >= t).astype(int).sum()
        print(f"    >= {t:>2}% : {(b >= bar).sum()} modelable variants")


if __name__ == '__main__':
    main()
