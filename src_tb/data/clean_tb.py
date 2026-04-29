from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def clean_mic(val: object) -> float:
    """Clean MIC values by handling sup prefixes and converting to float."""
    if pd.isna(val):
        return float('nan')
    val = str(val).strip().replace(',', '.')
    if val.startswith('sup'):
        num = val.replace('sup', '').strip()
        if not num:
            return float('nan')
        return float(num)
    return float(val)


def load_and_clean(raw_path: str) -> pd.DataFrame:
    """Load the raw TB data and perform cleaning steps."""
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    outcome_cols = ['interp_bdq12', 'interp_bdq25', 'interp_dlm016', 'interp_dlm06', 'interp_lnz1']
    for col in outcome_cols:
        df[col] = df[col].astype('Int64')
    for col in ['bdq_mic', 'cfz_mic', 'lnz_mic', 'dlm_mic', 'ptm_mic']:
        df[col] = df[col].apply(clean_mic)
    prop_cols = ['prop_mutants_bdq12', 'prop_mutants_bdq25', 'prop_mutants_dlm016', 'prop_mutants_dlm06', 'prop_mutants_lnz1']
    for col in prop_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(',', '.'), errors='coerce')
    return df


if __name__ == '__main__':
    df = load_and_clean(str(REPO_ROOT / 'data/real/raw/tb_pheno_geno.csv'))
    df.to_csv(REPO_ROOT / 'data/real/processed/tb_pheno_geno_clean.csv', index=False)