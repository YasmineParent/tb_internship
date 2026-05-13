import pandas as pd


META_COLS = ['isolate', 'glims', 'type', 'lineage']
MIC_COLS = ['dlm_mic', 'bdq_mic', 'cfz_mic', 'lnz_mic', 'ptm_mic']
INTERP_COLS = ['interp_dlm016', 'interp_dlm06', 'interp_bdq12', 'interp_bdq25', 'interp_lnz1']
PROP_MUTANT_COLS = [
    'prop_mutants_dlm016', 'prop_mutants_dlm06',
    'prop_mutants_bdq12', 'prop_mutants_bdq25',
    'prop_mutants_lnz1',
]
NON_MUTATION_COLS = set(META_COLS + MIC_COLS + INTERP_COLS + PROP_MUTANT_COLS)


def load_tb_data(path: str) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Load TB dataset. Returns (df, mutation_cols, mic_cols, interp_cols, prop_mutant_cols)."""
    df = pd.read_csv(path)
    mutation_cols = [c for c in df.columns if c not in NON_MUTATION_COLS]
    mic_cols = [c for c in df.columns if c in set(MIC_COLS)]
    interp_cols = [c for c in df.columns if c in set(INTERP_COLS)]
    prop_mutant_cols = [c for c in df.columns if c in set(PROP_MUTANT_COLS)]
    return df, mutation_cols, mic_cols, interp_cols, prop_mutant_cols


def prevalence_filter(df: pd.DataFrame, mutation_cols: list[str], min_prev: float = 0.05, max_prev: float = 0.98) -> list[str]:
    """Return mutation columns with prevalence between min_prev and max_prev."""
    n = len(df)
    min_count = int(round(min_prev * n))
    max_count = int(round(max_prev * n))
    return [c for c in mutation_cols if min_count <= df[c].sum() <= max_count]


def type_beyond_MDR(df: pd.DataFrame) -> pd.Series:
    """Binary indicator: 1 if resistance type is preXDR or XDR (type > 0), else 0.
    Collapses the 3-level type code into one covariate to keep the rare preXDR (n=23)
    and XDR (n=7) cells from fragmenting the mixture model."""
    return (df['type'] > 0).astype(int)


def lineage_dummies(df: pd.DataFrame, drop_first: bool = True, prefix: str = 'lineage',
                    merge_below: int | None = None) -> pd.DataFrame:
    """One-hot encode the 'lineage' column. drop_first=True drops the lowest-numbered
    lineage as reference, avoiding collinearity with the regression intercept R adds.

    merge_below: if set, lineages with count < merge_below are recoded to the smallest-count
    lineage value before encoding. Combined with drop_first=True this effectively pools them
    into the reference category. Use to absorb minority lineages too rare to model on their own."""
    s = df['lineage'].copy()
    if merge_below is not None:
        counts = s.value_counts()
        small = sorted(counts[counts < merge_below].index.tolist())
        if small:
            target = small[0]
            s = s.replace({v: target for v in small[1:]})
    return pd.get_dummies(s, prefix=prefix, drop_first=drop_first, dtype=int)

