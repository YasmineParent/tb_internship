import re

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

# resistance-mechanism groupings used to pool rare variants into burden indicators.
# membership is a biology call (bio team to confirm); kept here so it is explicit and editable.
RESISTANCE_PATHWAYS = {
    'f420_activation': ['ddn', 'fgd1', 'fbia', 'fbib', 'fbic', 'fbid'],  # delamanid/pretomanid activation
    'efflux': ['rv0678', 'mmpl5', 'mmps5', 'pepq', 'rv1979c'],            # bedaquiline/clofazimine efflux
    'linezolid': ['rrl', 'rplc'],
}

_AA3 = r'[A-Z][a-z]{2}'


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
        small = counts[counts < merge_below].index.tolist()
        if small:
            # Merge into the overall smallest lineage value so drop_first=True
            # actually pools them into the reference, regardless of whether the
            # smallest lineage is itself rare.
            target = s.min()
            s = s.replace({v: target for v in small if v != target})
    return pd.get_dummies(s, prefix=prefix, drop_first=drop_first, dtype=int)


def is_synonymous(variant: str) -> bool:
    """true if the variant tail is a protein change with identical reference and alt residues
    (e.g. fbic_Thr560Thr). these are treated as neutral and excluded from the burden."""
    tail = variant.split('_', 1)[1] if '_' in variant else variant
    m = re.match(rf'^({_AA3})(\d+)({_AA3})$', tail)
    return bool(m) and m.group(1) == m.group(3)


def collapse_burden(df: pd.DataFrame, mutation_cols: list[str], pathways: dict | None = None, *,
                    min_prev: float = 0.05, max_prev: float = 0.98, presence_threshold: float = 0.0,
                    drop_synonymous: bool = True) -> tuple[list[str], pd.DataFrame]:
    """Split mutations into singly-modelled common variants and pooled rare-variant burdens.

    A variant is 'present' in a strain when its value exceeds presence_threshold. For binary 0/1
    data leave the default (0). For allele-frequency data (0-100), pass e.g. 5 to call a variant
    present at >=5% (strictly >). Carrier prevalence is the fraction of strains where it is present.

    Common variants (prevalence in [min_prev, max_prev]) are returned to be modelled individually.
    Rare variants (0 < prevalence < min_prev) are pooled per pathway into a binary burden indicator
    (1 if the strain carries any pooled variant in that pathway's genes). Synonymous protein changes
    are dropped as neutral when drop_synonymous is set. Variants whose gene is in no pathway are
    dropped from the burden but, if common, still returned as single columns.

    Returns (single_cols, burden_df). burden_df has one 0/1 column per pathway that has >=1 pooled
    rare variant, indexed like df.
    """
    pathways = pathways or RESISTANCE_PATHWAYS
    n = len(df)
    present = df[mutation_cols].apply(pd.to_numeric, errors='coerce').fillna(0) > presence_threshold
    counts = present.sum()
    # integer count bounds, matching prevalence_filter's rounding so 'common' agrees with it
    min_count, max_count = int(round(min_prev * n)), int(round(max_prev * n))
    kept = [c for c in mutation_cols if not (drop_synonymous and is_synonymous(c))]
    single_cols = [c for c in kept if min_count <= counts[c] <= max_count]
    rare = [c for c in kept if 0 < counts[c] < min_count]

    gene_to_pathway = {g: name for name, genes in pathways.items() for g in genes}
    burden = {}
    for name in pathways:
        cols = [c for c in rare if gene_to_pathway.get(c.split('_', 1)[0]) == name]
        if cols:
            burden[name] = present[cols].any(axis=1).astype(int)
    return single_cols, pd.DataFrame(burden, index=df.index)

