# TB drug susceptibility dataset

## Overview

Phenotypic and genotypic drug susceptibility measurements for 164 clinical tuberculosis
strains, constructed to identify genetic markers of resistance to new antituberculosis drugs.

## Provenance

Dataset constructed and managed by A. Aubry's team, CIMI, Sorbonne Université. Used in
collaboration with the Computational Medicine team, CQSB, Sorbonne Université.

---

## Methods

**Agar proportion method:** proportion of resistant mutants determined by inoculation in
antibiotic-containing agar medium. A strain is resistant if the proportion of resistant
mutants is ≥ 1%.

**Minimum inhibitory concentration (MIC):** lowest antibiotic concentration inhibiting
bacterial growth. A strain is resistant if MIC > CC (WHO-defined critical concentration).

**Whole genome sequencing (WGS):** mutations in resistance-associated genes encoded as
binary presence/absence features.

---

## Study aim

Genetic markers of resistance to bedaquiline, clofazimine, delamanid, linezolid, and
pretomanid are not yet well characterised. This dataset associates mutations in
resistance-associated genes with phenotypic susceptibility testing results.

---

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| `bdq` | bedaquiline |
| `dlm` | delamanid |
| `lnz` | linezolid |
| `cfz` | clofazimine |
| `ptm` | pretomanid |
| `mic` | minimum inhibitory concentration |
| `prop_mutants` | percentage of resistant mutants in strain |

---

## Columns

### Strain metadata

| Column | Description |
|---|---|
| `isolate` | strain identifier (prefix `ss` has no significance) |
| `glims` | patient identifier |
| `type` | resistance profile: `0` = MDR, `1` = preXDR, `2` = XDR |
| `lineage` | phylogenetic lineage (integer) |

Resistance type definitions:
- `0` MDR: resistant to isoniazid and rifampicin
- `1` preXDR: resistant to isoniazid, rifampicin, and fluoroquinolones
- `2` XDR: resistant to isoniazid, rifampicin, fluoroquinolones, and bedaquiline and/or linezolid

### Agar proportion method

Proportion of resistant mutants (continuous, 0–100%):

| Column | Drug | Concentration |
|---|---|---|
| `prop_mutants_bdq12` | bedaquiline | 0.12 mg/L |
| `prop_mutants_bdq25` | bedaquiline | 0.25 mg/L |
| `prop_mutants_dlm016` | delamanid | 0.016 mg/L |
| `prop_mutants_dlm06` | delamanid | 0.06 mg/L |
| `prop_mutants_lnz1` | linezolid | 1 mg/L |

Resistance interpretation (ordinal, outcome variable):

| Column | Drug | Encoding |
|---|---|---|
| `interp_bdq12` | bedaquiline 0.12 mg/L | `0` = susceptible, `1` = intermediate, `2` = resistant |
| `interp_bdq25` | bedaquiline 0.25 mg/L | same |
| `interp_dlm016` | delamanid 0.016 mg/L | same |
| `interp_dlm06` | delamanid 0.06 mg/L | same |
| `interp_lnz1` | linezolid 1 mg/L | same |

### Minimum inhibitory concentration

Continuous values in mg/L. WHO-defined critical concentrations:

| Column | Drug | Critical concentration |
|---|---|---|
| `bdq_mic` | bedaquiline | 1 mg/L |
| `cfz_mic` | clofazimine | 1 mg/L |
| `lnz_mic` | linezolid | 1 mg/L |
| `dlm_mic` | delamanid | 0.06 mg/L |
| `ptm_mic` | pretomanid | 1 mg/L |

### Mutation features

193 binary columns encoding mutation presence (`1`) or absence (`0`) in
resistance-associated genes: `rv0678_*`, `mmpl5_*`, `atpe_*`, `pepq_*`, `rv1979c_*`,
`rplc_*`, `rrl_*`, `fbia_*`, `fbib_*`, `fbic_*`, `fbid_*`, `fgd1_*`, `ddn_*`.

Column names follow the format `gene_mutation`. Note that 170 out of 193 mutations are
present in fewer than 5% of strains, and two (`rv1979c_A-129G`, `mmpl5_Ile948Val`) are
near-universal (>98% prevalence) and uninformative.