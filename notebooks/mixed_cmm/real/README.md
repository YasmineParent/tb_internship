# Reading these notebooks

The notebooks all attack one question: which mutations are actually linked to a
drug's MIC (minimum inhibitory concentration), once we account for lineage,
resistance type, and the other mutations. The whole point of the method is to
separate genuine mutation-to-MIC links from apparent ones that only run through
the lineage a strain belongs to.

## The central idea

Lineage is read off the genome, so in these graphs it sits **downstream** of the
mutations: a mutation can determine or mark a lineage, never the other way round.
Lineage is in turn associated with the MIC. So a mutation can end up linked to an
MIC indirectly, through the lineage it travels with, without acting on the MIC
itself.

```
   mut ──> lineage ──> MIC     indirect: the mutation's link to the MIC
    │                          can run entirely through lineage
    └─────── ? ──────> MIC     direct: does a link survive that does
                               not pass through lineage?
```

A causal-discovery graph tests for that direct arrow: it keeps `mut -> MIC` only
if the link holds up after accounting for lineage (and the other mutations). If
the apparent association was only the indirect route, the edge collapses. That is
the difference from a plain correlation, which cannot tell the two apart.

## What the graphs are and how to read them

- **Nodes and arrows.** One graph is built over all the variables at once
  (mutations, lineage, resistance type, drug MICs). Each node is a variable; each
  arrow is a direct link kept only if it holds up after accounting for everything
  else in the graph.

- **The arrows are directed.** `X -> MIC` means X predicts that MIC directly, in
  that direction, after adjusting for the rest. (An undirected link would mean two
  variables are related but the direction is unknown; here the outcome direction
  is fixed by the setup.)

- **The number on an arrow is a reliability score** (see stability selection
  below): higher is more trustworthy, and 0.5 or above is treated as solid.

- **Node colours.** salmon/red = drug MIC (the outcome), blue = a single mutation,
  orange = a pooled variant burden, grey = lineage, khaki = resistance type.

- **Forbidden arrows.** Some arrows are blocked on purpose so the graph answers
  the question we want (see modelling choices). For example, MIC-to-MIC arrows are
  forbidden in the cross-resistance notebooks, so two drugs can only end up linked
  through a shared feature that points into both of them.

## Modelling choices

- **Outcome is the continuous MIC**, not the yes/no resistance call. For
  delamanid the two disagree a lot (32 strains called resistant on agar but
  susceptible by MIC), so the continuous MIC is the more reliable target. Higher
  MIC means more resistant.

- **Why a mixture model (CMM) and not a standard method.** The strains are not one
  homogeneous population: a mutation-to-MIC relationship can be strong in some
  strains and absent in others. A standard analysis averages those behaviours
  together and can miss or even flip the signal. CMM instead finds latent
  subgroups where the relationship is homogeneous, then learns a directed graph
  that accounts for that heterogeneity. Standard methods also assume smooth
  Gaussian relationships and continuous variables, which this mixed binary and
  continuous data does not satisfy.

- **Adaptation for binary mutations.** The original CMM is Gaussian. The binary
  (0/1) nodes use a logistic likelihood instead. Without this adaptation, on
  synthetic data with known answers about 97% of the mutation-to-mutation edges
  are missed.

- **The number of subgroups is chosen automatically** (by a statistical criterion,
  BIC), not by hand. It lands on 6 for the MIC and 1 for the mutations, lineage,
  and type. The plain reason for 6: the MIC is measured on a doubling-dilution
  scale, so it only takes a handful of distinct step values here, and the mixture
  places its components on those steps. This describes the shape of the MIC scale;
  it is not evidence of 6 different subgroups of strains. Binary mutations stay at
  1 component (nothing to split).

- **Stability selection.** The data is small and sparse, so a single fit is not
  reliable. Each model is refit on 100 random 80% subsamples, and only links that
  come back often are kept (threshold 0.5, the Meinshausen and Buhlmann
  convention). The number shown on each arrow is that survival rate.

  ```
  164 strains  ->  100 subsamples (80%)  ->  100 CMM graphs  ->  per-edge frequency
  ```

- **Prevalence eligibility.** A mutation present in only a few strains can be
  entirely absent from a given subsample, where no relationship can be learned for
  it. So a mutation is dropped from any subsample where it has fewer than 5
  positives, and its frequency is computed only over the subsamples where it was
  eligible.

- **Pooled burdens.** Most variants show up in very few strains (< 5% each), too
  rare to fit on their own. So within each predefined gene group, the rare
  non-synonymous variants are collapsed into a single binary column (the burden):
  it is 1 for a strain that carries any such variant in that group's genes, and 0
  otherwise. Common variants are kept as their own nodes; synonymous variants are
  dropped. This swaps "which exact variant" for "is this group hit at all", which
  is the level of detail the data can actually support when each variant alone is
  too sparse to model.
  - `burden_f420_activation` pools rare variants across `ddn`, `fgd1`, `fbiA`,
    `fbiB`, `fbiC`, `fbiD`.
  - `burden_efflux` pools rare variants across `rv0678`, `mmpL5`, `mmpS5`,
    `pepQ`, `rv1979c`.

- **Progressive adjustment.** The analysis is redone under five conditions, adding
  lineage and then resistance type, and blocking certain direct arrows into the
  MIC. A real link should hold across all of them.

  | # | Condition | Variables | Constraint |
  | --- | --- | --- | --- |
  | 1 | baseline | mutations only | |
  | 2 | +lineage | + lineage | |
  | 3 | +lineage +type | + resistance type | |
  | 4 | block lineage -> MIC | + lineage | lineage cannot point into MIC |
  | 5 | block type -> MIC | + lineage + type | type cannot point into MIC |

  Two orientation choices sit behind this. **Lineage is placed below the
  mutations** (it is fixed by the genome, so it is never allowed to point into a
  mutation; it is a background marker). **Resistance type is placed below the
  MICs** (the type label is defined from resistance to other drugs, so it is
  adjusted out rather than allowed to point into an MIC).

  **Why block an arrow?** Blocking the direct lineage-to-MIC path forces the model
  to explain the MIC without it. If a mutation's frequency then goes up in that
  condition, it is picking up the lineage's signal rather than carrying an
  independent one: an artifact, not a genuine link.

## Data realities behind the choices

- **Rare mutations.** Only 14 mutations exceed 5% prevalence and are usable.
- **Class imbalance.** Most strains are susceptible; delamanid has the most
  resistant cases, which makes it the most workable outcome.
- **Lineage and resistance type are linked** (lineage 4 is almost entirely MDR),
  so an apparent mutation-to-MIC link can run through either one, and both are
  adjusted for.

## What each notebook does

| Notebook | What it does |
| --- | --- |
| [data_exploration.ipynb](data_exploration.ipynb) | Preliminary look at the data: dataset shape, missingness (and a check on whether it is informative), class balance per drug, MIC distributions, agar vs MIC agreement, mutation sparsity, and mutation co-occurrence clusters. Sets up why the later modelling choices are made. |
| [delamanid_stable_graphs.ipynb](delamanid_stable_graphs.ipynb) | The stable causal graph for the delamanid MIC. Shows the arrows into `dlm_mic` when adjusting for lineage, the same model with full background structure, and the version with resistance type adjusted out. |
| [delamanid_ablation_comparison.ipynb](delamanid_ablation_comparison.ipynb) | Tracks the numbers rather than the pictures: how each mutation's reliability score into `dlm_mic` moves as lineage and then resistance type are added. Also compares the corrected lineage orientation against the old one. |
| [cross_resistance.ipynb](cross_resistance.ipynb) | Cross-resistance across several drugs on the original dataset. Puts all the drug MICs in one graph with MIC-to-MIC arrows forbidden, so any feature pointing into more than one MIC is a shared driver. Uses linezolid (no resistant strains) as a negative control. |
| [cross_resistance_freq.ipynb](cross_resistance_freq.ipynb) | The same cross-resistance design on the allele-frequency dataset, over a wider gene panel. Shows how the negative control catches a feature that only tracks population structure rather than resistance. |

This is an exploratory analysis on these strains, not a clinical validation. The
goal is to flag robust leads and to avoid over-reading a raw mutation-to-MIC
association without adjusting for lineage.
