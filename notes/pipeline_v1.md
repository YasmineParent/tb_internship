# Pipeline

## Setup

- **Data.** $n = 164$ TB patients, $p = 211$ variables (mostly
  binary mutations, some continuous MIC values, categorical
  lineage/resistance type).
- **Target variable.** Binary resistance interpretation vs
  continuous MIC vs hybrid (train continuous, evaluate
  binary)? Trade-offs: MIC matches CISPA's Gaussian
  assumption cleanly but has noisier labels and moves us
  outside the standard classification-scoring literature.
- **Regime.** $p > n$, sparse binary features, class imbalance.

## Stage 1: Causal discovery via CISPA

Run CISPA on the data with the hard ordering constraint:
mutations upstream of phenotypic/clinical variables.

Outputs:
- Causal graph $G = (V, E)$, a DAG over all variables
- Parent sets $\mathrm{Pa}_j$ for each variable $X_j$
- Component-specific linear coefficients $\beta_{jk} \in
  \mathbb{R}^{|\mathrm{Pa}_j|}$ for each $X_j$ and component $k$
- Mixture weights $\gamma_k$ over $K$ latent components
- Posterior assignments $r_{ik} = P(Z_i = k \mid x_i, \theta)$

CISPA's structural equation (linear Gaussian mechanism):
$$X_j = \beta_{jk}^\top \mathrm{Pa}_j + \beta_{jk}^{(0)} + N_j
\quad \text{when } Z_i = k$$

**Stability.** Run on $B \approx 50$ bootstrap resamples. Keep
edges that appear in $\geq 50\%$ of runs → stable graph $G^*$.

MIC, agar: mediators not features

## Stage 2: Causal feature engineering

### 2a. Feature filtering

Keep mutation $m$ in the candidate set iff there exists a
directed path $m \to \ldots \to Y$ in $G^*$, which may go through MIC. Optional softer
version: keep $m$ if it has a path OR a strong marginal
association with $Y$ (fallback for false negatives in $G^*$).

### 2b. Importance score

For each mutation $m$ with a directed path to $Y$ in $G^*$, the
importance score aggregates the total causal effect of $m$ on $Y$
across mixture components.

**Direct parents** ($m \in \mathrm{Pa}_Y$):
$$s_m = \sum_k \gamma_k \cdot \left| \beta_{Y, k}[m] \right|$$

**Mediated through MIC** ($m \in \mathrm{Pa}_{\text{MIC}}$,
$\text{MIC} \in \mathrm{Pa}_Y$):
$$s_m = \sum_k \gamma_k \cdot \left| \beta_{\text{MIC}, k}[m]
\cdot \beta_{Y, k}[\text{MIC}] \right|$$

**General case** (directed path $m \to X_1 \to \ldots \to X_\ell
\to Y$): product of coefficients along the path, then aggregated
over $k$ by mixture weight. If multiple paths exist, sum the
per-path effects before taking the absolute value (or after,
depending on whether you want signed or unsigned importance —
TBD).

This measures how much $m$ influences $Y$ in total, through
whatever causal route. It is not the norm of $m$'s
parent-coefficients, that's the sensitivity of $m$ to its own
parents.

### 2c. Interaction features from co-parentship

For each pair of mutations $(m, m')$ such that there exists
a common child $c$ with $\{m, m'\} \subseteq \mathrm{Pa}_c$ in
$G^*$, create a new binary feature:
$$x_{mm'} = x_m \cdot x_{m'}$$

Optional extensions:
- Logical OR features: $x_m \lor x_{m'}$ for redundant pathways
- Threshold features: $\mathbb{1}\left[\sum_{m \in S} x_m
  \geq k\right]$ for gene groups $S$ with shared causal effect

### 2d. Final feature set

$\mathcal{F} = \{\text{individual mutations with path to } Y\}
\cup \{\text{interaction features from co-parentship}\}
\cup \{\text{threshold features, if applicable}\}$

## Stage 3 — Scoring system

Chosen backbone (TBD with Sokolovska): FasterRisk or
Sokolovska 2018.

**Objective (FasterRisk-style, schematic):**
$$\min_{\lambda \in \mathbb{Z}^{|\mathcal{F}|}} \;
\mathcal{L}_{\text{logistic}}(\lambda; \mathcal{F}, Y)
- \mu \sum_{m \in \mathcal{F}} s_m \cdot \mathbb{1}[\lambda_m
  \neq 0]
\quad \text{s.t. } \|\lambda\|_0 \leq K, \; |\lambda_m| \leq
\lambda_{\max}$$

Where:
- $\mathcal{L}_{\text{logistic}}$ is the logistic loss
- $K$ is the sparsity budget (typical: 5-10 features)
- $\lambda_{\max}$ is the integer coefficient bound (typical: 5)
- $\mu \geq 0$ controls how strongly the causal importance
  score biases feature selection

**Causal constraints (optional, hard).**
Admissible coefficient vectors $\lambda$ must satisfy:
- No cause + direct-effect pairs: if $m \to m' \in E(G^*)$,
  then $\lambda_m \cdot \lambda_{m'} = 0$
- d-separation: if $m \perp_{G^*} m' \mid C$ and
  $\lambda_C \neq 0$, exclude one of $m, m'$

Apply these only for bootstrap-stable edges.

**Output.** Rashomon pool $\Lambda^* = \{\lambda^{(1)}, \ldots,
\lambda^{(R)}\}$ of near-optimal sparse integer solutions.

## Stage 4 — Stability & validation

- Run stages 1-3 on bootstrap resamples
- Feature stability: keep features selected in $\geq 60\%$ of runs
- Coefficient stability: track distribution of $\lambda_m$ across runs
- Clinical validation: present $\Lambda^*$ to CIMI for biological
  plausibility check

## Stage 5 — Comparative evaluation

Three controlled ablations to isolate each contribution:

| Ablation | Tests the value of |
|---|---|
| All features, no causal preselection | Stage 2a (filtering) |
| Causal features, univariate selection | Stage 2b (importance) |
| Causal features, no interaction features | Stage 2c (interactions) |

Metrics: AUC-ROC, Brier score (calibration), sparsity,
cross-validated accuracy, SEV.

## Key open questions

1. **Joint vs per-drug CISPA.** Run on all 5 resistance outcomes
   jointly (captures shared mechanisms, worsens $p/n$) or
   per-drug (loses shared structure, better dimensionality).
2. **Scoring backbone.** Sokolovska 2018 (lab code?) vs
   FasterRisk (Rashomon pool, public implementation).
3. **Integrated vs decoupled.** Is the causal penalty term
   $\mu \sum_m s_m \mathbb{1}[\lambda_m \neq 0]$ the target
   formulation, or does the pipeline stay fully decoupled
   with causal knowledge only entering via $\mathcal{F}$?
4. **$Z_i$ vs lineage.** Do CISPA's latent components map onto
   observable phylogenetic lineage? 
5. **Missing values (delamanid).** Imputation, exclusion, or
   native handling à la McTavish 2024?
6. Gaussian mechanism for binary Y in CISPA — acceptable scale 
   approximation for ranking, or pivot to the 'CISPA on continuous 
   ancestors only' framing? 

## Methodological contributions

1. **First causally-informed risk scoring system for TB drug
   resistance.** Causal structure enters at feature selection
   (filtering), feature generation (interactions), and
   objective weighting (importance prior).
2. **Co-parentship as an interaction-generation heuristic.**
   Principled way to decide which pairwise interactions to
   include in interpretable scoring systems: biologically
   justified, computationally tractable.
3. **Mixture-weighted importance scores.** Explicit use of
   subpopulation structure $(\gamma_k, \beta_{jk})$ to rank
   features by their causal effect on $Y$, aggregated over
   latent components.