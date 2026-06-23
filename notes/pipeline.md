# Causally-informed sparse risk scoring

A method for interpretable integer-coefficient risk scoring with a soft, threshold-free causal prior on feature selection. The document is organized as a paper: abstract, introduction, method, theory, experiments, related work, contributions, and limitations, with implementation and the full support-stability proofs in appendices.

## Abstract

Sparse integer-coefficient risk scores are valued in high-stakes settings because they can be read and checked by hand, but standard fitting selects features by predictive association alone and can lean on variables that correlate with the outcome only through confounding. We introduce a soft, one-parameter prior that biases scorecard feature selection toward features supported by conditional-independence evidence from causal discovery; the prior is the MAP of a Bernoulli inclusion model with a sigmoid link, reduces to a single linear bonus on the support indicator, and preserves the decomposability and integer-rounding guarantees of FasterRisk. On synthetic data with known ground truth, sourcing the evidence vector from causal discovery recovers the true sparse support; the gain tracks how well discovery concentrates evidence on causes rather than correlates, and degrades where discovery is sample-starved on dense graphs. On the same synthetic setup, scorecards built from causally-sourced evidence remain accurate under a controlled environment shift that rescales non-causal correlate associations while leaving the causal mechanism invariant, whereas scorecards built from predictively-sourced evidence lose transport as confounders are pulled out of distribution; this transport result rests entirely on synthetic data. On public real-data benchmarks the method matches vanilla FasterRisk's accuracy, produces a steadier feature selection on four of five benchmarks, and scores above hard causal feature selection in accuracy and stability.

## 1. Introduction

Standard sparse-scorecard fitting selects features by predictive association alone and can lean on variables that correlate with the outcome only through confounding. We add a one-parameter prior that biases selection toward features supported by conditional-independence evidence from causal discovery and preserves the guarantees of the FasterRisk solver it builds on.

Contributions:

1. A soft causal-prior penalty for sparse integer classification: the MAP of a Bernoulli inclusion prior, a single linear-in-$q$ bonus on the support indicator, threshold-free, with vanilla FasterRisk and hard pre-selection as its two limits (Section 2).
2. A decomposability-preserving integration into FasterRisk that inherits its integer-rounding bound (Section 2.5).
3. Causal discovery as the source of evidence that recovers the true support and transports across environments where predictive sourcing does not, because discovery concentrates evidence on causes rather than correlates (Section 4).
4. An exact support-stability analysis unifying the stability gain and the adversarial fragility as one $1/\mu$ exchange, with a closed-form threshold below which the prior cannot displace the loss-optimal support, a probabilistic stability rate that improves with the prior's margin, and a selection-consistency result (Section 3).

**Notation.** Following Liu et al. (2022, FasterRisk): dataset $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^n$ with $\mathbf{x}_i\in\mathbb{R}^p$, $y_i\in\{-1,+1\}$; integer coefficients $\mathbf{w}\in\mathbb{Z}^p$ with intercept $w_0$; sparsity $k$; coefficient bound $C$; multiplier $m>0$; integer-rounded solution $\mathbf{w}^+$. Feature index $j\in[1,p]$.

## 2. Method

### 2.1 Problem setup

Inputs to the classifier stage: features $X\in\mathbb{R}^{n\times p}$ (mixed binary and continuous), a binary target $y\in\{-1,+1\}^n$ (required by FasterRisk's logistic loss), and a causal-evidence vector $q\in[0,1]^p$ where $q_j$ encodes evidence that feature $j$ is a cause of the target rather than a confounded correlate. The vector $q$ comes from a separate procedure (Section 2.6) whose target need not be binary; any binarization happens only at the classifier stage, a constraint of FasterRisk's design rather than of the problem. Output: an integer-coefficient sparse logistic scorecard whose support is biased toward high-$q_j$ features, with strength controlled by a single continuous parameter $\mu\ge0$.

### 2.2 Modified objective

Standard FasterRisk minimizes the logistic loss under a sparsity and integer-box constraint:

$$
\min_{\mathbf{w},w_0}\; L(\mathbf{w},w_0,\mathcal{D})\quad\text{s.t.}\quad \|\mathbf{w}\|_0\le k,\;\mathbf{w}\in\mathbb{Z}^p,\;w_j\in[-C,C],
$$

with $L=\sum_i \log(1+\exp(-y_i(\mathbf{x}_i^\top\mathbf{w}+w_0)))$. The causal-prior bonus adds one term:

$$
\min_{\mathbf{w},w_0}\; L(\mathbf{w},w_0,\mathcal{D})\;-\;\mu\sum_{j=1}^p q_j\,\mathbb{1}[w_j\neq0],
$$

subject to the same constraints. Parameters: $\mu\ge0$ (prior strength), $k$ (sparsity), $C$ (bound).

### 2.3 MAP derivation

Place a discrete spike-and-slab prior  on each coefficient, with support indicators $z_j = \mathbb{1}[w_j \neq 0] \sim \text{Bernoulli}(\pi_j)$ independent across $j$ and a uniform slab on $w_j \mid z_j = 1$ over $\{-C, \ldots, -1, 1, \ldots, C\}$:

$$
p(w_j) = (1 - \pi_j)^{1 - z_j} \left(\frac{\pi_j}{2C}\right)^{z_j}, \qquad p(\mathbf{w}) = \prod_{j=1}^p p(w_j).
$$

Taking logs and collecting the $z_j$-dependent terms,

$$
\log p(\mathbf{w}) = \sum_{j=1}^p z_j \big(\operatorname{logit}(\pi_j) - \log(2C)\big) + \text{const}.
$$

Set $\pi_j = \sigma(\mu q_j + \log(2C))$, so the $\log(2C)$ offset cancels the slab cardinality and $\operatorname{logit}(\pi_j) - \log(2C) = \mu q_j$:

$$
\log p(\mathbf{w}) = \mu \sum_{j=1}^p q_j \, \mathbb{1}[w_j \neq 0] + \text{const}.
$$

MAP estimation then minimizes

$$
L(\mathbf{w}, w_0, \mathcal{D}) - \log p(\mathbf{w}) = L(\mathbf{w}, w_0, \mathcal{D}) - \mu \sum_{j=1}^p q_j \, \mathbb{1}[w_j \neq 0] + \text{const},
$$

which is the modified objective. The sigmoid is forced: $\operatorname{logit}$ is the Bernoulli natural parameter, so making the bonus linear in $q_j$ requires its inverse as the link. At $q_j = 0$ the bonus vanishes, so a feature with no causal evidence gets no steer and the objective reduces to vanilla. With $q_j \in [0,1]$ and $\mu \geq 0$ the prior only promotes inclusion; it cannot demote a feature. Within the fixed sparsity budget $K$, however, promoting causes occupies slots that would otherwise go to confounded correlates. The purely causal supports reported in Section 4.3 arise through this budget-competition mechanism: elevating causes under the sparsity constraint forces correlates out of the top-$K$. Section 4.3 does not contradict the promote-only property; it shows what budget competition achieves when the prior concentrates evidence on the true causes. Independence across $j$ gives the per-feature decomposability used in §2.4 but ignores collinearity, which is handled only by the hard sparsity cap (§6).

### 2.4 Limits

$\mu = 0$: the bonus vanishes ($\mu \sum_j q_j \mathbb{1}[w_j \neq 0] = 0$), the modified objective reduces to $L$, and vanilla FasterRisk is recovered.

### 2.5 Structural properties

**Linear separability.** The bonus decomposes as $\sum_j q_j\mathbb{1}[w_j\neq0]$, so per-feature marginal cost is computable without recomputing global quantities. FasterRisk's SparseBeamLR expansion and CollectSparseDiversePool swap remain per-feature decomposable, so the modification is a one-line reweighting of the beam search with no asymptotic-complexity change, and is bit-identical to vanilla at $\mu=0$ or $q=\mathbf{0}$. The same linear-in-$z_j$ structure admits a one-line addition to RiskSLIM's MIP formulation (a linear coefficient $-\mu q_j$ on the inclusion indicator; not evaluated here).

**Magnitude invariance under rounding** (conditional on support preservation). The bonus depends only on $\mathbb{1}[w_j\neq0]$, not on $|w_j|$, so it is identically zero across integer rounding whenever the support is preserved. Under that condition FasterRisk's AuxiliaryLossRounding bound (their Theorem 3.1) transfers unchanged to the modified objective. Support preservation can fail at large $\mu$ when the bonus forces low-magnitude features into the support. The rounding guarantee therefore holds most cleanly in the same weak-$\mu$ regime where the prior has modest influence, and its reliability narrows as $\mu$ grows toward values where the prior meaningfully reshapes the support.

**Scale.** $\mu$ has no data-invariant scale, since $L$ grows with $n$ while $q$ is unit-free. We report $\mu$ relative to $\mu_{\text{scale}}=\mathrm{median}_j\,|\nabla_j L|$ at $\mathbf{w}=\mathbf{0}$ (equivalently $\mathrm{median}(0.5|X^\top y|)$ on binarized data), computed once per dataset, so a single relative $\mu$ is comparable across datasets.

### 2.6 The causal-evidence interface

**Requirement.** $q$ should come from a procedure that performs conditional-independence reasoning to remove confounding-driven associations. Predictive-only signals (bootstrap stability of LASSO or tree ensembles, marginal mutual information) are not used as $q$: they derive from the same logistic objective the classifier already optimizes, so treating them as a prior duplicates information rather than adding it.

The method is source-agnostic: the MAP construction holds regardless of which procedure produces $q$. Admissible sources include global discovery (PC, GES), Markov-blanket-local learners (IAMB and variants, HITON-MB), Invariant Causal Prediction in multi-environment settings, curated knowledge graphs with directional edges, and expert elicitation conditioned on causal status. All are used through subsample stability selection ($B$ runs, $q_j=\mathrm{freq}(j\to t)$). Because the deployed prior is the Markov blanket of the target (Section 4.1), MB-local learners target exactly what the method consumes; global discovery learns the whole graph and keeps only the target's neighbourhood.

**Propagation to binarized columns.** When the classifier stage binarizes a continuous feature into several indicator columns, the prior is defined at the original-feature level and each binarized column inherits its parent's value, $q^{\mathrm{bin}}_c=q^{\mathrm{orig}}_{\mathrm{parent}(c)}$. The causal structure lives at the original-feature level; binarized columns are downstream encoding choices and inherit the causal status of their parent. This step is a deliberate modelling choice, presented as such.

## 3. Theory: support stability

Raising $\mu$ has two effects on the MAP support that pull in opposite directions. A larger $\mu$ makes the selected support more stable against fold-to-fold data variation, because the prior widens the gap between the chosen support and its nearest competitor, so a data perturbation must be larger to flip the selection. The same raise makes the support more sensitive to errors in $q$, because the prior enters the objective with weight $\mu$ and a wrong $q$ therefore steers more forcefully. Both effects are captured by a single exchange rate: the ratio of the $q$-invariance radius $\varepsilon^\star$ (the largest $\|q-q'\|_\infty$ that leaves the support unchanged) to the data-invariance radius $\eta^\star$ (the largest per-support loss perturbation that leaves the support unchanged) equals $1/\mu$ at a single-swap competitor. Halving $\mu$ doubles the tolerance for prior error and halves the tolerance for data perturbation. Appendix B derives this formally, and the brute-force checks of B.10 and B.11 confirm the radii on our own runs; what remains before they are treated as load-bearing is independent re-execution, not a missing proof.

Working at the support level with $F_q(S)=\ell(S)-\mu Q(S)$ and optimality gap $\Delta(q)$:

- **Theorem 1 (prior perturbation, tight).** If $\|q-q'\|_\infty<\varepsilon^\star=\min_S G_q(S)/(\mu|S\triangle S^\star|)$, the MAP support is unchanged.
- **Theorem 2 (data perturbation).** If the per-support loss moves by less than $\eta^\star=\Delta(q)/2$, the MAP is unchanged.
- The ratio $\varepsilon^\star/\eta^\star=1/\mu$ is the only $\mu$-monotone quantity: the prior trades data-variance for $q$-variance at a fixed rate. The optimality gap $\Delta(q)$ is itself non-monotone in $\mu$ because it vanishes at each support transition.
- **Lemma 1 (separation threshold).** The smallest $\mu$ at which the MAP leaves the loss-optimal support $S_{\mathrm{loss}}$ has a closed form, $\mu_0=\min_{Q(S)>Q(S_{\mathrm{loss}})}[\ell(S)-\ell(S_{\mathrm{loss}})]/[Q(S)-Q(S_{\mathrm{loss}})]$, the first crossing of the affine-in-$\mu$ support scores in their lower envelope.
- **A threshold below which the prior cannot displace the loss-optimal support.** A prior that is constant across all features gives $\mu_0=\infty$, so it can never move the loss-optimal support; we call this the do-no-harm property. This makes accuracy parity on held-out benchmarks a predicted corollary rather than only an observed tie (Section 4.2): the modification leaves FasterRisk's substrate unharmed by construction, not merely in measurement.
- **Theorem 3 (probabilistic stability).** With boxed weights the per-resample loss deviation obeys $\eta_b\le\varepsilon_n(\delta)=c_0 B\sqrt{(K\log p+\log(1/\delta))/n}$ with probability $1-\delta$, giving a stability rate that improves with $\Delta(q)$. This converts into a lower bound on the Nogueira stability index (a chance-corrected variant of Jaccard overlap across resamples) reported in Section 4.2. The union bound over $p^K$ supports is loose, so this is a rate statement about how stability improves with the prior's margin; realized stability is better than the bound predicts.
- **Theorem 4 (selection consistency).** Under population identifiability $\Delta_\infty(q)>0$, once $n\ge 4c_0^2 B^2(K\log p+\log(1/\delta))/\Delta_\infty(q)^2$ the prior-MAP support equals the population target, eventually almost surely. With estimated $q_n\to_p q_\infty$ it inherits the source's own consistency conditions (faithfulness for PC/GES, stability-selection conditions for MB learners).

Exact-MAP validation by brute-force enumeration (`exact_radii.py`, $p=12,K=3$) meets the tight $\varepsilon^\star$ with equality, confirms $\varepsilon^\star/\eta^\star\to1/\mu$, and reproduces the data-stability gain (vanilla MAP-stable 65% to prior 100% at zero violations). The radii also hold on the exact *integer* objective FasterRisk targets, not only a continuous proxy (`integer_radii.py`): the tight radius is verified and the prior pulls the integer MAP onto $S^\star$. And the deployed solver reaches the theory's optimum through its pool rather than its top card: the exact MAP support is in FasterRisk's diverse pool 80 to 100% of the time once the prior is on, and the prior surfaces it as the deployed scorecard (Section 4.5, Appendix B.11.1).

**Open target.** Characterizing the FasterRisk diverse pool as an approximation to the posterior mode region under the Bernoulli inclusion prior, quantifying the gap induced by the loss gap-tolerance, would connect the empirical pool result of Section 4.5 to the prior formally.

## 4. Experiments

The method makes four claims, each tested on the appropriate substrate (Sections 4.1, 4.2, 4.3, 4.5); Section 4.4 reports runtime rather than a claim.

**Status of claims.** A single ledger of what is proven, what is numerically validated, and what is still scoped to simulation, so a reader never has to guess whether a number is load-bearing.

| claim | status | backing | scope / caveat |
|---|---|---|---|
| modified objective is the MAP of a Bernoulli inclusion prior (§2.3) | proven | analytic derivation | independence across $j$ ignores collinearity |
| radii $\varepsilon^\star,\eta^\star$ and $\varepsilon^\star/\eta^\star=1/\mu$ (§3, Thm 1-2) | proven, numerically checked | `exact_radii.py`, `integer_radii.py` | support-level MAP; independent re-run pending |
| separation threshold $\mu_0$ closed form, do-no-harm (§3, Lemma 1) | proven | analytic | constant-$q$ case |
| probabilistic stability rate (§3, Thm 3) | proven, loose bound | analytic; realized stability better | union bound over $p^K$ supports |
| selection consistency (§3, Thm 4) | proven | analytic | needs $\Delta_\infty>0$ and source consistency |
| beam reaches the MAP support through its pool | numerically validated | `beam_gap.py` (80-100% pool) | $p=30,K=5$, 5 seeds, continuous reference |
| discovery recovers the planted support $S^\star$ (§4.1) | validated | recovery tables, 280 cells | synthetic linear-Gaussian sink |
| do-no-harm across 10 $q$-sources (§4.1) | validated | `q_robustness.py` | anchor cell |
| graceful degradation under $q$-corruption (§4.1) | validated | `q_corruption.py` | anchor cell |
| soft prior matches vanilla, beats hard CFS (§4.2) | validated | `cfs.py`, 5 benchmarks | public real data, single environment |
| out-of-environment transport (§4.3) | validated, synthetic only | `recovery_shift.py`, MB + GES | no real multi-environment test yet |
| runtime in seconds at $n=10^4$ (§4.4) | measured | `runtime_bench.py` | 5 benchmarks |
| predictive multiplicity resolved by $q$ (§4.5) | validated | `rashomon.py` | FICO, $k=10$ |

### 4.1 Markov-blanket recovery (synthetic, ground truth)

$q$ is the Markov blanket of the target $Y$; causal feature selection hard-selects that blanket, we softly prior it. On synthetic linear-Gaussian DAGs where $Y$ is a sink, $\mathrm{MB}(Y)=\mathrm{Pa}(Y)=S^\star$, so scoring a recovered blanket against the planted $S^\star$ is exactly MB-recovery scoring. Data generation: Erdős-Rényi DAG over $p$ continuous features plus a continuous sink $Y_{\text{lat}}$, planted sparse $S^\star=\mathrm{Pa}(Y_{\text{lat}})$, with the continuous $Y_{\text{lat}}$ fed to discovery and a median-thresholded binary $y$ fed to the scorecard (`src/data/synthetic_lingauss.py`). Discovery runs on the continuous $Y_{\text{lat}}$ rather than the thresholded $y$ because the causal mechanism operates on the continuous quantity and the conditional-independence relationships that PC and GES exploit are defined there; median thresholding distorts those relationships and would make the prior-estimation stage harder than necessary. Binarization is a constraint of FasterRisk's logistic loss, not of the causal estimation problem. The same principle holds for any application where the natural target is continuous: discovery runs on the continuous signal, and binarization is imposed only at the classifier stage.

The sparsity budget is set to $K=2k^\star$ so the prior has headroom: at $K=k^\star$ a separable signal saturates recovery and the prior can only break ties; the extra $k^\star$ slots are where vanilla admits the strongest confounded correlates, and the prior's effect on how those slots are used becomes observable.

Recovered blanket $\{j:q_j\ge0.5\}$ vs $S^\star$ over 280 cached cells (`loading.mb_recovery_table`):

| source | precision | recall | F1 |
|---|---|---|---|
| IAMB (MB recovery) | 0.98 | 1.00 | 0.99 |
| GES (global) | 0.66 | 0.90 | 0.74 |
| PC (global) | 0.62 | 0.21 | 0.42 |
| bootstrap-$L_1$ (predictive) | 0.30 | 0.75 | 0.41 |

Causal discovery recovers the true support. IAMB, which targets the Markov blanket directly, does so near-perfectly (F1 0.99). GES recovers the bulk of it (F1 0.74) but includes some non-parent ancestors at lower edge densities. PC degrades sharply as the graph becomes denser: constraint-based discovery is sample-starved on dense Gaussian DAGs and F1 falls to 0.42 at the Table 4.1 operating point. The predictive source has low precision because $L_1$ regularization includes confounders alongside causes and cannot distinguish them. The scorecard recovery sweep (`recovery_sweep_cv.py`, CV on log-loss to pick $\hat\mu$ per cell) quantifies the soft prior's effect on S_precision (fraction of selected features in $S^\star$). Both causal sources outperform the vanilla floor at every confounding density. The deployed MB source, IAMB, gains +0.13 to +0.28 over vanilla across $p_\text{edge}$, widening with confounding; GES tracks it closely (+0.13 to +0.29). The predictive bootstrap-$L_1$ hovers at the vanilla floor across densities, never cleanly separating causes from the confounded correlates. The effect requires a sparsity budget with headroom: at $K=k^\star$ the signal is strong enough that vanilla already fills the budget with causal features and the prior has no room to move the support; the gain appears from $K\ge1.5k^\star$ and is roughly flat across larger budgets. The benefit is largest at small-to-moderate $n$ and shrinks as $n$ grows and the data alone recovers $S^\star$, though IAMB keeps a margin even at $n=1000$ where the GES advantage has mostly closed. The p-sweep is sharpest: at $p=50$ the IAMB advantage grows to +0.42 (GES +0.34) because more candidate features give the prior more room to concentrate the $K$ slots on $S^\star$.

**q-source robustness** (`q_robustness.py`, anchor cell $p=30, n=300, k^\star=5, p_\text{edge}=0.2$, 20 seeds). Ten sources are compared across five paradigms: constraint-global (PC_stable), constraint-local (IAMB, GS, inter-IAMB), score-based (GES), continuous-optimization (DAGMA), predictive-stability (bootstrap-$L_1$), and three controls (oracle, uniform, adversarial). Per-source AUC is flat (0.994 to 0.998 across non-adversarial sources, spread 0.001); the prior causes no accuracy degradation regardless of which $q$ is supplied, consistent with the do-no-harm property of §3. Support quality tracks $q$ quality. The MB-local learners rank highest because they target exactly the Markov blanket the prior encodes: GS achieves S_precision 0.671, inter-IAMB 0.620, IAMB 0.611. Oracle (0.591) lands just below them: an all-or-nothing hard $q$ is sometimes distrusted by log-loss CV, which sets $\hat\mu\to0$ rather than committing to it. DAGMA (continuous-optimization, 0.531) and GES (score-based, global, 0.519) follow, then PC_stable (0.502). Uniform $q$ (0.485) is the vanilla floor. Bootstrap-$L_1$ (0.421) sits below the floor at this anchor, because the predictive source at moderate confounding does not concentrate evidence selectively on $S^\star$. The adversarial source (0.352) falls furthest below the floor and carries the lowest AUC of any source (0.993 against a non-adversarial mean of 0.997); it is the only source that both degrades recovery and nudges accuracy down, though that accuracy dip still sits inside the 0.001 across-source spread.

**q-corruption graceful degradation** (`q_corruption.py`, same anchor, starting from the GES prior, 20 seeds, 11 levels). The experiment interpolates $q$ from the GES prior toward uniform noise and scores both S_precision and held-out AUC at each level. Starting from S_precision 0.565 at corruption 0 (the GES source, consistent with the robustness panel's 0.519 up to the seed pool), recovery declines gradually to 0.463 at corruption 1.0 (pure noise $q$, approaching the vanilla floor of roughly 0.49). AUC stays flat throughout: 0.997 at corruption 0, 0.996 at corruption 1.0, a total spread of 0.001. CV tracks the degradation and lowers $\hat\mu$ as the prior becomes less informative, protecting accuracy automatically. A bad $q$ costs accuracy nothing; recovery slides to the no-information floor smoothly, not catastrophically.

### 4.2 Soft prior vs hard causal feature selection (public benchmarks)

How the soft prior compares to established causal feature selection (CFS, e.g. IAMB and HITON-MB), which outputs an estimated Markov blanket as a variable list, not a model. To compare on prediction we fit a FasterRisk scorecard on each method's selected features at matched sparsity (`experiments/causal_prior/real/cfs.py`, leakage-free, held-out AUC over repeated resamples). The arms form a $2\times2$ of soft-vs-hard against the conditional-independence test, so the soft-vs-hard effect is never confounded with the test: a soft prior and a hard selector both built on IAMB with the valid mixed-data (conditional-Gaussian) test, plus the off-the-shelf Fisher-Z selector as a naive reference. Every arm re-selects on each resample, otherwise a once-computed blanket is trivially stable and hides the instability we are measuring. Five benchmarks clear roughly five samples per feature (fico, heart, mammographic, ilpd, german); hepatitis (2.6) is the boundary case below.

Test AUC, mean $\pm$ std over resamples:

| dataset | vanilla | ours | CFS valid (mi-cg) | CFS naive (Fisher-Z) |
|---|---|---|---|---|
| fico | 0.763 | 0.762 | 0.640 | 0.765 |
| heart | 0.894 | 0.903 | 0.861 | 0.869 |
| mammographic | 0.864 | 0.865 | 0.826 | 0.827 |
| ilpd | 0.696 | 0.705 | 0.705 | 0.581 |
| german | 0.729 | 0.731 | 0.698 | 0.722 |

(spreads are $\pm0.01$ to $\pm0.06$ across resamples; ours is the IAMB soft prior, GES gives the same picture.)

- **Do no harm.** Ours matches or slightly beats vanilla on every benchmark, within the spread. The prior reshapes which features are picked at no accuracy cost; CV pulls the prior strength to zero when it is uninformative.
- **Beats the valid hard CFS.** Ours is above the mi-cg hard selector on four of five and ties on ilpd, never below it. On fico that hard selector collapses to 0.64, its blanket is too thin to support a scorecard, while the soft prior, which never hard-drops a feature, holds at 0.76.
- **Beats the naive Fisher-Z filter** everywhere except near-continuous fico, where that filter is well matched and ties; the same filter is invalid on mixed data and falls to 0.58 on ilpd.
- **Steadier selection.** Ours picks a more stable feature set than vanilla on four of five benchmarks (chance-corrected Nogueira stability index, where a score of 0 means random selection and 1 means identical selection across all splits). The Fisher-Z hard arms are steadier on heart and mammographic, at an accuracy cost of 2 to 4 points AUC; on mixed-data benchmarks where the Gaussian assumption is violated (ilpd) they collapse to 0.581 and any stability advantage is moot.
- **Boundary.** Below about five samples per feature (hepatitis, 2.6) discovery starves: the prior is diffuse yet CV keeps it on, so it costs a little accuracy. This is the operating boundary, not a failure; the prior should then come from outside the scarce sample.

The honest headline is not "highest AUC everywhere," it is: not dominated, soft beats hard, steadier selection when data is scarce, and an interpretable score that a feature selector does not give you.

### 4.3 Out-of-environment transport

A scorecard that relies on features correlated with the outcome only through confounding will fail when that confounding structure changes across deployment environments; one built on causal features should not. Testing this requires a controlled construction where the causal mechanism is known and fixed. Following the Invariant Causal Prediction framework (Peters et al. 2016)---which identifies causal features as those whose predictive relationship to the target remains stable across environments---`experiments/causal_prior/synthetic/recovery_shift.py` shifts the distribution by rescaling incoming edges to non-causal correlate nodes per environment ($\gamma$). Because correlates are never ancestors of $Y$ and causes never have correlate parents, the causal mechanism $P(Y\mid\mathrm{Pa}(Y))$ is invariant by construction while only the spurious correlate-to-$Y$ associations shift. The scorecard and evidence vector $q$ are trained at $\gamma=1$; AUC is scored at $\gamma\in\{1,0,-1\}$.

The construction places correlate associations under the shift by design. What is not guaranteed by the construction is that causal discovery actually produces a support free of those correlates at finite sample. The Markov-blanket sources with the soft prior do: GS gives a purely causal support in 78% of cells and IAMB in 63%, with the global score-based GES at 75%, across the parameter sweep. This is an empirical fact about the method, not a consequence of how the shift was built.

- **In-distribution parity, out-of-environment spread.** All operational sources reach in-distribution AUC $\approx0.99$ (spread 0.009); the same models spread by 0.060 in transport gap at $\gamma=-1$, while the adversarial control blows out to 0.237.
- **Transport gap and correlate reliance.** Pooled over cells, the transport gap rises with how much the selected support rests on correlate features (Pearson $r=0.58$); supports free of correlates transport with gap near zero.
- **The Markov-blanket sources transport best.** GS has the smallest transport gap (0.002) and the highest purely-causal rate (78%); IAMB follows (gap 0.004, 63%); the global GES matches them (gap 0.014, correlate reliance 0.034, 75% pure). All three beat vanilla (gap 0.049), the constraint-global PC (0.062), and the predictive bootstrap-$L_1$ (0.046), and reach the oracle ceiling (0.007). These are the sources the method deploys (§2.6) and the ones that recover best in Section 4.1, so a single source family wins both recovery and transport. The adversarial confounder-peaked prior collapses (gap 0.237) and degrades monotonically with confounding.

**Scope.** This experiment uses the MB-local learners (IAMB, GS) and GES on Gaussian synthetic data. PC is included (transport gap 0.062 pooled over cells, above vanilla's 0.049) but gives no improvement over vanilla and, as in Section 4.1, is not a surviving causal source under these linear-Gaussian conditions. The mean transport gap is modest when averaged across all cells (GS beats vanilla by 0.047 AUC, GES by 0.035) because most cells produce supports that are already largely causal regardless of source; the effect concentrates in cells where predictive shortcuts through confounders exist. The real-data version of this test, multi-environment data with known environment labels, is the main remaining gap-closer and is listed in Section 6.

### 4.4 Runtime and the fixed-mu fast variant

`experiments/causal_prior/real/runtime_bench.py` decomposes per-deployment wall-clock into discovery (the same step hard CFS runs) plus one FasterRisk fit, against the optional $\mu$-CV grid. The deployed variant `ours_fast` fixes $\mu=\texttt{mu\_fast\_rel}\cdot\mu_{\text{scale}}$ (no grid); `ours_full` adds 5-fold by 9-$\mu$ CV.

| dataset | $n$ | ours fast (s) | ours full (s) | fast vs full |
|---|---|---|---|---|
| fico | 10459 | 41 | 1199 | 29x |
| heart | 270 | 3.8 | 137 | 36x |
| mammographic | 831 | 2.6 | 95 | 37x |
| ilpd | 583 | 4.5 | 185 | 41x |
| german | 1000 | 5.0 | 174 | 35x |

The method (discovery plus one fit) runs in seconds even at $n=10^4$. The $\mu$-CV grid is the only slow part (29 to 41x) and buys no accuracy: the fixed and CV-tuned variants reach the same held-out AUC (the prior changes which support is picked, not the ranking), and CV occasionally even lands on a slightly worse $\mu$. The soft prior adds essentially no overhead over vanilla at fit time, and discovery is shared with hard CFS, so the method is speed-matched to hard CFS while scoring well above it on accuracy (the Section 4.2 table). Fixing $\mu$ is justified because AUC is $\mu$-invariant in this regime, $\mu$ is fixed relative to $\mu_{\text{scale}}$ so one global value transfers, and the chosen band sits above the Lemma 1 do-no-harm threshold and below over-shrink.

### 4.5 Resolving predictive multiplicity

Both vanilla and causal FasterRisk return a diverse pool of near-optimal scorecards (CollectSparseDiversePool, a Rashomon set at the chosen sparsity); what differs is the pool's composition and how the deployed card is chosen. Vanilla's pool disagrees on features at essentially equal accuracy (on heart, 15 distinct feature sets among the admissible members, mean causal mass 0.19), and its top member is an arbitrary element of that disagreeing set, so a responsible choice is offloaded to a domain expert who inspects the pool. The causal prior concentrates the pool on causally-supported features (mean causal mass rising from 0.19 with vanilla to 0.55 with the prior on heart, and from 0.33 to 0.41 on FICO) and supplies a principled selection criterion: the deployed top card is the member with highest causal support by construction. The method still computes a pool; what changes is that the choice of which card to deploy is resolved by $q$ rather than by manual expert triage, and a practitioner who still presents the pool now offers a set whose members all rest on justified features.

`experiments/causal_prior/real/rashomon.py`, leakage-free FICO ($k=10$), pool members within an AUC band of 0.01 of the best:

| arm | pool | mean $q$-mass | min to max $q$-mass | AUC range |
|---|---|---|---|---|
| vanilla | 50 | 0.332 | 0.27 to 0.37 | 0.767 to 0.775 |
| causal | 50 | 0.409 | 0.35 to 0.46 | 0.767 to 0.775 |

The admissible cards are accuracy-indistinguishable (within 0.008 AUC) yet differ in how much they rest on causal features; the prior slides the whole band upward (its least-grounded admissible card, 0.35, is about as causal as vanilla's most-grounded one, 0.37) at no accuracy cost. Per-member supports are logged so the disagreement and its resolution can be shown at the feature level.

## 5. Discussion

The benchmark comparison (Section 4.2) establishes that the modification is safe: the soft prior matches vanilla FasterRisk's accuracy on every benchmark tested and beats the hard causal selector, which collapses on FICO because its estimated Markov blanket is too sparse to support a scorecard. The do-no-harm property of Section 3 explains why: an uninformative prior cannot displace the loss-optimal support, so CV drives the prior strength to zero when the evidence is uninformative. What the benchmark comparison cannot establish is a causal advantage over a merely predictive prior, because single-environment data does not separate the two; the transport experiment provides that test.

The transport experiment (Section 4.3) shows that the Markov-blanket sources with the soft prior produce a support that retains accuracy under a controlled environment shift (GS gives a purely causal support in 78% of cells, at the smallest transport gap of any source), whereas the predictive source loses transport as confounders are pulled out of distribution. The scope of this result is synthetic data with a known causal graph; the real-data analogue requires multi-environment benchmarks with labeled environments, which is the main open question.

The support-stability analysis (Section 3) gives a unified account of the two empirical observations. The stability gain on scarce data (Section 4.2) and the adversarial collapse under a wrong prior are both consequences of the same $1/\mu$ exchange rate between data-variance and $q$-variance. An informative prior enlarges the gap between the selected support and its nearest competitor, which pins the support across resamples; the same prior amplifies the cost of a wrong $q$ proportionally. The safe operating regime is $\mu$ just above the do-no-harm threshold, where the stability gain is earned and the prior-fragility has not yet dominated.


## 6. Limitations and future work

- **$\mu$ tuning is empirical** with no closed-form scaling against $n$; the fixed relative $\mu$ of Section 4.4 is a defensible default, and a sensitivity sweep over a small band is the airtight version.
- **Faithfulness** is required wherever a causal graph is the prior source; failures show up as biased $q$.
- **Magnitude-vs-support tension at large $\mu$.** The bonus can drive low-magnitude features into the support that rounding may then eliminate if the multiplier is too small.
- **Independence of $z_j$ across $j$ ignores feature redundancy.** Under strong collinearity the per-feature $q_j$ does not capture select-one-not-both structure; the hard sparsity cap mitigates this only when $k$ is tight relative to the redundancy.
- **Single-solver evaluation.** All experiments use FasterRisk; whether the gain transports to RiskSLIM's MIP formulation is open.
- **The in-distribution comparison cannot separate a causal source from a merely selective one.** The distinctively causal advantage rests on out-of-environment transport (Section 4.3); a single-environment $q$ inherits environment-specific correlates, which is why transport, not in-distribution AUC, is where the causal claim is made.
- **Real multi-environment transport** is the main remaining gap-closer: the transport result is presently in simulation, and the natural next step is a real multi-environment benchmark (for example folktables, with shifts across regions and years). A single-environment $q$ inherits environment-specific correlates, so the source there should be invariance-based (Invariant Causal Prediction) or a cross-environment stability selection that keeps only features selected consistently across source environments.
- **Modern CFS baseline:** a gradient-based causal feature selector (GCFS) slots into the same downstream wrapper as the other CFS arms; it is recovery-oriented, so the honest framing of a recovery panel against it is "competitive," not "best."

## Appendix A. Implementation

Local modifications to FasterRisk's `sparseBeamSearch.py` (SparseBeamLR) and `sparseDiversePool.py` (CollectSparseDiversePool); the new state is the per-feature evidence vector (`self.freq`) and strength (`self.mu`). Pure Python, no asymptotic-complexity change, bit-identical to vanilla at $\mu=0$ or $q=\mathbf{0}$. FasterRisk's small $L_2$ ridge ($\lambda_2=10^{-8}$) is preserved exactly.

## Appendix B. Support-stability theory: full statements and proofs

Rigorous for the support-level MAP: the combinatorial choice of $S$ with a continuous within-support fit $\ell(S)$. This is the object FasterRisk's continuous beam stage targets before rounding; the beam-search gap (caveat B.11.1) and the rounding step (governed by Section 2.5 support preservation) are separate, acknowledged approximations. The radii are confirmed numerically in B.10 and B.11 on our own runs; what is still outstanding is independent re-execution before they are treated as load-bearing.

### B.1 Setup

Data fixed. For a support $S\subseteq[p]$ with $|S|\le K$, let

$$
\ell(S)\;=\;\min_{\substack{w:\ \mathrm{supp}(w)\subseteq S\\ w\in[-C,C]^p,\ w_0}} L(w,w_0,\mathcal{D})
$$

be the restricted logistic-loss minimum on support $S$ (well-defined: continuous loss over a compact box). It does not depend on the prior. The arguments below use only that $\ell(S)$ is a fixed per-support number independent of $q$; they therefore hold verbatim for any restricted minimum, the continuous boxed minimum or the integer-box minimum that FasterRisk ultimately targets (in the integer case the within-support optimum is over a finite grid, which only simplifies the uniform-convergence step of Theorem 3). We state it continuously because that is the object FasterRisk's beam stage selects the support over (integers come later, via the multiplier and rounding); the same radii govern the integer per-support objective. The remaining gap is therefore not an objective mismatch but the beam search itself: whether the heuristic reaches the exact MAP support, quantified in caveat B.11.1. With prior $q\in[0,1]^p$, strength $\mu\ge0$, and $Q(S)=\sum_{j\in S}q_j$, the support-level modified objective is $F_q(S)=\ell(S)-\mu Q(S)$, and the MAP support is $\hat S(q)=\arg\min_{|S|\le K}F_q(S)$ (finite candidate set; the minimizer is assumed unique, ties broken by a fixed rule). Define the optimality gap

$$
\Delta(q)\;=\;\min_{\substack{S\ne\hat S(q)\\ |S|\le K}}\big[\,F_q(S)-F_q(\hat S(q))\,\big]\;>\;0 .
$$

Two perturbations matter, and they behave oppositely in $\mu$.

### B.2 Theorem 1 (prior perturbation: turning up $\mu$ costs $q$-robustness)

Write $S^\star=\hat S(q)$ and, for any competitor $S$, its gap $G_q(S)=F_q(S)-F_q(S^\star)\ge\Delta(q)$. Let $q'$ satisfy $\lVert q-q'\rVert_\infty\le\varepsilon$. If

$$
\varepsilon\;<\;\varepsilon^\star\;:=\;\min_{S\ne S^\star}\frac{G_q(S)}{\mu\,|S\triangle S^\star|},
$$

then $\hat S(q')=\hat S(q)$: the MAP support is unchanged.

**Proof.** The prior enters $F$ only through $-\mu Q$, so for any $S$ the gap shifts by $G_{q'}(S)-G_q(S)=-\mu\sum_j(q'_j-q_j)(\mathbb 1[j\in S]-\mathbb 1[j\in S^\star])$. The indicator difference is supported only on $S\triangle S^\star$ and is $\pm1$ there, so $|G_{q'}(S)-G_q(S)|\le\mu\sum_{j\in S\triangle S^\star}|q'_j-q_j|\le\mu\,|S\triangle S^\star|\,\varepsilon$. Hence $G_{q'}(S)\ge G_q(S)-\mu\,|S\triangle S^\star|\,\varepsilon>0$ for every $S\ne S^\star$ whenever $\varepsilon<\varepsilon^\star$, so $S^\star$ strictly minimizes $F_{q'}$. $\blacksquare$

This is the tight prior-invariance radius: $\varepsilon^\star$ is the exact worst case. The adversary that attains it sets $q'_j-q_j=+\varepsilon$ on $S\setminus S^\star$ and $-\varepsilon$ on $S^\star\setminus S$ for the binding competitor.

**Remark (the easy bound, and where $K$ comes from).** Bounding the two endpoints separately, each by $\mu K\varepsilon$, gives the weaker one-line radius $\varepsilon_{\mathrm{easy}}=\Delta(q)/(2\mu K)\le\varepsilon^\star$, with $\varepsilon^\star/\varepsilon_{\mathrm{easy}}=2K/|S\triangle S^\star|$ at the binding competitor. The doubling and the $K$ come from discarding the cancellation in the indicator difference; they are bound slack, not a property of the MAP. Use $\varepsilon^\star$.

**Remark (box feasibility).** When $q$ lies on the boundary of $[0,1]^p$ some adversarial moves are infeasible, so the true box-constrained radius is $\ge\varepsilon^\star$; $\varepsilon^\star$ stays a valid sufficient radius.

### B.3 Theorem 2 (data perturbation: turning up $\mu$ buys data-robustness)

Suppose a data perturbation (subsample or CV fold) changes the restricted losses to $\ell'$ with $\sup_{|S|\le K}|\ell'(S)-\ell(S)|\le\eta$, and leaves $q$ fixed. If $\eta<\tfrac12\Delta(q)$, the MAP support is unchanged.

**Proof.** With $|F'_q(S)-F_q(S)|=|\ell'(S)-\ell(S)|\le\eta$, the gap change $G'_q(S)-G_q(S)=[\ell'(S)-\ell(S)]-[\ell'(S^\star)-\ell(S^\star)]$ and the margin argument give $\Delta(q)-2\eta>0$. $\blacksquare$

The data-invariance radius is $\eta^\star=\tfrac12\Delta(q)$. The factor of $2$ here is not slack: $\ell'(S)$ and $\ell'(S^\star)$ are two arbitrary, independent loss perturbations with no shared-coordinate structure to cancel, so the worst case genuinely moves the gap by $2\eta$. This radius is already tight.

### B.4 What is and isn't monotone in $\mu$

Both the tight prior radius $\varepsilon^\star(\mu)$ and the data radius $\eta^\star(\mu)=\tfrac{\Delta(q)}{2}$ are proportional to the binding gap. At a single-swap competitor (the generic nearest alternative, $|S\triangle S^\star|=2$) the ratio is

$$
\frac{\varepsilon^\star(\mu)}{\eta^\star(\mu)}\;=\;\frac{G/(2\mu)}{G/2}\;=\;\frac{1}{\mu}.
$$

This is the one $\mu$-monotone fact, and the cardinality factor is gone: increasing $\mu$ makes the support $1/\mu$ as robust to prior error as to data error. For a competitor swapping $m$ features the rate is $1/(\mu m)$, but the robust scaling is $1/\mu$. It is the precise form of "the prior trades data-variance for $q$-variance."

The individual radii are not globally monotone in $\mu$, because $\Delta(q)$ is not. Each $F_q(S)=\ell(S)-\mu Q(S)$ is a line in $\mu$ with slope $-Q(S)$; the MAP is their lower envelope, and the runner-up identity changes as $\mu$ grows. $\Delta(q)$ is therefore piecewise linear, vanishing at every $\mu$ where the MAP support transitions and peaking between transitions. The clean decomposition $\Delta=a+\mu b$ holds only within one stable-support interval. Below a binding threshold $\mu_0$ the prior does not move the support: $\hat S(q)=\arg\min_S\ell(S)=:S_{\mathrm{loss}}$, and a $q$-perturbation is inert. Above $\mu_0$ the explicit $1/\mu$ takes over.

### B.5 Lemma 1 (closed form for the separation threshold $\mu_0$)

Let $S_{\mathrm{loss}}=\arg\min_{|S|\le K}\ell(S)$ be the vanilla ($\mu=0$) support. Each $F_q(S)$ is affine in $\mu$ with intercept $\ell(S)$ and slope $-Q(S)$. A competitor $S$ can overtake $S_{\mathrm{loss}}$ at some $\mu>0$ only if $Q(S)>Q(S_{\mathrm{loss}})$; the two lines cross at $\mu(S)=[\ell(S)-\ell(S_{\mathrm{loss}})]/[Q(S)-Q(S_{\mathrm{loss}})]\ge0$ (numerator $\ge0$ since $S_{\mathrm{loss}}$ minimizes $\ell$; denominator $>0$ by assumption). Hence the smallest $\mu$ at which the MAP leaves $S_{\mathrm{loss}}$ is

$$
\mu_0\;=\;\min_{\substack{|S|\le K\\ Q(S)>Q(S_{\mathrm{loss}})}}\frac{\ell(S)-\ell(S_{\mathrm{loss}})}{Q(S)-Q(S_{\mathrm{loss}})},
$$

and the minimizing $S$ is the support the MAP jumps to at $\mu_0$. If no $|S|\le K$ has $Q(S)>Q(S_{\mathrm{loss}})$, then $\mu_0=\infty$ and the prior never moves the support. This is consistent with the transition picture: at $\mu_0$ the gap $\Delta(q)\to0$, the first support transition.

### B.6 Corollary (do-no-harm: an uninformative prior cannot demote the loss-optimal support)

Take an uninformative prior $q\equiv c\,\mathbf 1$, so $Q(S)=c\,|S|$. (i) Order preservation within a cardinality: for $|S|=|S'|$, $F_q(S)-F_q(S')=\ell(S)-\ell(S')$ for every $\mu$, so a constant prior never reorders supports of the same size. (ii) Inertness at budget: $Q(S)>Q(S_{\mathrm{loss}})$ requires $|S|>|S_{\mathrm{loss}}|$, so $\mu_0^{\mathrm{const}}=\min_{|S_{\mathrm{loss}}|<|S|\le K}[\ell(S)-\ell(S_{\mathrm{loss}})]/[c(|S|-|S_{\mathrm{loss}}|)]$; if the vanilla minimizer already uses the full budget, $\mu_0^{\mathrm{const}}=\infty$. This is the support-level statement behind the empirical "never dominated": an uninformative $q$ has no loss-side lever, only a sparsity knob, so it cannot select a higher-loss support than vanilla at the same size. It is also why log-loss CV drives $\hat\mu\to0$ for uninformative $q$.

### B.7 Theorem 3 (probabilistic data-stability and a Nogueira stability rate)

Theorem 2 is deterministic. We now bound $\eta$ with high probability and convert it into a lower bound on the Nogueira stability index $\hat\Phi$ of Section 4.2.

**Setup.** Resamples $b=1,\dots,m$. Binarized features lie in $\{0,1\}$ and weights in the box $[-C,C]^{|S|}\times[-C_0,C_0]$, so every margin obeys $|w\cdot x_S+w_0|\le KC+C_0=:M$ and the per-example loss lies in $[0,B]$ with $B=M+\log2$. Write $\ell_b(S)$ for the restricted boxed minimum on resample $b$ and $\eta_b=\sup_{|S|\le K}|\ell_b(S)-\ell(S)|$.

**Proposition.** There is a universal constant $c_0$ such that, with probability at least $1-\delta$ over a resample, $\eta_b\le\varepsilon_n(\delta):=c_0\,B\sqrt{(K\log p+\log(1/\delta))/n}$.

*Proof sketch.* The restricted minimum is $1$-Lipschitz in the sup-norm of the objective. For a fixed $S$ the class $\{x\mapsto\mathrm{loss}(w\cdot x_S+w_0):w\in\mathrm{box}\}$ is a $1$-Lipschitz transform of a bounded linear class in $|S|+1\le K+1$ dimensions, so its Rademacher complexity is $O(B\sqrt{K/n})$; a bounded-differences bound plus a union over the at most $p^K$ admissible supports (set $\delta'=\delta/p^K$) gives the stated uniform bound. $\blacksquare$

**Corollary (sample complexity).** Combining with Theorem 2: if $n\ge 4c_0^2B^2(K\log p+\log(1/\delta))/\Delta(q)^2$, then $\varepsilon_n(\delta)<\Delta(q)/2$ and a resample reproduces the MAP support with probability at least $1-\delta$.

**Corollary (Nogueira stability rate).** Let $\hat\rho$ be the fraction of the $m$ resamples whose support differs from $\hat S(q)$; then $\mathbb E[\hat\rho]\le\delta$. For every feature the selection frequency $\hat p_f$ lies within $\hat\rho$ of $\{0,1\}$, so $\hat p_f(1-\hat p_f)\le\hat\rho$ and $s_f^2=\tfrac{m}{m-1}\hat p_f(1-\hat p_f)\le\tfrac{m}{m-1}\hat\rho$. With $k=|\hat S(q)|$ and $d$ features,

$$
\mathbb E[1-\hat\Phi]\;\le\;\frac{m}{m-1}\cdot\frac{\delta}{(k/d)(1-k/d)} .
$$

**The role of the prior.** $\Delta(q)$ sits in the denominator of the sample complexity, and an informative $q$ that gives $\hat S(q)$ a genuine $Q$-margin enlarges $\Delta(q)$ beyond the vanilla loss gap. The prior therefore lowers the $n$ needed for a target stability, and the gain is largest where the vanilla loss gap is smallest (the near-tie regime). This is the theory behind the scarce-$n$ stability gain of Section 4.2.

**Caveats.** Inherits the support-level-MAP scope of Theorems 1 and 2. The union bound over $p^K$ supports is loose, so this is a rate statement, not a sharp prediction of $\hat\Phi$; realized stability is better.

### B.8 Large-sample behaviour: risk convergence and selection consistency

Let $\ell_\infty(S)=\min_{w\in\mathrm{box},\,\mathrm{supp}(w)\subseteq S}\mathbb E[\mathrm{loss}(w;x,y)]$ be the population restricted risk, $F_\infty(S)=\ell_\infty(S)-\mu Q(S)$, $S_{\mathrm{pop}}(q)=\arg\min_{|S|\le K}F_\infty(S)$, and $\Delta_\infty(q)=\min_{S\ne S_{\mathrm{pop}}}[F_\infty(S)-F_\infty(S_{\mathrm{pop}})]$.

**Proposition (uniform risk convergence).** With probability at least $1-\delta$, $\sup_{|S|\le K}|\ell_n(S)-\ell_\infty(S)|\le\varepsilon_n(\delta)=c_0B\sqrt{(K\log p+\log(1/\delta))/n}$. Same argument as B.7 with the population measure as the base.

**Corollary (risk convergence, no margin needed).** Because $\hat S_n(q)$ minimizes $F_n$ and $F_n$ is within $\varepsilon_n$ of $F_\infty$ uniformly, the ERM excess-risk inequality gives $F_\infty(\hat S_n(q))-F_\infty(S_{\mathrm{pop}}(q))\le 2\varepsilon_n(\delta)=O(B\sqrt{K\log p/n})$. The selected scorecard converges to the penalized population optimum in objective value at the $\sqrt{K\log p/n}$ rate, with no identifiability or margin assumption; this survives the near-tie regime where selection consistency fails (the support may flicker; the risk does not).

**Do-no-harm in risk units.** In the safe regime $\mu\le\mu_0$ the population minimizer is the unpenalized $K$-sparse risk optimum, so the prior steers to vanilla's target. Above $\mu_0$, $0\le\ell_\infty(S_{\mathrm{pop}})-\ell_\infty(S_{\mathrm{loss}}^\infty)\le\mu(Q(S_{\mathrm{pop}})-Q(S_{\mathrm{loss}}^\infty))$, the price paid to buy the $Q$-margin; it vanishes as $\mu\downarrow\mu_0$, so the safe regime gets the stability gain at no first-order risk cost.

**Theorem 4 (selection consistency).** Assume $\Delta_\infty(q)>0$. If $n\ge 4c_0^2B^2(K\log p+\log(1/\delta))/\Delta_\infty(q)^2$, then $\varepsilon_n(\delta)<\Delta_\infty(q)/2$ and the Theorem 2 margin argument gives $\hat S_n(q)=S_{\mathrm{pop}}(q)$ with probability at least $1-\delta$. Taking $\delta_n$ summable and Borel-Cantelli, $\hat S_n(q)=S_{\mathrm{pop}}(q)$ eventually almost surely.

**Estimated $q$.** If discovery returns $q_n$ with $\lVert q_n-q_\infty\rVert_\infty\to_p0$, Theorem 1 absorbs the prior perturbation once $\lVert q_n-q_\infty\rVert_\infty<\varepsilon^\star$, so $\hat S_n(q_n)=S_{\mathrm{pop}}(q_\infty)$ eventually; it inherits whatever conditions the $q$-source needs for consistency (faithfulness for PC/GES; the stability-selection conditions for the MB learners).

**Consistent to what.** $S_{\mathrm{pop}}(q)$ is the minimizer of population-risk-minus-prior over the boxed $K$-sparse class, the estimand the method defines, not by itself the causal or data-generating support. The two coincide for a well-specified model with an aligned (oracle causal) $q$ and can fail under a biased $q$ or misspecification. So Theorem 4 is consistency for the method's estimand; identifying that estimand with the causal truth is the separate question Sections 4.1 and 4.3 address.

**Parameter convergence.** Conditional on the selection event, the within-support fit is a smooth, strongly convex M-estimation problem on a fixed low-dimensional support, so $\hat w_n\to w_{\mathrm{pop}}$ at the $\sqrt n$ rate.

### B.9 Why the two empirical phenomena are one bound

The ratio $\varepsilon^\star/\eta^\star=1/\mu$ (single-swap competitor) is the exchange rate between data-variance and $q$-variance.

- **Adversarial collapse at large $\mu$.** $\varepsilon^\star$ carries an explicit $1/\mu$ and $\Delta(q)\to0$ at every support transition, so a wrong or perturbed $q$ flips the MAP support ever more easily as $\mu$ grows; the degradation is monotone-in-tendency, not graceful (the adversarial source in Section 4.1).
- **Stability gain (conditional, not a $\mu$-law).** An informative $q$ that gives a loss-reasonable support a real $Q$-margin enlarges $\Delta(q)$ beyond the vanilla loss gap, raising $\eta^\star$ and pinning the support against fold-to-fold loss perturbations (the scarce-$n$ stability gains of Section 4.2). This holds when $q$ is informative and $\mu$ sits just above $\mu_0$, where log-loss CV tends to place $\hat\mu$.

The safe regime is $\mu$ just above $\mu_0$: enough margin for data-stability, before the $1/\mu$ prior-fragility and the transition dips dominate.

### B.10 Numerical validation (exact MAP)

`experiments/causal_prior/synthetic/exact_radii.py` validates Theorem 1 by brute force: on a small cell ($p=12$, $n=200$, $k^\star=3$, $K=3$, $p_{\mathrm{edge}}=0.5$) it enumerates all 298 supports with $|S|\le K$, fits the restricted L2-regularized logistic loss once each, and computes the exact MAP, the gap, the easy bound, and the tight radius. With oracle $q$ (runner-up shares 2 of 3 true features):

- $\varepsilon_{\mathrm{easy}}\le\varepsilon^\star$ at every $\mu$ (valid but loose); the tight $\varepsilon^\star$ is the exact radius.
- $\varepsilon^\star/\varepsilon_{\mathrm{easy}}=2K/|S\triangle S^\star|=K$ here: the slack is the easy bound's doubling-plus-cardinality, not a property of the MAP.
- Asymptotics match: $\varepsilon_{\mathrm{easy}}\to b/(2K)$, $\varepsilon^\star\to b/2$, and $\varepsilon^\star/\eta^\star\to1/\mu$.

Theorem 2 is validated on the same cell by bootstrap (20 resamples): at $\mu=0$ the loss optimum is a near-tie and the bootstrap MAP is stable in only $65\%$ of resamples (recovers $S^\star$ in $0\%$); with the prior on, $\eta^\star$ grows with $\mu$ and the MAP is stable in $100\%$ and recovers $S^\star$ in $100\%$, with zero margin violations. This reproduces the data-stability gain ($65\%\to100\%$, recovery $0\%\to100\%$) in the exact setting, the same phenomenon as the Section 4.2 scarce-$n$ stability gains.

### B.11 Caveats

1. **Support-level MAP versus the deployed solver (measured, not assumed).** The radii are stated for the support-level MAP; three things sit between it and the deployed integer scorecard, and the first two are now measured. (i) *Objective.* The proofs use only that $\ell(S)$ is a fixed per-support number independent of $q$, so they hold for the integer-box objective FasterRisk targets, not only a continuous proxy; `integer_radii.py` confirms this at $p=12$, $K=k^\star$, brute-forcing the exact integer MAP over supports and integer coefficients: the tight prior radius $\varepsilon^\star$ is verified directly (the worst-case $q$-perturbation just below $\varepsilon^\star$ leaves the exact integer MAP fixed and just above flips it), and the prior pulls the integer MAP off the spurious loss-optimum onto $S^\star$. (ii) *Beam search.* FasterRisk's beam does not usually select the exact-MAP support as its top-1 (match $20$ to $40\%$ at $p=30$, $K=5$), but the exact MAP support is *in the diverse pool* $80$ to $100\%$ of the time once the prior is on (`beam_gap.py`); widening the beam barely changes this, so it is pool membership, not the top-1, that carries the link. The prior's role (Section 4.5) is exactly to surface that MAP-optimal member as the deployed card, so the theory governs the deployed method *through the admissible pool*: the optimum is reachable almost always, and the prior selects it. (iii) *Rounding.* Turning the selected support into integer coefficients is governed by Section 2.5 support preservation (preserved in the safe regime; the extreme-$\mu$ failure is the low-magnitude pathology described in §2.5 and §6). Scope: the pool numbers are $p=30$, $K=5$ over five seeds against a continuous reference; the headline $K=2k^\star$ enumeration is heavier and the direction is consistent but not yet a proof.
2. Uniqueness of $\hat S(q)$ ($\Delta(q)>0$) is assumed; degenerate ties need a tie-break and a separate argument, and $\Delta(q)\to0$ at support transitions is exactly where this is tightest.
3. The only $\mu$-monotone object is the ratio $\varepsilon^\star/\eta^\star=1/\mu$ (single-swap competitor); the individual radii inherit the non-monotone $\Delta(q)$. Do not state a global trend in the separate radii, and do not claim the cardinality factor $K$ in the exchange rate.
