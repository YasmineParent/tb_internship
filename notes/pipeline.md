# Causally-informed sparse risk scoring

A method for interpretable integer-coefficient risk scoring with a soft, threshold-free causal prior on feature selection. The document is organized as a paper: abstract, introduction, method, theory, experiments, related work, contributions, and limitations, with implementation and the full support-stability proofs in appendices.

## Abstract

Sparse integer-coefficient risk scores (scorecards) are valued in high-stakes settings because they can be read and checked by hand, but standard fitting selects features by predictive association alone and can lean on variables that merely correlate with the outcome through confounding. We introduce a soft, one-parameter prior that biases scorecard feature selection toward features supported by external per-feature causal evidence $q$. The prior is the MAP of a Bernoulli inclusion model with a sigmoid link, reducing to a single linear bonus on the support indicator; it preserves the decomposability and integer-rounding guarantees of the FasterRisk solver and recovers vanilla FasterRisk and hard pre-selection as its two limits. Sourcing $q$ from causal discovery, we show on synthetic data with known ground truth that the prior recovers the true sparse support and, the canonical payoff, keeps the scorecard invariant across an ICP-style environment shift where predictively-sourced evidence is pulled onto confounders and loses transport; the gain tracks how well discovery concentrates $q$ on causes rather than correlates, and within causal sources it appears where discovery succeeds and fades where it is sample-starved. An exact support-stability analysis explains the prior's two faces, a fold-to-fold stability gain and an adversarial fragility, as a single $1/\mu$ exchange between data-variance and prior-variance. On public benchmarks the prior matches vanilla accuracy at no compute overhead, improves selection stability most when data is scarce, and beats hard causal feature selection.

## 1. Introduction

Standard sparse-scorecard fitting selects features by predictive association alone and can lean on variables that correlate with the outcome only through confounding. We add a one-parameter prior that biases selection toward features supported by external per-feature causal evidence $q$, sourced from causal discovery, and preserves the guarantees of the FasterRisk solver it builds on.

Contributions:

1. A soft causal-prior penalty for sparse integer classification: the MAP of a Bernoulli inclusion prior, a single linear-in-$q$ bonus on the support indicator, threshold-free, with vanilla FasterRisk and hard pre-selection as its two limits (Section 2).
2. A decomposability-preserving integration into FasterRisk that inherits its integer-rounding bound (Section 2.5).
3. Causal discovery as the source of evidence that recovers the true support and, the canonical payoff, transports across environments where predictive sourcing does not (Section 4).
4. An exact support-stability analysis unifying the stability gain and the adversarial fragility as one $1/\mu$ exchange, with a closed-form do-no-harm threshold, a probabilistic stability bound, and a consistency result (Section 3).

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

Place a Bernoulli inclusion prior on the support indicators $z_j=\mathbb{1}[w_j\neq0]$, $z_j\sim\mathrm{Bernoulli}(\pi_j)$ independently, with a uniform conditional prior on $w_j\mid z_j=1$ over the integer box. This is the discrete spike-and-slab construction (Mitchell and Beauchamp 1988, George and McCulloch 1993) adapted to integer coefficients; the log-prior reduces to $\sum_j z_j\,\mathrm{logit}(\pi_j)+\text{const}$. Setting $\pi_j=\sigma(\mu q_j)$ gives $\log p(\mathbf{w})=\mu\sum_j q_j\mathbb{1}[w_j\neq0]+\text{const}$, so MAP estimation is exactly the modified objective. The sigmoid link makes the log-prior linear in $q_j$ with slope $\mu$. At $q_j=0$, $\pi_j=1/2$ (uniform inclusion at zero evidence): the prior treats $q$ as evidence on a positive scale, not as a probability centered at $1/2$. Independence across $j$ is the load-bearing assumption: it yields per-feature decomposability and matches the per-feature nature of $q_j$; collinearity-induced redundancy is mitigated only by the hard sparsity cap (Section 7).

### 2.4 Limits

$\mu=0$ recovers vanilla FasterRisk (uniform inclusion prior). $\mu\to\infty$ drives $\pi_j\to1$ for any $q_j>0$, so the $k$-constrained optimum collapses to the support maximizing $\sum_{j\in S}q_j$ (hard pre-selection by $q$). The single parameter $\mu$ interpolates between the two familiar endpoints with no threshold to set.

### 2.5 Structural properties

**Linear separability.** The bonus decomposes as $\sum_j q_j\mathbb{1}[w_j\neq0]$, so per-feature marginal cost is computable without recomputing global quantities. FasterRisk's SparseBeamLR expansion and CollectSparseDiversePool swap remain per-feature decomposable, so the modification is a one-line reweighting of the beam search with no asymptotic-complexity change, and is bit-identical to vanilla at $\mu=0$ or $q=\mathbf{0}$. The same linear-in-$z_j$ structure admits a one-line addition to RiskSLIM's MIP formulation (a linear coefficient $-\mu q_j$ on the inclusion indicator; not evaluated here).

**Magnitude invariance under rounding** (conditional on support preservation). The bonus depends only on $\mathbb{1}[w_j\neq0]$, not on $|w_j|$, so it is identically zero across integer rounding whenever the support is preserved. Under that condition FasterRisk's AuxiliaryLossRounding bound (their Theorem 3.1) transfers unchanged to the modified objective. Support preservation can fail only at extreme $\mu$ that forces low-magnitude features into the support (Section 7).

**Scale.** $\mu$ has no data-invariant scale, since $L$ grows with $n$ while $q$ is unit-free. We report $\mu$ relative to $\mu_{\text{scale}}=\mathrm{median}_j\,|\nabla_j L|$ at $\mathbf{w}=\mathbf{0}$ (equivalently $\mathrm{median}(0.5|X^\top y|)$ on binarized data), computed once per dataset, so a single relative $\mu$ is comparable across datasets.

### 2.6 The causal-evidence interface

**Requirement.** $q$ should come from a procedure that performs conditional-independence reasoning to remove confounding-driven associations. Predictive-only signals (bootstrap stability of LASSO or tree ensembles, marginal mutual information) are not used as $q$: they derive from the same logistic objective the classifier already optimizes, so treating them as a prior duplicates information rather than adding it.

The method is source-agnostic: the MAP construction holds regardless of which procedure produces $q$. Admissible sources include global discovery (PC, GES), Markov-blanket-local learners (IAMB and variants, HITON-MB), Invariant Causal Prediction in multi-environment settings, curated knowledge graphs with directional edges, and expert elicitation conditioned on causal status. All are used through subsample stability selection ($B$ runs, $q_j=\mathrm{freq}(j\to t)$). Because the deployed prior is the Markov blanket of the target (Section 4.1), MB-local learners target exactly what the method consumes; global discovery learns the whole graph and keeps only the target's neighbourhood.

**Propagation to binarized columns.** When the classifier stage binarizes a continuous feature into several indicator columns, the prior is defined at the original-feature level and each binarized column inherits its parent's value, $q^{\mathrm{bin}}_c=q^{\mathrm{orig}}_{\mathrm{parent}(c)}$. The causal structure lives at the original-feature level; binarized columns are downstream encoding choices and inherit the causal status of their parent. This step is a deliberate modelling choice, presented as such.

## 3. Theory: support stability

Working at the support level, $F_q(S)=\ell(S)-\mu Q(S)$ with optimality gap $\Delta(q)$ (Appendix B gives the full setup and proofs); the radii results await independent verification before they are treated as load-bearing.

- **Theorem 1 (prior perturbation, tight).** If $\|q-q'\|_\infty<\varepsilon^\star=\min_S G_q(S)/(\mu|S\triangle S^\star|)$, the MAP support is unchanged.
- **Theorem 2 (data perturbation).** If the per-support loss moves by less than $\eta^\star=\Delta(q)/2$, the MAP is unchanged.
- **The one monotone object** is the ratio $\varepsilon^\star/\eta^\star=1/\mu$: the prior trades data-variance for $q$-variance. This unifies the fold-to-fold stability gain (Theorem 2) and the adversarial-prior fragility (Theorem 1, explicit $1/\mu$) under a single bound. The gap $\Delta(q)$ is itself non-monotone in $\mu$ (it vanishes at support transitions).
- **Lemma 1 (separation threshold).** The smallest $\mu$ at which the MAP leaves the loss-optimal support $S_{\mathrm{loss}}$ has a closed form, $\mu_0=\min_{Q(S)>Q(S_{\mathrm{loss}})}[\ell(S)-\ell(S_{\mathrm{loss}})]/[Q(S)-Q(S_{\mathrm{loss}})]$, the first crossing of the affine-in-$\mu$ support scores in their lower envelope.
- **Do-no-harm corollary.** A prior constant on the budget (uninformative) gives $\mu_0=\infty$, so it can never demote the loss-optimal support: the prior is safe by construction. This makes substrate parity (Section 4.2) a predicted corollary rather than only an observed tie.
- **Theorem 3 (probabilistic stability and a Nogueira floor).** With boxed weights the per-resample loss deviation obeys $\eta_b\le\varepsilon_n(\delta)=c_0 B\sqrt{(K\log p+\log(1/\delta))/n}$ with probability $1-\delta$, giving a per-resample stability guarantee that converts into a floor on the Nogueira chance-corrected stability index reported in Section 4.2.
- **Theorem 4 (selection consistency).** Under population identifiability $\Delta_\infty(q)>0$, once $n\ge 4c_0^2 B^2(K\log p+\log(1/\delta))/\Delta_\infty(q)^2$ the prior-MAP support equals the population target, eventually almost surely. With estimated $q_n\to_p q_\infty$ it inherits the source's own consistency conditions (faithfulness for PC/GES, stability-selection conditions for MB learners).

Exact-MAP validation by brute-force enumeration (`exact_radii.py`, $p=12,K=3$) meets the tight $\varepsilon^\star$ with equality, confirms $\varepsilon^\star/\eta^\star\to1/\mu$, and reproduces the data-stability gain (vanilla MAP-stable 65% to prior 100% at zero violations).

**Open target.** Characterizing the FasterRisk diverse pool as an approximation to the posterior mode region under the Bernoulli inclusion prior, quantifying the gap induced by the loss gap-tolerance, would connect the empirical pool result of Section 4.5 to the prior formally.

## 4. Experiments

The method makes four claims, each tested on the appropriate substrate.

### 4.1 Markov-blanket recovery (synthetic, ground truth)

$q$ is the Markov blanket of the target $Y$; causal feature selection hard-selects that blanket, we softly prior it. On synthetic linear-Gaussian DAGs where $Y$ is a sink, $\mathrm{MB}(Y)=\mathrm{Pa}(Y)=S^\star$, so scoring a recovered blanket against the planted $S^\star$ is exactly MB-recovery scoring. Data generation: Erdős-Rényi DAG over $p$ continuous features plus a continuous sink $Y_{\text{lat}}$, planted sparse $S^\star=\mathrm{Pa}(Y_{\text{lat}})$, with the continuous $Y_{\text{lat}}$ fed to discovery and a median-thresholded binary $y$ fed to the scorecard (`src/data/synthetic_lingauss.py`).

The sparsity budget is set to $K=2k^\star$ so the prior has headroom: at $K=k^\star$ a separable signal saturates recovery and the prior can only break ties; the extra $k^\star$ slots are where vanilla admits the strongest confounded correlates, and the prior's effect on how those slots are used becomes observable.

Recovered blanket $\{j:q_j\ge0.5\}$ vs $S^\star$ over 280 cached cells (`loading.mb_recovery_table`):

| source | precision | recall | F1 |
|---|---|---|---|
| IAMB (MB recovery) | 0.98 | 1.00 | 0.99 |
| GES (global) | 0.66 | 0.90 | 0.74 |
| PC (global) | 0.62 | 0.21 | 0.42 |
| bootstrap-$L_1$ (predictive) | 0.30 | 0.75 | 0.41 |

Causal discovery recovers the true support, the MB-local learner near-perfectly; the predictive source has low precision because $L_1$ grabs correlates outside the blanket. The realized gain tracks how selective the recovered support is. Define the selectivity of a source as $\mathrm{sel}(q)=\bar q_C/\bar q_{S^\star}$, the ratio of mean evidence on confounded correlates $C$ to mean evidence on true causes $S^\star$ (lower is more selective). Recovery is a monotone-decreasing function of $\mathrm{sel}(q)$: GES attains $\mathrm{sel}\approx0.03$, bootstrap-$L_1$ about $0.43$, and PC degrades with confounding as constraint-based discovery becomes sample-starved on dense Gaussian DAGs. Causal discovery is the reliable route to a selective support at finite sample: the mechanism by which it recovers $S^\star$ is that it concentrates $q$ on causes rather than correlates, and where discovery is sample-starved (PC on dense Gaussian DAGs) that concentration and the recovery degrade together. The downstream sweep adding the recovered support across n, p, p_edge, and k_star is in `recovery_sweep_cv.py`.

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
- **Steadier selection.** Ours picks a more stable feature set than vanilla on all five (chance-corrected Nogueira index), the gain largest when data is scarce. The Fisher-Z hard arms are steadier still but stuck at much lower accuracy, stably mediocre.
- **Boundary.** Below about five samples per feature (hepatitis, 2.6) discovery starves: the prior is diffuse yet CV keeps it on, so it costs a little accuracy. This is the operating boundary, not a failure; the prior should then come from outside the scarce sample.

The honest headline is not "highest AUC everywhere," it is: not dominated, soft beats hard, steadier selection when data is scarce, and an interpretable score that a feature selector does not give you.

### 4.3 Out-of-environment transport (the causal payoff)

The canonical reason to prefer causal features is invariance under environment shift, which a single i.i.d. benchmark cannot test. `experiments/causal_prior/synthetic/recovery_shift.py` adds an ICP-style shift: one shared SCM with the incoming edges to non-causal correlate nodes rescaled per environment ($\gamma$). Because correlates are never ancestors of $Y$ and causes never have correlate parents, this leaves the causal mechanism $P(Y\mid\mathrm{Pa}(Y))$ invariant and moves only the spurious correlate-to-$Y$ associations. Train $q$ and scorecard at $\gamma=1$; score AUC at $\gamma\in\{1,0,-1\}$.

- **In-distribution parity, out-of-environment spread.** All operational sources reach in-distribution AUC $\approx0.99$ (spread 0.017); the same models spread by 0.230 in transport gap at $\gamma=-1$.
- **Mechanism.** Pooled over cells, the transport gap rises with how much the support leans on correlates (Pearson correlation about 0.61); a purely causal support transports with almost no gap. Low correlate reliance is what buys transport.
- **Causal discovery reaches that regime.** GES has the lowest correlate reliance (0.034) and the smallest transport gap (0.014), beats both vanilla and the predictive source, matches the oracle, and gives a purely causal support in 75% of cells. The adversarial confounder-peaked prior collapses (gap 0.237), worse the more it leans on confounders.

**Circularity caveat (stated for the reviewer).** The construction makes correlate associations shift-variant by design, so "correlate-reliant supports fail to transport" is true by setup. The non-trivial empirical claim is that causal discovery actually reaches the low-correlate-reliance regime at finite sample (a purely causal support in 75% of cells), which is a fact about the method, not the construction. The shift is on correlate edges only, so the gain is a low-density effect that closes as the correlate set shrinks, and the present run is GES-on-Gaussian (PC near-noise here). The mean gain is modest (GES beats vanilla by 0.035 AUC of transport gap) because it is diluted by the many cells where every source transports fine; the result carries on the mechanism and the 75% number, not the average magnitude, with the effect concentrated in the high-correlate-reliance cells where a predictive shortcut exists.

This is where a causally-sourced $q$ separates from a predictively-sourced one.

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

Both vanilla and causal FasterRisk return a diverse pool of near-optimal scorecards (CollectSparseDiversePool, a Rashomon set at the chosen sparsity); what differs is the pool's composition and how the deployed card is chosen. Vanilla's pool disagrees on features at essentially equal accuracy (on heart, 15 distinct feature sets among the admissible members, mean causal mass 0.19), and its top member is an arbitrary element of that disagreeing set, so a responsible choice is offloaded to a domain expert who inspects the pool. The causal prior concentrates the pool on externally justified features (mean causal mass 0.19 to 0.55 on heart, 0.33 to 0.41 on FICO) and supplies a principled selection criterion: the deployed top card is the causally grounded member by construction. The method still computes a pool; what changes is that the choice of which card to deploy is resolved by $q$ rather than by manual expert triage, and a practitioner who still presents the pool now offers a set whose members all rest on justified features.

`experiments/causal_prior/real/rashomon.py`, leakage-free FICO ($k=10$), pool members within an AUC band of 0.01 of the best:

| arm | pool | mean $q$-mass | min to max $q$-mass | AUC range |
|---|---|---|---|---|
| vanilla | 50 | 0.332 | 0.27 to 0.37 | 0.767 to 0.775 |
| causal | 50 | 0.409 | 0.35 to 0.46 | 0.767 to 0.775 |

The admissible cards are accuracy-indistinguishable (within 0.008 AUC) yet differ in how much they rest on causal features; the prior slides the whole band upward (its least-grounded admissible card, 0.35, is about as causal as vanilla's most-grounded one, 0.37) at no accuracy cost. Per-member supports are logged so the disagreement and its resolution can be shown at the feature level.

## 5. Related work

The prior sits at the intersection of five literatures; the novelty is in their intersection. (Citations are to be bibliographically reconfirmed before submission.)

**Interpretable risk scores.** Integer-coefficient scorecards optimized directly, rather than rounded from a continuous model, were formalized by RiskSLIM (Ustun and Rudin 2019) and accelerated by FasterRisk (Liu et al. 2022), which enlarges the searchable hypothesis class through a multiplier and a beam-search-plus-rounding pipeline with a certified loss bound; GroupFasterRisk (Zhu et al. 2025) adds monotonicity constraints and group sparsity. We build on FasterRisk as a substrate and inherit its guarantees; our contribution is a feature-selection prior that composes with this machinery without altering its optimization structure. This literature already offers three ways to inject domain knowledge: manual selection from the diverse pool, monotonicity constraints, and group sparsity. Ours is a fourth, a soft, per-feature, evidence-weighted inclusion prior derived as a MAP, sourced from causal discovery, and optimized inside the solver rather than applied by hand or as a hard constraint.

**Externally-informed and knowledge-guided penalization.** A long line of work modulates regularization with external information: the adaptive lasso (Zou 2006) reweights the penalty by an initial estimate, IPF-LASSO (Boulesteix et al. 2017) assigns penalty factors per data source, and feature-weighted methods set per-feature penalties from meta-features or co-data (xtune, Zeng et al. 2021; fwelnet, Tay et al. 2020). Closer to our setting, knowledge-guided regularization has been used inside interpretable predictive models: EYE (Wang et al. 2018) steers a clinical model toward expert-flagged covariates and validates by overlap with known risk factors, and Causal Regularization (Bahadori et al. 2017) places a per-feature causal weight in the penalty. All of these act on coefficient magnitude in a continuous convex problem. Our prior instead acts on the binary inclusion indicator, is invariant to coefficient magnitude, and is optimized combinatorially inside the integer-scorecard solver, where support and magnitude are distinct objects; and where these methods validate that the selected support is plausible, we make the evidence source the object of study (selectivity, provenance versus mechanism, transport).

**Bayesian variable selection.** Spike-and-slab priors (Mitchell and Beauchamp 1988; George and McCulloch 1993) and their structured variants, including covariate-dependent inclusion probabilities and the spike-and-slab lasso (Ročková and George 2018), encode prior beliefs about which features enter a model, typically via posterior sampling over continuous coefficients. We take the MAP of a Bernoulli inclusion prior, which reduces to a single linear penalty on the support and is solved by the scorecard optimizer, yielding an integer model with a transferable rounding bound rather than a posterior.

**Causal and invariance-based selection.** Markov-blanket and causal feature selection (Aliferis et al. 2010; Yu et al. 2020) identify features by causal relevance, usually by hard thresholding. Invariance-based methods, including invariant causal prediction (Peters et al. 2016), invariant risk minimization (Arjovsky et al. 2019), and anchor regression (Rothenhäusler et al. 2021), target out-of-distribution generalization by modifying the predictor, generally requiring multi-environment training data. We differ on both axes: our prior is soft, propagating selection uncertainty through $\mu$ rather than thresholding (we compare soft against hard directly), and we obtain transport from the selectivity of the selected support, sourced from single-environment causal discovery and evaluated under a held-out environment shift, rather than by training an invariant predictor.

**Positioning.** To our knowledge no prior work operates on the combinatorial integer-scorecard problem with a support-level (rather than magnitude-level) inclusion bonus that is the exact MAP of a Bernoulli inclusion prior and preserves the decomposability and rounding guarantees of a certified sparse-scorecard solver, while making the evidence source the object of study through selectivity and transport. The nearest precedents are EYE and Causal Regularization (knowledge-guided penalties in continuous predictive models) and the weighted-lasso family (co-data penalties); none reach the integer-scorecard, support-level-MAP, source-as-object-of-study intersection.

| Axis | Weighted lasso | Bayesian spike-and-slab | Causal feature selection | ICP / IRM / anchor | This work |
|---|---|---|---|---|---|
| Acts on | coefficient magnitude | coefficient + inclusion | inclusion (hard) | predictor / objective | inclusion (soft) |
| Optimization | continuous convex | posterior sampling | filter / wrapper | continuous | combinatorial integer |
| Output | continuous model | posterior | feature set | invariant predictor | integer scorecard |
| External info | any weights | structured prior | causal | environments | causal $q$, as object of study |
| Needs multi-env data | no | no | no | yes | no |
| Evaluated on | prediction | prediction / recovery | recovery | OOD risk | selectivity + transport + stability |
| Stability guarantee | oracle (asymptotic) | none stated | none | none | exact MAP radii, $1/\mu$ exchange |

## 6. Contributions

1. **A soft causal-prior penalty for sparse integer classification.** A Bernoulli inclusion prior with sigmoid link whose MAP is a single linear-in-$q$ bonus on the support indicator; threshold-free, one parameter, with vanilla FasterRisk and hard pre-selection as the two limits. Implemented, numerically equivalent to vanilla at $\mu=0$.
2. **Decomposability-preserving integration into FasterRisk.** Linear separability preserves the beam-search and diverse-pool structure and the integer-rounding bound transfers under support preservation. The same form admits a one-line RiskSLIM port (future work).
3. **Causal discovery as the source of evidence that recovers and transports.** Causally-sourced $q$ recovers the true sparse support (4.1) and, the canonical payoff, keeps the scorecard invariant across an ICP-style environment shift (4.3) where a predictively-sourced $q$ is pulled onto confounders and loses transport. The mechanism is that discovery yields selective supports; the GES-vs-PC gradient shows discovery quality drives the gain.
4. **An exact support-stability analysis** unifying the fold-to-fold stability gain and the adversarial fragility as one $1/\mu$ exchange, with a closed-form do-no-harm threshold, a probabilistic stability bound with a Nogueira floor, and a selection-consistency result.

## 7. Limitations and future work

- **$\mu$ tuning is empirical** with no closed-form scaling against $n$; the fixed relative $\mu$ of Section 4.4 is a defensible default, and a sensitivity sweep over a small band is the airtight version.
- **Faithfulness** is required wherever a causal graph is the prior source; failures show up as biased $q$.
- **Magnitude-vs-support tension at large $\mu$.** The bonus can drive low-magnitude features into the support that rounding may then eliminate if the multiplier is too small.
- **Independence of $z_j$ across $j$ ignores feature redundancy.** Under strong collinearity the per-feature $q_j$ does not capture select-one-not-both structure; the hard sparsity cap mitigates this only when $k$ is tight relative to the redundancy.
- **Single-solver evaluation.** All experiments use FasterRisk; whether the gain transports to RiskSLIM's MIP formulation is open.
- **The in-distribution comparison cannot separate a causal source from a merely selective one.** The distinctively causal advantage rests on out-of-environment transport (Section 4.3); a single-environment $q$ inherits environment-specific correlates, which is why transport, not in-distribution AUC, is where the causal claim is made.
- **Real multi-environment transport** is the main remaining gap-closer: the transport result is presently in simulation, and the natural next step is a real multi-environment benchmark (for example folktables, with shifts across regions and years). A single-environment $q$ inherits environment-specific correlates, so the source there should be invariance-based (Invariant Causal Prediction) or a cross-environment stability selection that keeps only features selected consistently across source environments.
- **Mechanism polish:** a $q$-corruption graceful-degradation curve and a controlled source comparison would characterize how the recovery gain depends on the selectivity of $q$, sharpening the account of why causal discovery helps and how gracefully it degrades when $q$ is imperfect.
- **Modern CFS baseline:** a gradient-based causal feature selector (GCFS) slots into the same downstream wrapper as the other CFS arms; it is recovery-oriented, so the honest framing of a recovery panel against it is "competitive," not "best."

## Appendix A. Implementation

Local modifications to FasterRisk's `sparseBeamSearch.py` (SparseBeamLR) and `sparseDiversePool.py` (CollectSparseDiversePool); the new state is the per-feature evidence vector (`self.freq`) and strength (`self.mu`). Pure Python, no asymptotic-complexity change, bit-identical to vanilla at $\mu=0$ or $q=\mathbf{0}$. FasterRisk's small $L_2$ ridge ($\lambda_2=10^{-8}$) is preserved exactly.

## Appendix B. Support-stability theory: full statements and proofs

Rigorous for the support-level MAP: the combinatorial choice of $S$ with a continuous within-support fit $\ell(S)$. This is the object FasterRisk's continuous beam stage targets before rounding; the beam-search gap (caveat B.11.1) and the rounding step (governed by Section 2.5 support preservation) are separate, acknowledged approximations. The radii results await independent verification before they are treated as load-bearing.

### B.1 Setup

Data fixed. For a support $S\subseteq[p]$ with $|S|\le K$, let

$$
\ell(S)\;=\;\min_{\substack{w:\ \mathrm{supp}(w)\subseteq S\\ w\in[-C,C]^p,\ w_0}} L(w,w_0,\mathcal{D})
$$

be the restricted logistic-loss minimum on support $S$ (well-defined: continuous loss over a compact box). It does not depend on the prior, and is the continuous boxed minimum, so the MAP below is combinatorial over supports with a continuous within-support fit, exactly the object FasterRisk selects the support over (integers come later, via the multiplier and rounding). Conditional on Section 2.5 support preservation, the integer scorecard inherits this continuously-selected support, so the theorem governs its support too. With prior $q\in[0,1]^p$, strength $\mu\ge0$, and $Q(S)=\sum_{j\in S}q_j$, the support-level modified objective is $F_q(S)=\ell(S)-\mu Q(S)$, and the MAP support is $\hat S(q)=\arg\min_{|S|\le K}F_q(S)$ (finite candidate set; the minimizer is assumed unique, ties broken by a fixed rule). Define the optimality gap

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

### B.7 Theorem 3 (probabilistic data-stability and a Nogueira floor)

Theorem 2 is deterministic. We now bound $\eta$ with high probability and convert it into a lower bound on the Nogueira stability index $\hat\Phi$ of Section 4.2.

**Setup.** Resamples $b=1,\dots,m$. Binarized features lie in $\{0,1\}$ and weights in the box $[-C,C]^{|S|}\times[-C_0,C_0]$, so every margin obeys $|w\cdot x_S+w_0|\le KC+C_0=:M$ and the per-example loss lies in $[0,B]$ with $B=M+\log2$. Write $\ell_b(S)$ for the restricted boxed minimum on resample $b$ and $\eta_b=\sup_{|S|\le K}|\ell_b(S)-\ell(S)|$.

**Proposition.** There is a universal constant $c_0$ such that, with probability at least $1-\delta$ over a resample, $\eta_b\le\varepsilon_n(\delta):=c_0\,B\sqrt{(K\log p+\log(1/\delta))/n}$.

*Proof sketch.* The restricted minimum is $1$-Lipschitz in the sup-norm of the objective. For a fixed $S$ the class $\{x\mapsto\mathrm{loss}(w\cdot x_S+w_0):w\in\mathrm{box}\}$ is a $1$-Lipschitz transform of a bounded linear class in $|S|+1\le K+1$ dimensions, so its Rademacher complexity is $O(B\sqrt{K/n})$; a bounded-differences bound plus a union over the at most $p^K$ admissible supports (set $\delta'=\delta/p^K$) gives the stated uniform bound. $\blacksquare$

**Corollary (sample complexity).** Combining with Theorem 2: if $n\ge 4c_0^2B^2(K\log p+\log(1/\delta))/\Delta(q)^2$, then $\varepsilon_n(\delta)<\Delta(q)/2$ and a resample reproduces the MAP support with probability at least $1-\delta$.

**Corollary (Nogueira floor).** Let $\hat\rho$ be the fraction of the $m$ resamples whose support differs from $\hat S(q)$; then $\mathbb E[\hat\rho]\le\delta$. For every feature the selection frequency $\hat p_f$ lies within $\hat\rho$ of $\{0,1\}$, so $\hat p_f(1-\hat p_f)\le\hat\rho$ and $s_f^2=\tfrac{m}{m-1}\hat p_f(1-\hat p_f)\le\tfrac{m}{m-1}\hat\rho$. With $k=|\hat S(q)|$ and $d$ features,

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

1. **Support-level MAP with a continuous within-support fit only.** Two approximations sit between it and the deployed integer scorecard. (i) Beam-search support gap: SparseBeamLR may not select the exact-MAP support, so the heuristic can violate both radii; the matched oracle is the exact continuous per-support optimum by brute-force enumeration (`exact_radii.py` at tiny $p$, `beam_gap.py` at the headline regime), which is directly measurable at $p=30$, $K=10$ but out of reach at very large $p$. (ii) Rounding: turning the continuously-selected support into integer coefficients is governed by Section 2.5 support preservation, preserved in the safe regime, with the extreme-$\mu$ failure being the low-magnitude pathology of Section 7.
2. Uniqueness of $\hat S(q)$ ($\Delta(q)>0$) is assumed; degenerate ties need a tie-break and a separate argument, and $\Delta(q)\to0$ at support transitions is exactly where this is tightest.
3. The only $\mu$-monotone object is the ratio $\varepsilon^\star/\eta^\star=1/\mu$ (single-swap competitor); the individual radii inherit the non-monotone $\Delta(q)$. Do not state a global trend in the separate radii, and do not claim the cardinality factor $K$ in the exchange rate.
