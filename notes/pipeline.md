# Causally-informed sparse risk scoring

A method for interpretable integer-coefficient risk scoring with a soft, threshold-free causal prior on feature selection. The document is organized as a paper: abstract, introduction, method, theory, experiments, related work (drafted separately), contributions, and limitations, with implementation and proofs in appendices. Proofs live in `notes/perturbation_theorem.md`.

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

Working at the support level, $F_q(S)=\ell(S)-\mu Q(S)$ with optimality gap $\Delta(q)$. Full statements and proofs in `notes/perturbation_theorem.md`; the radii results await independent verification before they are treated as load-bearing.

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

Causal discovery recovers the true support, the MB-local learner near-perfectly; the predictive source has low precision because $L_1$ grabs correlates outside the blanket. The realized gain tracks how selective the recovered support is. Define the selectivity of a source as $\mathrm{sel}(q)=\bar q_C/\bar q_{S^\star}$, the ratio of mean evidence on confounded correlates $C$ to mean evidence on true causes $S^\star$ (lower is more selective). Recovery is a monotone-decreasing function of $\mathrm{sel}(q)$: GES attains $\mathrm{sel}\approx0.03$, bootstrap-$L_1$ about $0.43$, and PC degrades with confounding as constraint-based discovery becomes sample-starved on dense Gaussian DAGs. So causal discovery is the reliable route to a selective support at finite sample, and the within-causal-family gradient (GES retains selectivity where PC does not) shows discovery quality, not the causal label alone, is what drives recovery. The downstream sweep adding the recovered support across n, p, p_edge, and k_star is in `recovery_sweep_cv.py`.

### 4.2 Soft prior vs hard causal feature selection (public benchmarks)

How the soft prior compares to established causal feature selection (CFS: IAMB, HITON-MB, and gradient variants), which output an estimated Markov blanket. CFS returns a variable list, not a model; to compare on prediction we fit a FasterRisk scorecard on each method's selected features at matched sparsity. Runner `experiments/causal_prior/real/cfs.py`: leakage-free, matched $K$, held-out AUC, paired Wilcoxon. Arms form a CI-test by use $2\times2$ so soft-vs-hard is never confounded with the conditional-independence test: soft prior (IAMB plus the valid mixed-data conditional-Gaussian test) vs hard CFS (the same IAMB plus the same test), with off-the-shelf Fisher-Z CFS arms as a naive reference. Six benchmarks; five (fico, heart, mammographic, ilpd, german) clear roughly five samples per feature, hepatitis is the boundary case. Every arm re-selects per resample: computing a fixed blanket once would make hard CFS trivially stable and hide the very instability the comparison probes. Stability is reported as the Nogueira chance-corrected index rather than raw Jaccard, since raw Jaccard inflates for the small CFS blankets (three to four features) against the prior's larger support.

- **Do-no-harm.** The soft prior matches or beats vanilla AUC on five of the six benchmarks (CV self-corrects $\hat\mu\to0$ when the prior is uninformative), the empirical face of the do-no-harm corollary; the one loss is the starved hepatitis boundary below.
- **Soft beats hard, CI test held fixed.** With the test held fixed, soft beats hard at the valid mixed-data test on five of six (one tie; FICO $+0.131$, heart $+0.042$ AUC) and at Fisher-Z on four of six. Soft is the safety net exactly where hard selection fails (a thin mixed-data blanket on FICO, a collapsed Fisher-Z blanket on ILPD) and neutral where hard already works, so the soft mechanism, not the CI test, is the active ingredient.
- **Never loses to the valid hard CFS** (six of six, one tie on ILPD).
- **Stability.** The soft prior gains on four of six (Nogueira), most when data is scarce (FICO $+0.37$ at small $n$), matching the regime dependence of Theorem 2. The hard Fisher-Z arms can be more stable but at significantly lower accuracy.
- **Not an accuracy-supremacy claim.** On near-continuous FICO the causal arm sits slightly below the off-the-shelf Fisher-Z filter, which is invalid on mixed data and itself collapses on genuinely mixed data; the honest headline is not-dominated, soft-beats-hard, scarce-regime stability, and an interpretable score that a feature selector does not provide.
- **Boundary.** Below roughly five samples per feature (hepatitis, 2.6) discovery is infeasible: the prior is diffuse yet CV keeps it on, so the method underperforms (the one do-no-harm loss). This is the operating boundary, not a failure; the prior should then come from outside the scarce sample.

Notebook: `notebooks/causal_prior/real/cfs_results.ipynb`.

### 4.3 Out-of-environment transport (the causal payoff)

The canonical reason to prefer causal features is invariance under environment shift, which a single i.i.d. benchmark cannot test. `experiments/causal_prior/synthetic/recovery_shift.py` adds an ICP-style shift: one shared SCM with the incoming edges to non-causal correlate nodes rescaled per environment ($\gamma$). Because correlates are never ancestors of $Y$ and causes never have correlate parents, this leaves the causal mechanism $P(Y\mid\mathrm{Pa}(Y))$ invariant and moves only the spurious correlate-to-$Y$ associations. Train $q$ and scorecard at $\gamma=1$; score AUC at $\gamma\in\{1,0,-1\}$.

- **In-distribution parity, out-of-environment spread.** All operational sources reach in-distribution AUC $\approx0.99$ (spread 0.017); the same models spread by 0.230 in transport gap at $\gamma=-1$.
- **Mechanism.** Pooled over cells, the transport gap is a monotone function of the support's correlate reliance (Pearson $r=0.61$; a purely causal support transports with gap $\approx0$). Low correlate reliance is what buys transport.
- **Causal discovery reaches that regime.** GES has the lowest correlate reliance (0.034) and smallest gap (0.014), beats vanilla ($p=0.002$) and the predictive source ($p=0.023$, paired Wilcoxon), is indistinguishable from oracle, and yields a purely causal support in 75% of cells. The adversarial confounder-peaked prior collapses (gap 0.237), monotone in confounding.

**Circularity caveat (stated for the reviewer).** The construction makes correlate associations shift-variant by design, so "correlate-reliant supports fail to transport" is true by setup. The non-trivial empirical claim is that causal discovery actually reaches the low-correlate-reliance regime at finite sample (a purely causal support in 75% of cells), which is a fact about the method, not the construction. The shift is on correlate edges only, so the gain is a low-density effect that closes as the correlate set shrinks, and the present run is GES-on-Gaussian (PC near-noise here). The mean gain is modest (GES beats vanilla by 0.035 AUC of transport gap) because it is diluted by the many cells where every source transports fine; the result carries on significance and mechanism rather than magnitude, with the effect concentrated in the high-correlate-reliance cells where a predictive shortcut exists.

This is where a causally-sourced $q$ separates from a predictively-sourced one. Notebook: `notebooks/causal_prior/synthetic/recovery_shift_plots.ipynb`.

### 4.4 Runtime and the fixed-mu fast variant

`experiments/causal_prior/real/runtime_bench.py` decomposes per-deployment wall-clock into discovery (the same step hard CFS runs) plus one FasterRisk fit, against the optional $\mu$-CV grid. The deployed variant `ours_fast` fixes $\mu=\texttt{mu\_fast\_rel}\cdot\mu_{\text{scale}}$ (no grid); `ours_full` adds 5-fold by 9-$\mu$ CV.

| dataset | $n$ | ours fast (s) | ours full (s) | fast vs full | AUC fast / full |
|---|---|---|---|---|---|
| fico | 10459 | 41 | 1199 | 29x | 0.781 / 0.785 |
| heart | 270 | 3.8 | 137 | 36x | 0.909 / 0.911 |
| mammographic | 831 | 2.6 | 95 | 37x | 0.846 / 0.846 |
| ilpd | 583 | 4.5 | 185 | 41x | 0.714 / 0.672 |
| german | 1000 | 5.0 | 174 | 35x | 0.727 / 0.737 |

The method (discovery plus one fit) runs in seconds even at $n=10^4$. The $\mu$-CV grid is the only slow part (29 to 41x) and buys no AUC: fast matches full within 0.01 to 0.04, and on ilpd beats it. The soft prior adds essentially no overhead over vanilla FasterRisk at fit time, and discovery is shared with hard CFS, so the method is speed-matched to hard CFS while scoring higher. Fixing $\mu$ is justified because AUC is $\mu$-invariant in this regime (the prior reshapes which features are picked, not the ranking), $\mu$ is fixed relative to $\mu_{\text{scale}}$ so one global value transfers, and the chosen band sits above the Lemma 1 do-no-harm threshold and below over-shrink.

### 4.5 Resolving predictive multiplicity

Both vanilla and causal FasterRisk return a diverse pool of near-optimal scorecards (CollectSparseDiversePool, a Rashomon set at the chosen sparsity); what differs is the pool's composition and how the deployed card is chosen. Vanilla's pool disagrees on features at essentially equal accuracy (on heart, 15 distinct feature sets among the admissible members, mean causal mass 0.19), and its top member is an arbitrary element of that disagreeing set, so a responsible choice is offloaded to a domain expert who inspects the pool. The causal prior concentrates the pool on externally justified features (mean causal mass 0.19 to 0.55 on heart, 0.33 to 0.41 on FICO) and supplies a principled selection criterion: the deployed top card is the causally grounded member by construction. The method still computes a pool; what changes is that the choice of which card to deploy is resolved by $q$ rather than by manual expert triage, and a practitioner who still presents the pool now offers a set whose members all rest on justified features.

`experiments/causal_prior/real/rashomon.py`, leakage-free FICO ($k=10$), pool members within an AUC band of 0.01 of the best:

| arm | pool | mean $q$-mass | min to max $q$-mass | AUC range |
|---|---|---|---|---|
| vanilla | 50 | 0.332 | 0.27 to 0.37 | 0.767 to 0.775 |
| causal | 50 | 0.409 | 0.35 to 0.46 | 0.767 to 0.775 |

The admissible cards are accuracy-indistinguishable (within 0.008 AUC) yet differ in how much they rest on causal features; the prior slides the whole band upward (its least-grounded admissible card, 0.35, is about as causal as vanilla's most-grounded one, 0.37) at no accuracy cost. Per-member supports are logged so the disagreement and its resolution can be shown at the feature level.

## 5. Related work

Drafted separately in `notes/related_work_draft.md`. In brief, the prior sits at the intersection of five literatures (interpretable sparse risk scores; externally-informed and knowledge-guided penalization; Bayesian variable selection with informative inclusion priors; causal and Markov-blanket feature selection; invariance-based out-of-distribution generalization), and is distinguished by acting on the binary inclusion indicator inside a combinatorial integer solver (rather than on coefficient magnitude in a continuous convex problem), by being the exact MAP of a Bernoulli inclusion prior, and by making the evidence source the object of study through selectivity and transport rather than only validating that the selected support is plausible.

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
- **Mechanism polish:** a selective predictive source and a $q$-corruption graceful-degradation curve would confirm directly that selectivity, not the causal label, is the mediator.
- **Modern CFS baseline:** a gradient-based causal feature selector (GCFS) slots into the same downstream wrapper as the other CFS arms; it is recovery-oriented, so the honest framing of a recovery panel against it is "competitive," not "best."

## Appendix A. Implementation

Local modifications to FasterRisk's `sparseBeamSearch.py` (SparseBeamLR) and `sparseDiversePool.py` (CollectSparseDiversePool); the new state is the per-feature evidence vector (`self.freq`) and strength (`self.mu`). Pure Python, no asymptotic-complexity change, bit-identical to vanilla at $\mu=0$ or $q=\mathbf{0}$. FasterRisk's small $L_2$ ridge ($\lambda_2=10^{-8}$) is preserved exactly.

## Appendix B. Proofs

Full statements, proofs, and the exact-MAP numerical validation of Section 3 are in `notes/perturbation_theorem.md`.
