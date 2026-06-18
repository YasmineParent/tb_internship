# Support-stability of the causal-prior MAP

Draft of the §5 robustness target. Rigorous for the **support-level MAP**: the
combinatorial choice of $S$ with a *continuous* within-support fit $\ell(S)$
(see Setup). This is the object FasterRisk's continuous beam stage targets before
rounding; the beam-search gap (caveat 1) and the rounding step (governed by §2.4
support-preservation) are separate, acknowledged approximations. To be verified
before use.

## Setup

Data fixed. For a support $S \subseteq [p]$ with $|S| \le K$, let

$$
\ell(S) \;=\; \min_{\substack{w:\ \mathrm{supp}(w)\subseteq S \\ w \in [-C,C]^p,\ w_0}} L(w, w_0, \mathcal{D})
$$

be the restricted logistic-loss minimum on support $S$ (well-defined: continuous
loss over a compact box). It does **not** depend on the prior. Note $\ell(S)$ is
the *continuous* boxed minimum, so the "MAP" below is combinatorial over supports
with a continuous within-support fit, which is exactly the object FasterRisk
selects the support over (SparseBeamLR is continuous; integers come later, via the
multiplier and rounding). Conditional on §2.4 support-preservation (the multiplier
$m$ large enough that no $|m\,w_j|$ rounds to zero), the integer scorecard inherits
this continuously-selected support, so the theorem governs its support too. That
condition holds in the safe regime ($\mu$ just above $\mu_0$) and can fail only at
extreme $\mu$, the low-$|w_j|$ pathology already flagged in §8; it is that existing
caveat, not a new one. With prior
$q \in [0,1]^p$, strength $\mu \ge 0$, and $Q(S) = \sum_{j \in S} q_j$, the
support-level modified objective is

$$
F_q(S) \;=\; \ell(S) \;-\; \mu\, Q(S),
$$

and the MAP support is $\hat S(q) = \arg\min_{|S|\le K} F_q(S)$ (finite set of
candidates; assume the minimizer is unique, ties broken by a fixed rule). Define
the **optimality gap**

$$
\Delta(q) \;=\; \min_{\substack{S \ne \hat S(q) \\ |S| \le K}} \big[\, F_q(S) - F_q(\hat S(q)) \,\big] \;>\; 0 .
$$

Two perturbations matter, and they behave oppositely in $\mu$.

## Theorem 1 (prior perturbation: turning up $\mu$ costs $q$-robustness)

Write $S^\star = \hat S(q)$ and, for any competitor $S$, its gap
$G_q(S) = F_q(S) - F_q(S^\star) \ge \Delta(q)$. Let $q'$ satisfy
$\lVert q - q' \rVert_\infty \le \varepsilon$. If

$$
\varepsilon \;<\; \varepsilon^\star \;:=\; \min_{S \ne S^\star} \frac{G_q(S)}{\mu\,|S \triangle S^\star|},
$$

then $\hat S(q') = \hat S(q)$: the MAP support is unchanged.

**Proof.** The prior enters $F$ only through $-\mu Q$, so for any $S$ the *gap*
shifts by

$$
G_{q'}(S) - G_q(S)
= -\mu \sum_j (q'_j - q_j)\big(\mathbb 1[j\in S]-\mathbb 1[j\in S^\star]\big).
$$

The indicator difference is supported only on $S \triangle S^\star$ and is $\pm 1$
there, so

$$
\big|G_{q'}(S) - G_q(S)\big| \;\le\; \mu \sum_{j \in S \triangle S^\star} |q'_j - q_j|
\;\le\; \mu\,|S \triangle S^\star|\,\varepsilon .
$$

Hence $G_{q'}(S) \ge G_q(S) - \mu\,|S \triangle S^\star|\,\varepsilon > 0$ for every
$S \ne S^\star$ whenever $\varepsilon < \varepsilon^\star$, so $S^\star$ strictly
minimizes $F_{q'}$. $\qquad\blacksquare$

This is the **tight prior-invariance radius**: $\varepsilon^\star$ is the exact
worst-case radius. The adversary that attains it sets $q'_j - q_j = +\varepsilon$
on $S \setminus S^\star$ and $-\varepsilon$ on $S^\star \setminus S$ for the binding
competitor, attaining the bound with equality, and it equals the brute-force
$\varepsilon_{\mathrm{adv}}$ measured in `exact_radii.py`.

**Remark (the easy bound, and where $K$ comes from).** Bounding the two endpoints
$|F_{q'}(S)-F_q(S)|$ and $|F_{q'}(S^\star)-F_q(S^\star)|$ *separately*, each by
$\mu K \varepsilon$, gives the weaker but one-line radius

$$
\varepsilon_{\mathrm{easy}} \;=\; \frac{\Delta(q)}{2\mu K} \;\le\; \varepsilon^\star,
\qquad \frac{\varepsilon^\star}{\varepsilon_{\mathrm{easy}}} \;=\; \frac{2K}{|S\triangle S^\star|}
$$

at the binding competitor. The doubling and the $K$ both come from discarding the
cancellation in the indicator difference (replacing $|S\triangle S^\star|$ by its
maximum $2K$); they are **bound slack, not a property of the MAP**. Use
$\varepsilon^\star$.

**Remark (box feasibility).** $\varepsilon^\star$ is the unconstrained worst case.
When $q$ lies on the boundary of $[0,1]^p$ (e.g. an oracle $q$ with $0/1$ entries)
some adversarial moves are infeasible, so the true box-constrained radius is
$\ge \varepsilon^\star$; $\varepsilon^\star$ stays a valid sufficient radius.

## Theorem 2 (data perturbation: turning up $\mu$ buys data-robustness)

Suppose a data perturbation (subsample / CV fold) changes the restricted losses to
$\ell'$ with $\sup_{|S|\le K} |\ell'(S) - \ell(S)| \le \eta$, and leaves $q$ fixed.
If $\eta < \tfrac{1}{2}\Delta(q)$, then the MAP support is unchanged.

**Proof.** With $|F'_q(S) - F_q(S)| = |\ell'(S)-\ell(S)| \le \eta$,
$G_{q}'(S) - G_q(S) = [\ell'(S)-\ell(S)] - [\ell'(S^\star)-\ell(S^\star)]$, and the
margin argument gives $\Delta(q) - 2\eta > 0$. $\qquad\blacksquare$

The **data-invariance radius** is $\eta^\star = \tfrac{1}{2}\Delta(q)$. Unlike the
prior side, the factor of $2$ here is **not** slack: $\ell'(S)$ and $\ell'(S^\star)$
are two arbitrary, independent loss perturbations with no shared-coordinate
structure to cancel, so the worst case genuinely moves the gap by $2\eta$. This
radius is already tight.

## What is and isn't monotone in $\mu$

Both the tight prior radius $\varepsilon^\star(\mu) = \min_{S}\tfrac{G_q(S)}{\mu|S\triangle S^\star|}$
(Thm 1) and the data radius $\eta^\star(\mu) = \tfrac{\Delta(q)}{2}$ (Thm 2) are
proportional to the binding gap. At a single-swap competitor (the generic nearest
alternative, $|S \triangle S^\star| = 2$) the ratio is

$$
\frac{\varepsilon^\star(\mu)}{\eta^\star(\mu)} \;=\; \frac{G/(2\mu)}{G/2} \;=\; \frac{1}{\mu}.
$$

This is the **one $\mu$-monotone fact**, and the cardinality factor is gone:
increasing $\mu$ makes the support $1/\mu$ as robust to prior error as to data error.
The earlier "$1/(\mu K)$, fragile by exactly $\mu K$" was an artifact of the easy
bound (which replaced $|S\triangle S^\star|$ by its maximum $2K$); $K$ does not
survive tightening, so do not claim it. For a competitor swapping $m$ features
($|S\triangle S^\star| = 2m$) the rate is $1/(\mu m)$ (denser swaps are
comparatively more prior-fragile), but the robust scaling is $1/\mu$. It is the
precise form of "the prior trades data-variance for $q$-variance."

The **individual** radii are *not* globally monotone in $\mu$, because $\Delta(q)$ is
not. Each $F_q(S) = \ell(S) - \mu Q(S)$ is a line in $\mu$ with slope $-Q(S)$; the
MAP is their lower envelope, and the runner-up identity changes as $\mu$ grows.
$\Delta(q)$ is therefore piecewise linear, **vanishing at every $\mu$ where the MAP
support transitions** and peaking between transitions. So $\varepsilon^\star$ and
$\eta^\star$ rise and dip *in tandem* within each stable-support interval, with
stability dips at the transitions. The clean decomposition $\Delta = a + \mu b$ holds
only *within* one interval (fixed $S^\star, S_2$), where $a = \ell(S_2)-\ell(S^\star)$
and $b = Q(S^\star)-Q(S_2)$; both $a,b$ jump at each transition, so one must not read
a global trend off it.

Below a binding threshold $\mu_0$ the prior does not move the support:
$\hat S(q) = \arg\min_S \ell(S) =: S_{\mathrm{loss}}$, and a $q$-perturbation is inert
($\varepsilon^\star$ large). $\mu_0$ is the smallest $\mu$ at which
$\hat S(q) \ne S_{\mathrm{loss}}$, the "separation threshold" of §5. Above $\mu_0$
the explicit $1/\mu$ in $\varepsilon^\star$ takes over.

## Lemma 1 (closed form for the separation threshold $\mu_0$)

Let $S_{\mathrm{loss}} = \arg\min_{|S|\le K}\ell(S)$ be the vanilla ($\mu=0$) support.
Each $F_q(S)=\ell(S)-\mu Q(S)$ is affine in $\mu$ with intercept $\ell(S)$ and slope
$-Q(S)$, and the MAP is their lower envelope. A competitor $S$ can overtake
$S_{\mathrm{loss}}$ at some $\mu>0$ only if its line falls faster, i.e.
$Q(S)>Q(S_{\mathrm{loss}})$; the two lines cross at

$$
\mu(S) \;=\; \frac{\ell(S)-\ell(S_{\mathrm{loss}})}{Q(S)-Q(S_{\mathrm{loss}})}\;\ge\;0
$$

(numerator $\ge 0$ since $S_{\mathrm{loss}}$ minimizes $\ell$; denominator $>0$ by
assumption). A competitor with $Q(S)\le Q(S_{\mathrm{loss}})$ never crosses for
$\mu>0$. Hence the smallest $\mu$ at which the MAP leaves $S_{\mathrm{loss}}$ is

$$
\mu_0 \;=\; \min_{\substack{|S|\le K\\ Q(S)>Q(S_{\mathrm{loss}})}}
\frac{\ell(S)-\ell(S_{\mathrm{loss}})}{Q(S)-Q(S_{\mathrm{loss}})},
$$

and the minimizing $S$ is the support the MAP jumps to at $\mu_0$. If no $|S|\le K$
has $Q(S)>Q(S_{\mathrm{loss}})$, then $\mu_0=\infty$ and the prior never moves the
support.

This is consistent with the transition picture above: at $\mu_0$ the runner-up $S_2$
attaining the minimum has $Q(S_2)>Q(S_{\mathrm{loss}})$, so in $\Delta(q)=a+\mu b$
with $a=\ell(S_2)-\ell(S_{\mathrm{loss}})\ge 0$ and $b=Q(S_{\mathrm{loss}})-Q(S_2)<0$,
the gap decreases in $\mu$ and reaches $0$ exactly at $-a/b=\mu_0$. So $\mu_0$ is the
first support transition, where $\Delta(q)\to 0$ as already noted. This is the closed
form left open in caveat 4.

## Corollary (do-no-harm: an uninformative prior cannot demote the loss-optimal support)

Take an uninformative prior $q\equiv c\,\mathbf 1$ (every feature equally credible),
so $Q(S)=c\,|S|$. Two consequences follow from Lemma 1.

(i) *Order preservation within a cardinality.* For any two supports with $|S|=|S'|$,
$Q(S)=Q(S')=c|S|$, so $F_q(S)-F_q(S')=\ell(S)-\ell(S')$ for every $\mu$. A constant
prior never reorders supports of the same size: it cannot promote a higher-loss
support above a lower-loss one. Its only degree of freedom is total sparsity, through
the size reward $-\mu c|S|$.

(ii) *Inertness at budget.* By Lemma 1, $Q(S)>Q(S_{\mathrm{loss}})$ requires
$|S|>|S_{\mathrm{loss}}|$, so

$$
\mu_0^{\mathrm{const}} \;=\;
\min_{|S_{\mathrm{loss}}|<|S|\le K}\frac{\ell(S)-\ell(S_{\mathrm{loss}})}{c\,(|S|-|S_{\mathrm{loss}}|)} .
$$

In particular, if the vanilla minimizer already uses the full budget
($|S_{\mathrm{loss}}|=K$), there is no admissible larger competitor and
$\mu_0^{\mathrm{const}}=\infty$: a constant prior leaves the MAP at $S_{\mathrm{loss}}$
for every $\mu$. When $|S_{\mathrm{loss}}|<K$, the sole effect of raising $\mu$ is to
fill the support toward $K$ with the next-lowest-loss features, never to swap among a
fixed size.

This is the support-level statement behind the empirical "never dominated": an
uninformative $q$ has no loss-side lever, only a sparsity knob, so it cannot select a
higher-loss support than vanilla at the same size. It is also why log-loss CV drives
$\hat\mu\to 0$ for uninformative $q$: with no loss margin to buy, CV sees only the
sparsity shove and declines to pay for it, recovering vanilla FasterRisk.

## Theorem 3 (probabilistic data-stability and a Nogueira floor)

Theorem 2 is deterministic: it certifies stability per resample once the realized loss
deviation $\eta$ falls below $\eta^\star=\Delta(q)/2$. We now bound $\eta$ with high
probability and convert the result into a lower bound on the Nogueira stability index
$\hat\Phi$ reported in §6.2.

**Setup.** Resamples $b=1,\dots,m$ are drawn from the data (subsample of size $n$, or
bootstrap). Binarized features lie in $\{0,1\}$ and weights in the box
$[-C,C]^{|S|}\times[-C_0,C_0]$, so every margin obeys $|w\cdot x_S+w_0|\le KC+C_0=:M$
and the per-example logistic loss lies in $[0,B]$ with $B=M+\log 2$. Write $\ell_b(S)$
for the restricted boxed minimum on resample $b$ and
$\eta_b=\sup_{|S|\le K}|\ell_b(S)-\ell(S)|$.

**Proposition.** There is a universal constant $c_0$ such that, with probability at
least $1-\delta$ over a resample,

$$
\eta_b \;\le\; \varepsilon_n(\delta) \;:=\; c_0\,B\sqrt{\frac{K\log p+\log(1/\delta)}{n}} .
$$

*Proof sketch.* The restricted minimum is $1$-Lipschitz in the sup-norm of the
objective: $|\ell_b(S)-\ell(S)|\le\sup_{w\in\mathrm{box}}|\hat R_b(w,S)-\hat R(w,S)|$,
where $\hat R$ is the average logistic loss. For a fixed $S$, the class
$\{x\mapsto\mathrm{loss}(w\cdot x_S+w_0):w\in\mathrm{box}\}$ is a $1$-Lipschitz
transform of a bounded linear class in $|S|+1\le K+1$ dimensions (the logistic loss
has derivative in $[0,1]$ w.r.t. the margin), so its Rademacher complexity is
$O(B\sqrt{K/n})$; a bounded-differences bound gives
$\sup_w|\hat R_b-\hat R|\le O(B\sqrt{K/n})+B\sqrt{\log(1/\delta')/(2n)}$ with
probability $1-\delta'$. Union over the at most $\binom{p+1}{\le K}\le p^{K}$
admissible supports (set $\delta'=\delta/p^K$, contributing
$\log(1/\delta')=K\log p+\log(1/\delta)$) gives the stated uniform bound.
$\qquad\blacksquare$

**Corollary (sample complexity for per-resample stability).** Combining the
Proposition with Theorem 2 ($\eta_b<\Delta(q)/2\Rightarrow\hat S_b(q)=\hat S(q)$): if

$$
n \;\ge\; \frac{4c_0^2\,B^2\,\bigl(K\log p+\log(1/\delta)\bigr)}{\Delta(q)^2},
$$

then $\varepsilon_n(\delta)<\Delta(q)/2$ and a resample reproduces the MAP support with
probability at least $1-\delta$.

**Corollary (Nogueira floor).** Let $\hat\rho$ be the fraction of the $m$ resamples
whose support differs from $\hat S(q)$; by the previous corollary
$\mathbb E[\hat\rho]\le\delta$. For every feature $f$, the selection frequency
$\hat p_f$ lies within $\hat\rho$ of $\{0,1\}$, so $\hat p_f(1-\hat p_f)\le\hat\rho$
and $s_f^2=\tfrac{m}{m-1}\hat p_f(1-\hat p_f)\le\tfrac{m}{m-1}\hat\rho$. With
$k=|\hat S(q)|$ and $d$ features,

$$
\mathbb E\bigl[1-\hat\Phi\bigr] \;\le\;
\frac{m}{m-1}\cdot\frac{\delta}{(k/d)(1-k/d)} .
$$

So under the sample-complexity condition the support is selected identically across
resamples with high probability and $\hat\Phi\to 1$, at a rate governed by the loss
margin $\Delta(q)$.

**The role of the prior.** $\Delta(q)$ sits in the denominator of the sample
complexity, and an informative $q$ that gives the loss-reasonable support $\hat S(q)$
a genuine $Q$-margin enlarges $\Delta(q)$ beyond the vanilla loss gap (the Theorem 2
discussion). The prior therefore lowers the $n$ needed for a target stability: this is
the theory behind the scarce-$n$ stability gain measured by the Jaccard and Nogueira
curves in §6.2, and it predicts the gain should be largest exactly where the vanilla
loss gap is smallest (the near-tie regime, where $\mathbb E[1-\hat\Phi]$ is otherwise
worst). This carries out the probabilistic upgrade left open in caveat 3.

**Caveats.** Inherits the support-level-MAP scope of Theorems 1 and 2 (the
beam-search gap and rounding, caveat 1, sit between this and the integer scorecard).
The union bound over $p^K$ supports is loose, so this is a rate statement
($n=\tilde O(K/\Delta(q)^2)$), not a sharp prediction of $\hat\Phi$; the realized
stability is better. Constants are not optimized, and the without-replacement
subsample case uses the same bound up to the usual sampling constant.

## Large-sample behaviour: risk convergence and selection consistency

Theorems 1 to 3 are finite-sample and fix the data or compare resamples of it. The
same uniform-convergence tool, run against the population risk rather than the
full-data empirical risk, gives the large-sample picture. Let

$$
\ell_\infty(S) \;=\; \min_{\substack{w\in\mathrm{box}\\ \mathrm{supp}(w)\subseteq S}}
\mathbb E_{(x,y)}\bigl[\mathrm{loss}(w;x,y)\bigr]
$$

be the population restricted risk, $F_\infty(S)=\ell_\infty(S)-\mu Q(S)$ the penalized
population objective, $S_{\mathrm{pop}}(q)=\arg\min_{|S|\le K}F_\infty(S)$ its
minimizer, and $\Delta_\infty(q)=\min_{S\ne S_{\mathrm{pop}}}[F_\infty(S)-F_\infty(S_{\mathrm{pop}})]$
the population margin.

**Proposition (uniform risk convergence).** The Theorem 3 Proposition holds verbatim
with the population measure as the base: with probability at least $1-\delta$,

$$
\sup_{|S|\le K}\bigl|\ell_n(S)-\ell_\infty(S)\bigr| \;\le\;
\varepsilon_n(\delta)=c_0\,B\sqrt{\frac{K\log p+\log(1/\delta)}{n}} .
$$

Same $1$-Lipschitz-min, boxed-logistic Rademacher, and union-over-$p^K$-supports
argument; only the reference measure changes.

**Corollary (risk convergence, no margin needed).** Because $\hat S_n(q)$ minimizes
the empirical $F_n$ and $F_n$ is within $\varepsilon_n$ of $F_\infty$ uniformly, the
standard ERM excess-risk inequality gives

$$
F_\infty\bigl(\hat S_n(q)\bigr)-F_\infty\bigl(S_{\mathrm{pop}}(q)\bigr)
\;\le\; 2\,\varepsilon_n(\delta)=O\!\Bigl(B\sqrt{\tfrac{K\log p}{n}}\Bigr).
$$

The selected scorecard converges to the penalized population optimum in objective
value at the $\sqrt{K\log p/n}$ rate, with no identifiability or margin assumption. In
particular this convergence survives the near-tie regime where $\Delta_\infty(q)\to 0$
and selection consistency (below) fails: at a near-tie the competing supports have
nearly equal $F_\infty$ by definition, so picking the "wrong" one of two near-tied
supports costs $O(\Delta_\infty)\to 0$ in objective. The support may flicker; the risk
does not. This is the assumption-light guarantee, and the operative one at near-ties.

**Do-no-harm in risk units.** In the safe regime $\mu\le\mu_0$ the population minimizer
is the unpenalized $K$-sparse risk optimum $S_{\mathrm{loss}}^\infty$ (Lemma 1,
population version), so the prior steers to vanilla's target and the bound above is
pure risk convergence to the best boxed $K$-sparse model. Above $\mu_0$ the target
shifts by the deliberate prior bias: since $S_{\mathrm{pop}}$ minimizes
$\ell_\infty-\mu Q$,

$$
0\;\le\;\ell_\infty(S_{\mathrm{pop}})-\ell_\infty(S_{\mathrm{loss}}^\infty)
\;\le\;\mu\bigl(Q(S_{\mathrm{pop}})-Q(S_{\mathrm{loss}}^\infty)\bigr),
$$

the price paid to buy the $Q$-margin that Theorem 3 converts into stability. It
vanishes as $\mu\downarrow\mu_0$, so the safe regime gets the stability gain at no
first-order risk cost.

**Theorem 4 (selection consistency).** Assume population identifiability
$\Delta_\infty(q)>0$. If

$$
n \;\ge\; \frac{4c_0^2\,B^2\bigl(K\log p+\log(1/\delta)\bigr)}{\Delta_\infty(q)^2},
$$

then $\varepsilon_n(\delta)<\Delta_\infty(q)/2$ and the margin argument of Theorem 2
gives $\hat S_n(q)=S_{\mathrm{pop}}(q)$ with probability at least $1-\delta$. Taking
$\delta=\delta_n$ summable (e.g. $\delta_n=n^{-2}$) and Borel-Cantelli,
$\hat S_n(q)=S_{\mathrm{pop}}(q)$ eventually almost surely: the prior-MAP support is
selection-consistent for $S_{\mathrm{pop}}(q)$.

**Estimated $q$.** If the discovery procedure returns $q_n$ with
$\lVert q_n-q_\infty\rVert_\infty\to_p 0$, Theorem 1 absorbs the prior perturbation
once $\lVert q_n-q_\infty\rVert_\infty<\varepsilon^\star$ (the population prior radius),
so on the intersection of the two high-probability events
$\hat S_n(q_n)=S_{\mathrm{pop}}(q_\infty)$ eventually. The clause is honest about the
dependency: it inherits whatever conditions the $q$-source needs for consistency
(faithfulness for PC/GES; the stability-selection conditions for iamb\_soft).

**Consistent to what.** $S_{\mathrm{pop}}(q)$ is the minimizer of
population-risk-minus-prior over the boxed $K$-sparse class, the estimand the method
defines, not by itself the causal or data-generating support. The two coincide when
the prior-shifted population optimum equals the truth, which holds for a well-specified
model with an aligned (e.g. oracle causal) $q$ and can fail under a biased $q$ or
misspecification. So Theorem 4 is consistency for the method's estimand; identifying
that estimand with the causal truth is the separate question that §6.1 (mechanism) and
the transport story (ICP) address. The separation is deliberate: the optimizer is
consistent for its estimand, and the causal content lives in $q$'s identification, not
in the estimator.

**Parameter convergence.** Conditional on the selection event
$\hat S_n(q)=S_{\mathrm{pop}}(q)$ (Theorem 4), the within-support fit is a smooth,
strongly convex (with the ridge term) M-estimation problem on a fixed low-dimensional
support, so $\hat w_n\to w_{\mathrm{pop}}$ at the usual $\sqrt n$ rate. Unconditionally
it inherits Theorem 4's margin assumption.

**Scope.** As with Theorems 1 to 3 these are support-level-MAP statements (the
beam-search gap and rounding, caveat 1, sit between them and the integer scorecard),
the union bound over $p^K$ supports is loose (rates, not sharp constants), and
$\Delta_\infty(q)>0$ is the population analogue of the uniqueness assumption, failing
at population near-ties, where the risk Corollary, not Theorem 4, is the operative
guarantee.

## Why this matters: the two empirical phenomena are one bound

The causal prior reparametrizes the source of support variance from the data toward
$q$; the ratio $\varepsilon^\star/\eta^\star = 1/\mu$ (single-swap competitor) is the
exchange rate.

- **Adversarial collapse at large $\mu$** follows directly: $\varepsilon^\star$ carries
  an explicit $1/\mu$, and $\Delta(q)\to 0$ at every support transition, so a wrong or
  perturbed $q$ flips the MAP support ever more easily as $\mu$ grows. The degradation
  is monotone-in-tendency, not graceful, matching the adversarial source in §6.1.

- **Stability gain (conditional, not a $\mu$-law).** An informative $q$ that gives a
  loss-reasonable support $S^\star$ a real $Q$-margin enlarges $\Delta(q)$ beyond the
  vanilla loss-gap, raising $\eta^\star$ and pinning the support against fold-to-fold
  loss perturbations (the synthetic CV Jaccard gain; the TB $+0.10$). This holds when
  $q$ is informative and $\mu$ sits just above $\mu_0$, which is also where log-loss
  CV tends to place $\hat\mu$, and why it collapses $\hat\mu\to 0$ for uninformative $q$.

The safe regime is $\mu$ just above $\mu_0$: enough margin for data-stability, before
the $1/\mu$ prior-fragility and the transition dips dominate.

## Numerical validation (exact MAP)

`experiments/causal_prior/synthetic/exact_radii.py` validates Theorem 1 by brute
force: on a small cell ($p=12$, $n=200$, $k^\star=3$, $K=3$, $p_{\mathrm{edge}}=0.5$)
it enumerates all 298 supports with $|S|\le K$, fits the restricted
(L2-regularized) logistic loss $\ell(S)$ once each, and computes the exact MAP, the
gap $\Delta(q)$, the easy bound $\varepsilon_{\mathrm{easy}} = \Delta/(2\mu K)$, and
the tight radius $\varepsilon^\star = \min_{S_2} G_q(S_2)/(\mu\,|S^\star \triangle S_2|)$
(the code's `eps_adv` column).

Result (oracle $q$; the runner-up shares 2 of 3 true features, so $b=1$, $K=3$,
$|S\triangle S^\star|=2$):

- $\varepsilon_{\mathrm{easy}} \le \varepsilon^\star$ at every $\mu$: the easy
  bound is a valid but loose lower bound; the tight $\varepsilon^\star$ is the exact
  radius (the MAP provably cannot flip below it, and an adversary flips it exactly
  at it).
- $\varepsilon^\star/\varepsilon_{\mathrm{easy}} = 2K/|S\triangle S^\star| = 3 = K$
  here (since $|S\triangle S^\star|=2$): the slack is the doubling-plus-cardinality
  of the easy bound, not a property of the MAP. The true radius is $\Delta/(2\mu)$
  while the easy bound is $\Delta/(2\mu K)$.
- Asymptotics match: $\varepsilon_{\mathrm{easy}} \to b/(2K) = 1/6$ and
  $\varepsilon^\star \to b/2 = 1/2$ as $\mu$ grows ($\Delta = a + \mu b$); and
  $\varepsilon^\star/\eta^\star \to 1/\mu$, confirming the cardinality-free exchange
  rate.

**Theorem 2 (data radius)** is validated on the same cell by bootstrap (20 resamples,
recomputing $\ell(S)$ for every support; $\eta = \sup_S |\ell_b(S) - \ell(S)|$):

- $\mu=0$ (vanilla): $r_\ell = \Delta/2 \approx 0.01$ (the loss optimum is a near-tie);
  the bootstrap MAP is stable in only $65\%$ of resamples and recovers $S^\star$ in $0\%$.
- $\mu>0$ (prior on): $r_\ell$ grows with $\mu$ ($8.8 \to 896$ as $\mu_{\mathrm{rel}}$
  goes $0.05 \to 5$); the bootstrap MAP is stable in $100\%$ and recovers $S^\star$ in $100\%$.
- $\mathrm{viol} = 0$ at every $\mu$: no bootstrap with $\eta < r_\ell$ ever changed the
  MAP, so the guarantee holds.

This reproduces the data-stability gain ($65\% \to 100\%$; $S^\star$-recovery
$0\% \to 100\%$) in the exact setting (the same phenomenon as the $+0.10$ TB support
stability and the §6.1 CV-stability panels), with $r_\ell = \Delta/2$ as the explicit
mechanism.

Not exhibited by this cell: a support *transition* (the loss already recovers
$S^\star$, so the MAP $= S^\star$ for all $\mu>0$ and $\Delta$ grows monotonically); a
cell with a loss-optimal support $\ne S^\star$ would show $\Delta \to 0$ at the
crossing. A FasterRisk-based probe (`two_radii.py`) does **not** cleanly exhibit
either bound: the beam-search heuristic departs from the exact MAP (caveat 1), which
is why exact enumeration is the appropriate test.

## Caveats / to verify

1. Support-level MAP with a *continuous* within-support fit only. Two distinct
   approximations sit between it and the deployed integer scorecard, and they belong
   to different sections. **(i) Beam-search support gap (this caveat).** SparseBeamLR
   may not select the exact-MAP support, so the beam heuristic can violate both
   radii; quantifying it is the §5 Rashomon-pool target. The matched oracle is the
   exact *continuous* per-support optimum, by brute-force enumeration of $\ell(S)$:
   `exact_radii.py` at tiny $p$, and `beam_gap.py` at the §6.1 anchor. With a compact
   Newton fit ($\approx 0.55$ ms/support) the $p=30$, $K=2k^\star=10$ cell
   ($\approx 5.3\times10^7$ supports) runs in about 10 min/seed on 48 cores, so the
   beam gap is directly measurable at the headline regime; $p=211$ (TB) stays out of
   reach. There is no off-the-shelf certified continuous per-support solver for
   logistic loss (OKRidge is ridge/squared-loss only), so brute force on many cores
   is the instrument. **(ii) Rounding.** Turning the continuously-selected
   support into integer coefficients is governed by §2.4 support-preservation
   (preserved in the safe regime; the extreme-$\mu$ failure is the §8 low-$|w_j|$
   pathology), not by this caveat. A certified-*integer* solver (RiskSlim) would
   answer (ii)'s end-to-end "FasterRisk vs integer MAP" question on a smaller,
   no-multiplier class. It is **not** the oracle for the beam-search gap (i), whose
   object is the continuous $\ell(S)$.
2. Uniqueness of $\hat S(q)$ (i.e. $\Delta(q) > 0$) assumed; degenerate ties need
   a tie-break and a separate argument. Note $\Delta(q) \to 0$ at support
   transitions is exactly where this assumption is tightest.
2b. The only $\mu$-monotone object is the ratio $\varepsilon^\star/\eta^\star = 1/\mu$
   (single-swap competitor); the individual radii inherit the non-monotone
   $\Delta(q)$. Do not state a global "$r_\ell\uparrow$, $r_q\downarrow$" trend, and
   do not claim the cardinality factor $K$ in the exchange rate; it is slack from
   the easy bound and does not survive the tight Theorem 1.
3. $r_\ell$ uses a uniform bound $\eta$ over all supports; a subsample-specific
   high-probability bound on $\eta$ (Meinshausen-Bühlmann style) turns Theorem 2
   into a probabilistic stability statement. Done in Theorem 3 (probabilistic
   data-stability and a Nogueira floor) above.
4. The $\mu_0$ characterization is now in closed form: Lemma 1 (closed form for the
   separation threshold $\mu_0$) above, with the do-no-harm corollary for
   uninformative $q$.
