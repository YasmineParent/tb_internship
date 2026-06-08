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
   exact *continuous* per-support optimum: brute-force enumeration of $\ell(S)$ at
   small $p$ (`exact_radii.py`, capped at $p\approx12$, $\binom{p}{\le K}$ supports),
   and a certified continuous per-support solver (the OKRidge-GLM class) at moderate
   $p$ where enumeration is dead. **(ii) Rounding.** Turning the continuously-selected
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
   high-probability bound on $\eta$ (Meinshausen-Bühlmann style) would turn
   Theorem 2 into a probabilistic stability statement and is the natural next step.
4. The $\mu_0$ characterization is stated, not yet a closed form; deriving
   $\mu_0$ in terms of $a$ and the $q$-mass advantage is a clean small lemma.
