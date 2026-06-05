# Support-stability of the causal-prior MAP

Draft of the §5 robustness target. Rigorous for the **exact combinatorial MAP**
(the idealized objective FasterRisk approximates by beam search); the beam-search
gap is a separate, acknowledged approximation. To be verified before use.

## Setup

Data fixed. For a support $S \subseteq [p]$ with $|S| \le K$, let

$$
\ell(S) \;=\; \min_{\substack{w:\ \mathrm{supp}(w)\subseteq S \\ w \in [-C,C]^p,\ w_0}} L(w, w_0, \mathcal{D})
$$

be the restricted logistic-loss minimum on support $S$ (well-defined: continuous
loss over a compact box). It does **not** depend on the prior. With prior
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

Let $q'$ satisfy $\lVert q - q' \rVert_\infty \le \varepsilon$. If

$$
\varepsilon \;<\; \frac{\Delta(q)}{2\mu K},
$$

then $\hat S(q') = \hat S(q)$: the MAP support is unchanged.

**Proof.** For every $S$ with $|S| \le K$,

$$
\big| F_{q'}(S) - F_q(S) \big| \;=\; \mu\Big| \textstyle\sum_{j\in S}(q'_j - q_j) \Big|
\;\le\; \mu \sum_{j\in S} |q'_j - q_j| \;\le\; \mu |S|\,\varepsilon \;\le\; \mu K \varepsilon .
$$

Write $S^\star = \hat S(q)$. For any $S \ne S^\star$,

$$
F_{q'}(S) - F_{q'}(S^\star)
= \underbrace{[F_q(S) - F_q(S^\star)]}_{\ge\, \Delta(q)}
+ \underbrace{[F_{q'}(S)-F_q(S)]}_{\ge -\mu K\varepsilon}
- \underbrace{[F_{q'}(S^\star)-F_q(S^\star)]}_{\le +\mu K\varepsilon}
\;\ge\; \Delta(q) - 2\mu K \varepsilon \;>\; 0 .
$$

So $S^\star$ strictly minimizes $F_{q'}$. $\qquad\blacksquare$

The **prior-invariance radius** is therefore

$$
r_q(\mu) \;=\; \frac{\Delta(q)}{2\mu K}.
$$

## Theorem 2 (data perturbation: turning up $\mu$ buys data-robustness)

Suppose a data perturbation (subsample / CV fold) changes the restricted losses to
$\ell'$ with $\sup_{|S|\le K} |\ell'(S) - \ell(S)| \le \eta$, and leaves $q$ fixed.
If $\eta < \tfrac{1}{2}\Delta(q)$, then the MAP support is unchanged.

**Proof.** Identical, with $|F'_q(S) - F_q(S)| = |\ell'(S)-\ell(S)| \le \eta$ and
the same margin argument gives $\Delta(q) - 2\eta > 0$. $\qquad\blacksquare$

The **data-invariance radius** is $r_\ell(\mu) = \tfrac{1}{2}\Delta(q)$.

## What is and isn't monotone in $\mu$

Write the two tolerances as $\varepsilon^\star(\mu) = \tfrac{\Delta(q)}{2\mu K}$ (prior,
Thm 1) and $\eta^\star(\mu) = \tfrac{\Delta(q)}{2}$ (data, Thm 2). Their ratio is exact
and cancels $\Delta$:

$$
\frac{\varepsilon^\star(\mu)}{\eta^\star(\mu)} \;=\; \frac{1}{\mu K}.
$$

This is the **one $\mu$-monotone fact**, true regardless of the landscape: per unit
margin, increasing $\mu$ makes the support proportionally more fragile to prior
error than to data error, by exactly $\mu K$. It is the precise form of "the prior
trades data-variance for $q$-variance."

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
$\hat S(q) \ne S_{\mathrm{loss}}$ — the "separation threshold" of §5. Above $\mu_0$
the explicit $1/\mu$ in $\varepsilon^\star$ takes over.

## Why this matters: the two empirical phenomena are one bound

The causal prior reparametrizes the source of support variance from the data toward
$q$; the ratio $\varepsilon^\star/\eta^\star = 1/(\mu K)$ is the exact exchange rate.

- **Adversarial collapse at large $\mu$** follows directly: $\varepsilon^\star$ carries
  an explicit $1/\mu$, and $\Delta(q)\to 0$ at every support transition, so a wrong or
  perturbed $q$ flips the MAP support ever more easily as $\mu$ grows. The degradation
  is monotone-in-tendency, not graceful — matching the adversarial source in §6.1.

- **Stability gain (conditional, not a $\mu$-law).** An informative $q$ that gives a
  loss-reasonable support $S^\star$ a real $Q$-margin enlarges $\Delta(q)$ beyond the
  vanilla loss-gap, raising $\eta^\star$ and pinning the support against fold-to-fold
  loss perturbations (the synthetic CV Jaccard gain; the TB $+0.10$). This holds when
  $q$ is informative and $\mu$ sits just above $\mu_0$ — which is also where log-loss
  CV tends to place $\hat\mu$, and why it collapses $\hat\mu\to 0$ for uninformative $q$.

The safe regime is $\mu$ just above $\mu_0$: enough margin for data-stability, before
the $1/\mu$ prior-fragility and the transition dips dominate.

## Caveats / to verify

1. Exact combinatorial MAP only; the beam-search heuristic can violate both
   radii. Quantifying that gap is open (it is the §5 Rashomon-pool target).
2. Uniqueness of $\hat S(q)$ (i.e. $\Delta(q) > 0$) assumed; degenerate ties need
   a tie-break and a separate argument. Note $\Delta(q) \to 0$ at support
   transitions is exactly where this assumption is tightest.
2b. The only $\mu$-monotone object is the ratio $\varepsilon^\star/\eta^\star = 1/(\mu K)$;
   the individual radii inherit the non-monotone $\Delta(q)$. Do not state a global
   "$r_\ell\uparrow$, $r_q\downarrow$" trend.
3. $r_\ell$ uses a uniform bound $\eta$ over all supports; a subsample-specific
   high-probability bound on $\eta$ (Meinshausen-Bühlmann style) would turn
   Theorem 2 into a probabilistic stability statement and is the natural next step.
4. The $\mu_0$ characterization is stated, not yet a closed form; deriving
   $\mu_0$ in terms of $a$ and the $q$-mass advantage is a clean small lemma.
