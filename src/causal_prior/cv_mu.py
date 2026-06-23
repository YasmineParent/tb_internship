"""CV-on-mu: pick the prior strength by held-out predictive performance.

For each mu in the grid, run k-fold CV with FasterRisk(k=K, mu, freq=q) and
score predictions on the held-out fold. Pick mu_star = argmin of mean
held-out logistic loss (primary criterion) and report mean held-out AUC at
mu_star for reference. Then refit on full data at mu_star and return the
support and integer betas; recovery metrics against ground truth S* are
computed downstream.

Why logistic loss is the primary criterion (not AUC):
- AUC is essentially invariant to mu in this regime (the prior changes
  which features are selected, not how the resulting model ranks samples).
  Empirically AUC is flat to ~1e-3 across the mu grid, so CV-on-AUC
  collapses to "pick vanilla".
- Logistic loss penalises miscalibration as well as ranking, so it sees
  the prior's effect on the integer betas. CV-on-loss reliably picks
  a small-but-nonzero mu (the "elbow" where the prior starts mattering).

The held-out criterion is *predictive*, but the metric reported by the
caller is *support recovery* (S_recall, S_precision, C_inclusion). This
mirrors the practitioner setup: CV picks mu using only y, the analyst
then asks whether the resulting support matches the true causes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .scorecard import import_fasterrisk


@dataclass
class CVResult:
    mu_star: float                # mu value selected by CV
    auc_star: float               # mean held-out AUC at mu_star
    log_loss_star: float          # mean held-out log loss at mu_star
    stability_star: float         # mean pairwise Jaccard at mu_star
    aucs_per_mu: np.ndarray       # mean held-out AUC over the mu grid (shape: (n_mu,))
    log_losses_per_mu: np.ndarray # mean held-out log loss over the mu grid (shape: (n_mu,))
    stabilities_per_mu: np.ndarray # mean pairwise Jaccard over the mu grid (shape: (n_mu,))
    support: list[int]            # nonzero coefficient indices after refit at mu_star
    betas: np.ndarray             # integer beta vector after refit at mu_star (shape: (p,))


def _mean_pairwise_jaccard(supports: list[list[int]]) -> float:
    """Average Jaccard similarity across all fold pairs; supports with no
    overlap and no union (both empty) treated as Jaccard = 1.0. With <2
    folds present, returns 0.0.
    """
    if len(supports) < 2:
        return 0.0
    sets = [set(s) for s in supports]
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                pairs.append(1.0)
                continue
            pairs.append(len(sets[i] & sets[j]) / len(union))
    return float(np.mean(pairs)) if pairs else 0.0


def make_mu_grid(X: np.ndarray, y: np.ndarray, n_mu: int = 8) -> tuple[float, np.ndarray]:
    """Return (mu_scale, mu_grid) where mu_scale = median(0.5*|X^T y|) and
    mu_grid = [0] + logspace(-2, 1, n_mu) * mu_scale."""
    mu_scale = float(np.median(0.5 * np.abs(X.T @ y)))
    mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, n_mu)]) * mu_scale
    return mu_scale, mu_grid


def cv_pick_mu(X: np.ndarray, y: np.ndarray, K: int,
               mu_grid: np.ndarray, q: np.ndarray | None,
               n_splits: int = 5,
               criterion: str = 'log_loss',
               rng: np.random.Generator | None = None) -> CVResult:
    """Pick mu by k-fold CV under the chosen criterion, then refit on full data.

    criterion = 'log_loss' (default): mu_star = argmin mean held-out log loss.
    criterion = 'stability':          mu_star = argmax mean pairwise Jaccard
                                      of per-fold supports.

    log_loss is the standard practitioner criterion (predictive performance
    on held-out data). stability rewards mu values where the prior's choice
    of support is robust to data perturbation, which can favor causally
    consistent over spurious-but-predictive features.

    All three diagnostics (log_loss, AUC, stability per mu) are always
    computed and recorded; only the selection criterion differs. AUC is
    typically mu-flat for FR's integer betas and is reported for reference.
    """
    if criterion not in ('log_loss', 'stability'):
        raise ValueError(f"criterion must be 'log_loss' or 'stability', got {criterion!r}")
    FasterRisk = import_fasterrisk()
    if rng is None:
        rng = np.random.default_rng()

    y_true_binary = (y > 0).astype(int)
    fold_seed = int(rng.integers(0, 2**31 - 1))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)

    aucs = np.full((len(mu_grid), n_splits), np.nan)
    losses = np.full((len(mu_grid), n_splits), np.nan)
    supports_per_mu: list[list[list[int]]] = [[] for _ in range(len(mu_grid))]
    for s_idx, (tr, te) in enumerate(skf.split(X, y_true_binary)):
        single_class = y_true_binary[te].sum() in (0, len(te))
        for m_idx, mu in enumerate(mu_grid):
            fr = FasterRisk(k=K, mu=float(mu),
                            freq=q.astype(float) if q is not None else None)
            fr.fit(X[tr], y[tr])
            betas_fold = fr.betas_[0]
            supports_per_mu[m_idx].append(
                sorted(int(j) for j in np.where(np.abs(betas_fold) > 0)[0])
            )
            y_prob = np.clip(fr.predict_proba(X[te]), 1e-7, 1 - 1e-7)
            losses[m_idx, s_idx] = log_loss(y_true_binary[te], y_prob,
                                            labels=[0, 1])
            if not single_class:
                try:
                    aucs[m_idx, s_idx] = roc_auc_score(y_true_binary[te], y_prob)
                except ValueError:
                    pass

    mean_losses = np.nanmean(losses, axis=1)
    with warnings.catch_warnings():  # a fully single-class mu column is all-nan; mean is nan, not an error
        warnings.simplefilter('ignore', RuntimeWarning)
        mean_aucs = np.nanmean(aucs, axis=1)
    stabilities = np.array([_mean_pairwise_jaccard(s) for s in supports_per_mu])

    if criterion == 'log_loss':
        mu_star_idx = int(np.nanargmin(mean_losses))
    else:
        mu_star_idx = int(np.nanargmax(stabilities))
    mu_star = float(mu_grid[mu_star_idx])

    # refit on full data at mu_star
    fr_final = FasterRisk(k=K, mu=mu_star,
                          freq=q.astype(float) if q is not None else None)
    fr_final.fit(X, y)
    betas = fr_final.betas_[0]
    support = sorted(int(j) for j in np.where(np.abs(betas) > 0)[0])

    return CVResult(
        mu_star=mu_star,
        auc_star=float(mean_aucs[mu_star_idx]),
        log_loss_star=float(mean_losses[mu_star_idx]),
        stability_star=float(stabilities[mu_star_idx]),
        aucs_per_mu=mean_aucs,
        log_losses_per_mu=mean_losses,
        stabilities_per_mu=stabilities,
        support=support, betas=np.asarray(betas, dtype=int),
    )
