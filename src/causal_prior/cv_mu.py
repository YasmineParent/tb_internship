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


@dataclass
class CVResult:
    mu_star: float                # mu value selected by CV (argmin of log_loss)
    auc_star: float               # mean held-out AUC at mu_star
    log_loss_star: float          # mean held-out log loss at mu_star (the selection criterion)
    aucs_per_mu: np.ndarray       # mean held-out AUC over the mu grid (shape: (n_mu,))
    log_losses_per_mu: np.ndarray # mean held-out log loss over the mu grid (shape: (n_mu,))
    support: list[int]            # nonzero coefficient indices after refit at mu_star
    betas: np.ndarray             # integer beta vector after refit at mu_star (shape: (p,))


def _import_fasterrisk():
    """Defer FR import; the R-binding warning fires on every fresh worker."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
    return FasterRisk


def cv_pick_mu(X: np.ndarray, y: np.ndarray, K: int,
               mu_grid: np.ndarray, q: np.ndarray | None,
               n_splits: int = 5,
               rng: np.random.Generator | None = None) -> CVResult:
    """Pick mu by k-fold CV minimising held-out logistic loss, then refit.

    y is FasterRisk's signed convention: {-1, +1}. sklearn's log_loss /
    roc_auc_score want 0/1, so we map y -> (y > 0). StratifiedKFold preserves
    class balance per fold; the rng controls the shuffle seed for
    reproducibility. Both log_loss and AUC are computed per (mu, fold) and
    averaged across folds; mu_star = argmin mean log_loss. AUC at mu_star is
    reported alongside for context but does NOT drive selection.
    """
    FasterRisk = _import_fasterrisk()
    if rng is None:
        rng = np.random.default_rng()

    y_true_binary = (y > 0).astype(int)
    fold_seed = int(rng.integers(0, 2**31 - 1))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)

    aucs = np.full((len(mu_grid), n_splits), np.nan)
    losses = np.full((len(mu_grid), n_splits), np.nan)
    for s_idx, (tr, te) in enumerate(skf.split(X, y_true_binary)):
        single_class = y_true_binary[te].sum() in (0, len(te))
        for m_idx, mu in enumerate(mu_grid):
            fr = FasterRisk(k=K, mu=float(mu),
                            freq=q.astype(float) if q is not None else None)
            fr.fit(X[tr], y[tr])
            y_prob = np.clip(fr.predict_proba(X[te]), 1e-7, 1 - 1e-7)
            losses[m_idx, s_idx] = log_loss(y_true_binary[te], y_prob,
                                            labels=[0, 1])
            if not single_class:
                try:
                    aucs[m_idx, s_idx] = roc_auc_score(y_true_binary[te], y_prob)
                except ValueError:
                    pass

    mean_losses = np.nanmean(losses, axis=1)
    mean_aucs = np.nanmean(aucs, axis=1)
    mu_star_idx = int(np.nanargmin(mean_losses))
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
        aucs_per_mu=mean_aucs,
        log_losses_per_mu=mean_losses,
        support=support, betas=np.asarray(betas, dtype=int),
    )
