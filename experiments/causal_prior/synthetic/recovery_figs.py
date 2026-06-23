"""recovery figures from the full cv sweep (recovery_cv, 20 seeds).

regenerates the headline support-recovery panels now that the sweep includes
iamb (constraint-local mb), the strong source vanilla pc is not. drops oracle
(non-operational ceiling), pc (near-noise under fisher-z on dense gaussian
dags), and adversarial (a worst-case control that lives in the q-robustness
anchor panel, not the across-regime story), leaving four lines: vanilla floor,
iamb (deployed mb source), ges (global causal), bootstrap_l1 (predictive).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.loading import load_recovery_csvs   # noqa: E402
from src.causal_prior import visualization as viz         # noqa: E402

CV_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'recovery_cv'
OUT = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'recovery_figs'

AXES = [('n', 'n'), ('p', 'p'), ('p_edge', 'p_edge'),
        ('k_star', 'k_star'), ('noise_scale', 'noise_scale')]
SLICES = {
    'n': 'p == 30 and k_star == 5 and p_edge == 0.2',
    'p': 'n == 300 and k_star == 5 and p_edge == 0.2',
    'p_edge': 'p == 30 and n == 300 and k_star == 5',
    'k_star': 'p == 30 and n == 300 and p_edge == 0.2',
    'noise_scale': 'p == 30 and n == 300 and k_star == 5 and p_edge == 0.2',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv = load_recovery_csvs(CV_DIR)
    cv = cv[~cv['q_source'].str.startswith('pc')]
    cv = cv[~cv['q_source'].isin(['oracle', 'adversarial'])]
    nz = cv['noise_scale'] == 1.0

    def sl(axis):
        d = cv if axis == 'noise_scale' else cv[nz]
        return d.query(SLICES[axis])

    # fig 1: S_precision across every axis
    fig, axs = plt.subplots(1, len(AXES), figsize=(4 * len(AXES), 4))
    for ax, (col, _) in zip(axs, AXES):
        viz.plot_recovery_cv_vs_axis(sl(col), axis_col=col,
                                     metric='S_precision', ax=ax)
        if ax is not axs[0]:
            ax.set_ylabel('')
    fig.suptitle('Support recovery (S_precision) at CV-picked $\\hat\\mu$, 20 seeds',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'recovery_S_precision_all_axes.png', dpi=130,
                bbox_inches='tight')

    # fig 2: three metrics vs p_edge
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axs2, ['S_precision', 'correlate_inclusion',
                                 'causal_precision']):
        viz.plot_recovery_cv_vs_axis(sl('p_edge'), axis_col='p_edge',
                                     metric=metric, ax=ax)
    fig2.suptitle('Density sweep: recovery, confounder avoidance, any-cause precision',
                  fontsize=12)
    fig2.tight_layout()
    fig2.savefig(OUT / 'recovery_metrics_vs_p_edge.png', dpi=130,
                 bbox_inches='tight')

    # fig 3: mu_star vs p_edge diagnostic
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    viz.plot_mu_star_vs_axis(sl('p_edge'), axis_col='p_edge', ax=ax3)
    fig3.savefig(OUT / 'mu_star_vs_p_edge.png', dpi=130, bbox_inches='tight')

    print('sources plotted:', sorted(cv['q_source'].unique()))
    print(f'Done. Figures in {OUT}/')


if __name__ == '__main__':
    main()
