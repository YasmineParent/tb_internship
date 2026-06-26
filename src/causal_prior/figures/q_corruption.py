"""q_corruption: graceful degradation — recovery slides to the floor, auc stays flat."""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.causal_prior import figstyle
from src.causal_prior.figures import figure, latest_run

DIR = figstyle.ROOT / 'results/causal_prior/synthetic/q_corruption'


@figure('q_corruption', kind='appendix')
def q_corruption():
    g = pd.read_csv(latest_run(DIR, 'headline', needs='summary.csv') / 'summary.csv')
    fig, ax1 = plt.subplots(figsize=figstyle.PANEL)
    ax1.errorbar(g['corruption'], g['S_precision'], yerr=g['S_prec_sem'], marker='o', ms=3, color='C0')
    ax1.set_xlabel('q corruption (0 = discovered, 1 = noise)')
    ax1.set_ylabel(r'support recovery $S_\mathrm{prec}$', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.errorbar(g['corruption'], g['auc'], yerr=g['auc_sem'], marker='s', ms=3, color='C3')
    ax2.set_ylabel('held-out AUC', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax2.set_ylim(0.5, 1.0)
    return fig
