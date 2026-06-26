"""c3: markov-blanket recovery f1 by discovery source."""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.causal_prior import figstyle
from src.causal_prior.figures import figure
from src.causal_prior.loading import mb_recovery_table

CACHE = figstyle.ROOT / 'results/causal_prior/synthetic/cache_p30_headline'
ORDER = ['iamb', 'ges', 'pc', 'bootstrap_l1']


@figure('c3_mb_recovery', kind='appendix')
def c3_mb_recovery():
    t = mb_recovery_table(CACHE, threshold=0.5)
    fig, ax = plt.subplots(figsize=figstyle.PANEL)
    for i, src in enumerate(ORDER):
        label, colour, _ = figstyle.SOURCES[src]
        f1 = t.loc[src, 'f1']
        ax.bar(i, f1, color=colour)
        ax.text(i, f1 + 0.02, f'{f1:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([figstyle.SOURCES[s][0].split(' (')[0] for s in ORDER])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(r'MB recovery F1 ($q\geq0.5$ vs $S^\star$)')
    return fig
