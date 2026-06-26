"""c4: k-sweep grid — test auc vs model size k, one panel per benchmark."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.causal_prior import figstyle
from src.causal_prior.figures import figure
from src.causal_prior.cfs_results import load_ksweep, ARM_ORDER, ARM_LABEL

NCOL = 3


@figure('c4_ksweep', kind='appendix')
def c4_ksweep():
    ksweep = load_ksweep()
    ds = list(ksweep)
    nrow = int(np.ceil(len(ds) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(figstyle.TEXT_W, 2.2 * nrow))
    axf = axes.ravel()
    for i, (ax, d) in enumerate(zip(axf, ds)):
        df = ksweep[d]
        n, p = int(df['n'].iloc[0]), int(df['p'].iloc[0])
        for a in ARM_ORDER:
            g = df[df['arm'] == a].groupby('k')['auc']
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                        marker='o', ms=3, lw=1, capsize=2, color=figstyle.ARM_COLOR[a])
        ax.set_title(f'{d} (n={n}, p={p})')
        ax.set_xlabel('model size $k$')
        if i % NCOL == 0:
            ax.set_ylabel('test AUC')
    for ax in axf[len(ds):]:
        ax.axis('off')
    handles = [Line2D([], [], color=figstyle.ARM_COLOR[a], marker='o', ms=3, label=ARM_LABEL[a])
               for a in ARM_ORDER]
    fig.legend(handles=handles, loc='outside lower center', ncol=4)
    return fig
