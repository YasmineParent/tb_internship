"""fig1: scorecards (a, latex fasterrisk-style) and rashomon pool cloud (b)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.causal_prior import figstyle
from src.causal_prior.figures import figure

CARDS = figstyle.ROOT / 'results/causal_prior/real/scorecard_diabetes130/cards.json'
POOL = figstyle.ROOT / 'results/causal_prior/rashomon/heart_k10/pool.csv'

# human-readable labels for the binarized diabetes130 feature names (render-only).
LABELS = {
    'number_inpatient_le_0': 'no prior inpatient stays',
    'number_inpatient_le_1': r'$\leq$1 prior inpatient stays',
    'number_emergency_le_0': 'no prior ER visits',
    'discharge_disposition_id_le_1': 'discharged home (vs facility)',
    'discharge_disposition_id_le_2': 'discharged home or self-care',
    'metformin=No': 'not on metformin',
    'metformin=Up': 'metformin dose increased',
    'insulin=No': 'not on insulin',
    'insulin=Down': 'insulin dose decreased',
    'A1Cresult=Norm': 'HbA1c normal',
    'A1Cresult=>7': 'HbA1c $>$7%',
    'payer_code=DM': 'payer: Medicare/Medicaid',
    'payer_code=HM': 'payer: HMO',
    'payer_code=MC': 'payer: Medicaid',
    'age=[30-40)': 'age 30-40',
    'age=[40-50)': 'age 40-50',
    'age=[50-60)': 'age 50-60',
    'age=[60-70)': 'age 60-70',
    'age=[70-80)': 'age 70-80',
    'age=[80-90)': 'age 80-90',
}


def _esc(s: str) -> str:
    for a, b in (('%', r'\%'), ('&', r'\&'), ('#', r'\#'), ('_', r'\_')):
        s = s.replace(a, b)
    return s


def _points_table(card):
    lines = [r'\begin{tabular}{|rlr|cl|}', r'\hline']
    for i, r in enumerate(card['rows'], 1):
        name = _esc(LABELS.get(r['feature'], r['feature']))
        plus = '' if i == 1 else r'$+$'
        lines.append(rf"{i}. & {name} & ${r['points']:+d}$ points & {plus} & $\cdots$ \\")
    lines += [r'\hline', r'\multicolumn{3}{|r|}{\textbf{SCORE}} & $=$ & \\',
              r'\hline', r'\end{tabular}']
    return '\n'.join(lines)


def _risk_table(card):
    pts = [r['points'] for r in card['rows']]
    b0 = card['intercept']
    scores = range(sum(p for p in pts if p < 0), sum(p for p in pts if p > 0) + 1)
    risk = [100.0 / (1.0 + np.exp(-(b0 + s))) for s in scores]
    spec = '|l|' + 'c|' * len(risk)
    head = ' & '.join([r'\textbf{SCORE}'] + [str(s) for s in scores])
    row = ' & '.join([r'\textbf{RISK}'] + [f'{r:.1f}\\%' for r in risk])
    return '\n'.join([rf'\begin{{tabular}}{{{spec}}}', r'\hline',
                      head + r' \\', r'\hline', row + r' \\', r'\hline', r'\end{tabular}'])


def _block(card, label):
    return '\n'.join([
        rf'% {label} (test AUC {card["auc"]:.3f})',
        r'\begin{minipage}[t]{0.49\linewidth}\centering',
        rf'\textbf{{{label}}}\quad(AUC {card["auc"]:.3f})\\[4pt]',
        rf'\resizebox{{\linewidth}}{{!}}{{{_points_table(card)}}}\\[6pt]',
        rf'\resizebox{{\linewidth}}{{!}}{{{_risk_table(card)}}}',
        r'\end{minipage}',
    ])


@figure('fig1a_scorecard', kind='main')
def fig1a_scorecard():
    data = json.loads(CARDS.read_text())
    return '\n'.join([
        r'% fig 1a: vanilla vs causal integer scorecards (fasterrisk style).',
        r'% \input inside a figure; needs graphicx (\resizebox). RISK = sigmoid(intercept + SCORE).',
        _block(data['vanilla'], 'vanilla'),
        r'\hfill',
        _block(data['causal'], 'causal (ours)'),
    ])


@figure('fig1b_rashomon', kind='main')
def fig1b_rashomon():
    pool = pd.read_csv(POOL)
    fig, ax = plt.subplots(figsize=(figstyle.COL_W, 2.6))
    for arm, label, colour, marker in [('vanilla', 'vanilla', '0.5', 'o'),
                                       ('causal', 'causal (ours)', 'C0', 's')]:
        d = pool[pool['arm'] == arm]
        ax.scatter(d['q_mass'].iloc[1:], d['auc'].iloc[1:], c=colour, alpha=0.4, s=18,
                   marker=marker, label=f'{label} (n={len(d)})')
        ax.scatter(d['q_mass'].iloc[0], d['auc'].iloc[0], c=colour, s=80, marker='*',
                   edgecolors='k', linewidths=0.5, zorder=5, label=f'{label} deployed')
    ax.set_xlabel('support q-mass')
    ax.set_ylabel('test AUC')
    ax.legend()
    return fig
