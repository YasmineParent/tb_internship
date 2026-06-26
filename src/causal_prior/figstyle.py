"""figure style: rcParams (paper.mplstyle), column widths, palette, and save().

builders pull COL_W / TEXT_W and the palette from here instead of hard-coding,
and never write files; the build driver calls save().
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / 'figures'
_STYLE = Path(__file__).with_name('paper.mplstyle')

# aaai two-column widths in inches; size figures at the printed width so fonts
# render at the intended pt.
COL_W = 3.3
TEXT_W = 7.0
ROW_H = 2.4
PANEL = (COL_W, ROW_H)   # single-column panel size

# synthetic q-sources -> (label, colour, linestyle). deployed convention:
# vanilla grey, causal blue, predictive warm, controls dotted.
SOURCES: dict[str, tuple[str, str, str]] = {
    'uniform':      ('vanilla (uniform $q$)',        '0.5', '-'),
    'iamb':         ('IAMB (MB, deployed)',          'C0',  '-'),
    'ges':          ('GES (global causal)',          'C9',  '-'),
    'pc':           ('PC (global causal)',           'C2',  '-'),
    'bootstrap_l1': ('bootstrap-$L_1$ (predictive)', 'C3',  '-'),
    'adversarial':  ('adversarial (control)',        'C1',  ':'),
}

# real-data arms -> colour. labels stay in cfs_results.ARM_LABEL; keep names in sync.
ARM_COLOR: dict[str, str] = {
    'vanilla':      '0.5',
    'causal':       'C0',
    'iamb_soft_cg': 'C9',
    'iamb_soft_fz': 'C4',
    'cfs_cg':       'C1',
    'cfs_iamb':     'C3',
    'cfs_hiton_mb': 'C5',
}


def use_paper_style() -> None:
    plt.style.use(str(_STYLE))


def save(fig, name: str, kind: str = 'main') -> Path:
    """write fig to figures/<kind>/<name>.pdf; return the path."""
    if kind not in ('main', 'appendix'):
        raise ValueError(f"kind must be 'main' or 'appendix', got {kind!r}")
    out = FIG_DIR / kind
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f'{name}.pdf'
    fig.savefig(pdf)
    return pdf


def save_tex(text: str, name: str, kind: str = 'main') -> Path:
    """write a latex snippet to figures/<kind>/<name>.tex; return the path."""
    if kind not in ('main', 'appendix'):
        raise ValueError(f"kind must be 'main' or 'appendix', got {kind!r}")
    out = FIG_DIR / kind
    out.mkdir(parents=True, exist_ok=True)
    tex = out / f'{name}.tex'
    tex.write_text(text + '\n')
    return tex
