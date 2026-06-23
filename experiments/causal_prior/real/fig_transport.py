"""Fig 2: out-of-environment transport composite (synthetic / semi-synthetic / real)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SHIFT_CSV = ROOT / 'results/causal_prior/synthetic/recovery_shift/headline/shift.csv'
# most recent inject-spurious musweep (named with inject suffix)
FOLKTABLES_DIR = ROOT / 'results/causal_prior/folktables_transport'
OUT_DEFAULT = ROOT / 'results/causal_prior/real/fig_transport.png'

CAUSAL_COLOR = '#2171b5'
VANILLA_COLOR = '#888'
ICP_COLOR = '#e6550d'


def _latest_dir(prefix: str) -> Path | None:
    cands = sorted(FOLKTABLES_DIR.glob(f'{prefix}*'), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def panel_a(ax, df_shift):
    """panel (a): transport gap vs correlate reliance in synthetic data."""
    bins = [-0.01, 0.001, 0.1, 0.2, 0.3, 1.01]
    labels = ['0', '(0,.1]', '(.1,.2]', '(.2,.3]', '>.3']
    d = df_shift[df_shift.test_gamma == -1.0].dropna(subset=['correlate_inclusion', 'delta_auc'])
    d = d.copy()
    d['bin'] = pd.cut(d['correlate_inclusion'], bins, labels=labels)
    grp = d.groupby('bin', observed=True)['delta_auc']
    mean, sem = grp.mean(), grp.std() / np.sqrt(grp.count())
    r = np.corrcoef(d['correlate_inclusion'], d['delta_auc'])[0, 1]
    ax.bar(range(len(mean)), mean.values, yerr=sem.values, capsize=3,
           color='#c44', alpha=0.82, edgecolor='white')
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels(mean.index)
    ax.set_xlabel('correlate reliance of fitted support')
    ax.set_ylabel('transport gap  (AUC$_{in}$ - AUC$_{shift}$)')
    ax.set_title(f'(a) synthetic mechanism\nPearson $r$ = {r:.2f}', fontsize=9)


def panel_b(ax, df_inject):
    """panel (b): inject-spurious semi-synthetic — gap vs mu."""
    targets = [c.replace('gap_', '') for c in df_inject.columns if c.startswith('gap_')]
    for arm, color, label, ls in [
        ('vanilla',    VANILLA_COLOR, 'vanilla', '--'),
        ('causal',     CAUSAL_COLOR,  'causal (ours)', '-'),
        ('causal_icp', ICP_COLOR,     'causal (ICP)', ':'),
    ]:
        d = df_inject[df_inject['arm'] == arm].copy()
        if d.empty:
            continue
        d = d.sort_values('mu_rel')
        for t in targets:
            y = d[f'gap_{t}'].values
            x = d['mu_rel'].values
            ax.plot(x, y, marker='o', ms=4, color=color, ls=ls, lw=1.5,
                    label=f'{label} → {t}' if len(targets) > 1 else label)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('prior strength  $\\mu / \\mu_{\\mathrm{scale}}$')
    ax.set_ylabel('transport gap  (AUC$_{src}$ - AUC$_{tgt}$)')
    ax.set_title('(b) semi-synthetic (real CA features)\nvanilla flat, causal closes gap as $\\mu$ rises',
                 fontsize=9)
    ax.legend(fontsize=7, framealpha=0.8)


def panel_c(ax, df_real):
    """panel (c): real folktables — do-no-harm (causal ≈ vanilla)."""
    targets = [c.replace('gap_', '') for c in df_real.columns if c.startswith('gap_')]
    for arm, color, label, ls in [
        ('vanilla',    VANILLA_COLOR, 'vanilla', '--'),
        ('causal',     CAUSAL_COLOR,  'causal', '-'),
    ]:
        d = df_real[df_real['arm'] == arm].copy()
        if d.empty:
            continue
        d = d.sort_values('mu_rel')
        for t in targets:
            y = d[f'gap_{t}'].values
            x = d['mu_rel'].values
            ax.plot(x, y, marker='o', ms=4, color=color, ls=ls, lw=1.5,
                    label=f'{label} → {t}' if len(targets) > 1 else label)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('prior strength  $\\mu / \\mu_{\\mathrm{scale}}$')
    ax.set_ylabel('transport gap')
    ax.set_title('(c) folktables real (CA → PR, SD)\ndo-no-harm: causal $\\approx$ vanilla', fontsize=9)
    ax.legend(fontsize=7, framealpha=0.8)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--no-panel-b', action='store_true',
                    help='build panels a and c only (inject-spurious run pending)')
    ap.add_argument('--inject-dir', type=Path, default=None,
                    help='specific inject-spurious result dir; auto-detected if omitted')
    ap.add_argument('--real-dir', type=Path, default=None,
                    help='specific folktables result dir; auto-detected if omitted')
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # panel (a): always available
    if not SHIFT_CSV.exists():
        print(f'shift CSV not found: {SHIFT_CSV}', flush=True)
        return
    df_shift = pd.read_csv(SHIFT_CSV)

    # panel (c): real folktables musweep, no inject-spurious (do-no-harm)
    import json as _json
    if args.real_dir:
        real_dir = args.real_dir
    else:
        named = FOLKTABLES_DIR / 'plain'
        if named.exists() and (named / 'summary.csv').exists():
            real_dir = named
        else:
            real_cands = sorted(FOLKTABLES_DIR.glob('*'),
                                key=lambda p: p.stat().st_mtime)
            real_dir = next(
                (p for p in real_cands if p.is_dir()
                 and (p / 'config.json').exists()
                 and not _json.load(open(p / 'config.json')).get('inject_spurious')),
                None)
    if real_dir is None or not (real_dir / 'summary.csv').exists():
        print(f'real folktables dir not found under {FOLKTABLES_DIR}', flush=True)
        return
    df_real = pd.read_csv(real_dir / 'summary.csv').reset_index()
    df_real.columns = [c if c != 'level_0' else 'arm' for c in df_real.columns]
    df_real = df_real.rename(columns={'level_1': 'mu_rel'}) if 'level_1' in df_real.columns else df_real
    print(f'real dir: {real_dir.name}  arms={df_real["arm"].unique().tolist()[:4]}', flush=True)

    if args.no_panel_b:
        fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(11, 4))
        panel_a(ax_a, df_shift)
        panel_c(ax_c, df_real)
        fig.suptitle('Out-of-environment transport', fontsize=11)
    else:
        if args.inject_dir:
            inject_dir = args.inject_dir
        else:
            named = FOLKTABLES_DIR / 'inject'
            if named.exists() and (named / 'summary.csv').exists():
                inject_dir = named
            else:
                cands = sorted(FOLKTABLES_DIR.glob('*'), key=lambda p: p.stat().st_mtime)
                inject_dir = next(
                    (p for p in reversed(cands) if p.is_dir()
                     and (p / 'config.json').exists()
                     and _json.load(open(p / 'config.json')).get('inject_spurious')),
                    None)
        if inject_dir is None or not (inject_dir / 'summary.csv').exists():
            print('inject-spurious dir not found; build with --no-panel-b for now', flush=True)
            return
        df_inject = pd.read_csv(inject_dir / 'summary.csv').reset_index()
        df_inject.columns = [c if c != 'level_0' else 'arm' for c in df_inject.columns]
        df_inject = df_inject.rename(columns={'level_1': 'mu_rel'}) if 'level_1' in df_inject.columns else df_inject
        print(f'inject dir: {inject_dir.name}  arms={df_inject["arm"].unique().tolist()[:4]}', flush=True)
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4))
        panel_a(ax_a, df_shift)
        panel_b(ax_b, df_inject)
        panel_c(ax_c, df_real)
        fig.suptitle('Out-of-environment transport: mechanism, semi-synthetic demonstration, and real-data scope',
                     fontsize=10)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
