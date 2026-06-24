"""load and plot the real-data results, so the notebook stays thin: it imports these
helpers and calls them one line at a time, mirroring the synthetic recovery_plots.

no statistical tests, just test AUC and selection stability read straight off the
runs. run directories are produced by experiments/causal_prior/real/cfs.py (the cfs
comparison and the FICO scarcity sweep) and ksweep.py (the FICO parity sweep).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_RESULTS = Path(__file__).resolve().parents[2] / 'results' / 'causal_prior'
_CFS = _RESULTS / 'cfs'
_KSWEEP = _RESULTS / 'ksweep'
_RUNTIME = _RESULTS / 'real' / 'runtime' / 'runtime.csv'

BENCHMARKS = ['fico', 'heart', 'mammographic', 'ilpd', 'hepatitis', 'german']
# in-regime: enough samples per feature for causal discovery to be feasible.
# hepatitis (2.6 samples/feature) sits well below that and is the boundary case.
IN_REGIME = ['fico', 'heart', 'mammographic', 'ilpd', 'german']

# readable labels, grouped: vanilla, then our method, then the cfs baselines.
ARM_ORDER = ['vanilla', 'causal', 'iamb_soft_cg', 'iamb_soft_fz',
             'cfs_cg', 'cfs_iamb', 'cfs_hiton_mb']
ARM_LABEL = {'vanilla': 'vanilla', 'causal': 'ours (GES)', 'iamb_soft_cg': 'ours (IAMB)',
             'iamb_soft_fz': 'ours (IAMB, Fisher-Z)', 'cfs_cg': 'CFS valid (mi-cg)',
             'cfs_iamb': 'CFS naive (Fisher-Z)', 'cfs_hiton_mb': 'CFS naive (HITON)'}
# colour by group: ours = blues, vanilla = grey, cfs = warm
ARM_COLOR = {'vanilla': '0.5', 'causal': 'C0', 'iamb_soft_cg': 'C9', 'iamb_soft_fz': 'C4',
             'cfs_cg': 'C1', 'cfs_iamb': 'C3', 'cfs_hiton_mb': 'C5'}


def _latest_run(ds: str) -> str | None:
    """newest non-smoke run directory for a benchmark."""
    cands = sorted(glob.glob(str(_CFS / f'{ds}_k*')), key=os.path.getmtime)
    real = [d for d in cands
            if not (os.path.exists(d + '/config.json')
                    and json.load(open(d + '/config.json')).get('smoke'))]
    cands = real or cands
    return cands[-1] if cands else None


def load_runs(benchmarks: list[str] = IN_REGIME) -> dict:
    """{benchmark: {dir, summary, resamples}} for whichever runs are on disk.
    defaults to the in-regime benchmarks; hepatitis is handled separately."""
    out = {}
    for ds in benchmarks:
        d = _latest_run(ds)
        if d and os.path.exists(d + '/resamples.csv'):
            out[ds] = {'dir': os.path.basename(d),
                       'summary': pd.read_csv(d + '/summary.csv'),
                       'resamples': pd.read_csv(d + '/resamples.csv')}
    return out


def auc_table(runs: dict) -> pd.DataFrame:
    """test AUC as 'mean ± std', one row per benchmark, one column per arm. the ± is
    the standard deviation across the repeated train/test splits (higher mean better,
    overlapping ranges mean no reliable difference)."""
    rows = {}
    for ds, v in runs.items():
        s = v['summary'].iloc[-1]
        res = v['resamples']
        res = res[res['n'] == s['n']] if 'n' in res else res
        rows[ds] = {ARM_LABEL[a]: f"{float(s[f'auc_{a}']):.3f} ±{res[f'auc_{a}'].std():.3f}"
                    for a in ARM_ORDER}
    return pd.DataFrame(rows).T[[ARM_LABEL[a] for a in ARM_ORDER]]


def stability_table(runs: dict) -> pd.DataFrame:
    """selection stability (Nogueira index, 0 to 1), one row per benchmark per arm.
    higher means the same features are picked more reliably across resamples."""
    rows = {ds: {ARM_LABEL[a]: round(float(v['summary'].iloc[-1][f'nog_{a}']), 3)
                 for a in ARM_ORDER}
            for ds, v in runs.items()}
    return pd.DataFrame(rows).T[[ARM_LABEL[a] for a in ARM_ORDER]]


def samples_per_feature(benchmarks: list[str] = BENCHMARKS) -> pd.DataFrame:
    """n, one-hot feature count, and samples-per-feature per benchmark, sorted. causal
    discovery needs enough samples per feature; below about 5 it cannot find stable
    structure, which is why hepatitis is the boundary case."""
    import warnings
    from experiments.causal_prior.real.datasets import load_dataset

    class _A:
        sentinel_nan = True

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for ds in benchmarks:
            X, names, y = load_dataset(ds, _A())
            ntr = int(0.7 * len(y))
            rows.append({'benchmark': ds, 'n': len(y), 'features (1-hot)': len(names),
                         'samples/feature': round(ntr / len(names), 1)})
    return pd.DataFrame(rows).set_index('benchmark').sort_values('samples/feature', ascending=False)


def boundary_case(ds: str = 'hepatitis') -> pd.DataFrame:
    """the out-of-regime benchmark: our method underperforms vanilla because, with too
    few samples per feature, discovery yields a noisy prior and cv keeps it on. the
    'prior on' column is the fraction of resamples where cv selected a nonzero mu."""
    runs = load_runs([ds])
    s = runs[ds]['summary'].iloc[-1]
    return pd.DataFrame({'vanilla AUC': [round(float(s['auc_vanilla']), 3)],
                         'our AUC': [round(float(s['auc_causal']), 3)],
                         'prior on (frac of runs)': [round(float(s['mu_nonzero_frac']), 2)]},
                        index=[ds])


def plot_auc(runs: dict):
    """one small bar chart per benchmark: each arm's test AUC, with a dashed line at
    vanilla. our method sits at or above vanilla; the CFS baselines sit below."""
    ds = list(runs)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, d in zip(axes.ravel(), ds):
        s = runs[d]['summary'].iloc[-1]
        res = runs[d]['resamples']
        res = res[res['n'] == s['n']] if 'n' in res else res
        aucs = [float(s[f'auc_{a}']) for a in ARM_ORDER]
        errs = [float(res[f'auc_{a}'].std()) for a in ARM_ORDER]
        ax.barh(range(len(ARM_ORDER)), aucs, xerr=errs, capsize=2,
                color=[ARM_COLOR[a] for a in ARM_ORDER])
        ax.axvline(float(s['auc_vanilla']), color='0.3', ls='--', lw=1)
        ax.set_yticks(range(len(ARM_ORDER)))
        ax.set_yticklabels([ARM_LABEL[a] for a in ARM_ORDER], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(min(aucs) - 0.02, max(aucs) + 0.01)
        ax.set_title(d)
    for ax in axes.ravel()[len(ds):]:
        ax.axis('off')
    fig.supxlabel('test AUC (dashed line = vanilla)')
    fig.tight_layout()


def plot_parity(ksweep_dir: str | None = None):
    """FICO parity sweep: our method vs vanilla AUC as model size k grows. The two
    lines lie on top of each other, so the prior costs no accuracy (do-no-harm)."""
    if ksweep_dir is None:
        cands = sorted(glob.glob(str(_KSWEEP / '*')), key=os.path.getmtime)
        cands = [c for c in cands if os.path.exists(c + '/ksweep.csv')]
        ksweep_dir = cands[-1]
    df = pd.read_csv(ksweep_dir + '/ksweep.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, color, lab in [('vanilla', '0.5', 'vanilla'), ('causal', 'C0', 'ours')]:
        a = df[df['arm'] == arm].sort_values('k')
        ax.errorbar(a['k'], a['auc_mean'], yerr=a['auc_std'], marker='o',
                    color=color, label=lab, capsize=3)
    ax.set_xlabel('model size (k)')
    ax.set_ylabel('test AUC')
    ax.set_title('FICO: accuracy parity (ours vs vanilla)')
    ax.legend()
    fig.tight_layout()


def plot_scarcity(runs: dict):
    """FICO as the training set shrinks. Left: our method keeps a far more stable
    feature set than vanilla. Right: accuracy stays level with vanilla and well above
    the valid CFS baseline. The stability gain is largest when data is scarcest."""
    s = runs['fico']['summary'].sort_values('n')
    arms = [('vanilla', '0.5', 'vanilla'), ('causal', 'C0', 'ours'),
            ('cfs_cg', 'C1', 'CFS valid')]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4))
    for arm, color, lab in arms:
        axl.plot(s['n'], s[f'nog_{arm}'], marker='o', color=color, label=lab)
        axr.plot(s['n'], s[f'auc_{arm}'], marker='o', color=color, label=lab)
    axl.set_xlabel('training rows (n)'); axl.set_ylabel('selection stability')
    axl.set_title('feature stability'); axl.legend()
    axr.set_xlabel('training rows (n)'); axr.set_ylabel('test AUC')
    axr.set_title('accuracy'); axr.legend()
    fig.tight_layout()


_KSWEEP_CFS = _RESULTS / 'ksweep_cfs'


def load_ksweep(benchmarks: list[str] = IN_REGIME) -> dict:
    """{benchmark: long dataframe [rep, arm, k, auc, n, p]} from the k-sweep runs.
    skips smoke runs; defaults to the in-regime benchmarks."""
    out = {}
    for ds in benchmarks:
        good = []
        for c in sorted(glob.glob(str(_KSWEEP_CFS / f'{ds}*')), key=os.path.getmtime):
            if not os.path.exists(c + '/ksweep_arms.csv'):
                continue
            cfg = c + '/config.json'
            if os.path.exists(cfg) and json.load(open(cfg)).get('smoke'):
                continue
            good.append(c)
        if good:
            out[ds] = pd.read_csv(good[-1] + '/ksweep_arms.csv')
    return out


def plot_ksweep_grid(ksweep: dict, arms: list[str] | None = None):
    """FasterRisk-style grid: test AUC vs model size k, one panel per benchmark, each
    arm a line with error bars (mean ± std across the repeated splits)."""
    if not ksweep:
        print('k-sweep runs not finished yet; re-run this cell when they land.')
        return
    if arms is None:
        arms = ARM_ORDER
    ds = list(ksweep)
    ncol = 3
    nrow = int(np.ceil(len(ds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
    for ax, d in zip(axes.ravel(), ds):
        df = ksweep[d]
        n, p = int(df['n'].iloc[0]), int(df['p'].iloc[0])
        for a in arms:
            g = df[df['arm'] == a].groupby('k')['auc']
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker='o',
                        ms=4, lw=1, capsize=2, color=ARM_COLOR[a], label=ARM_LABEL[a])
        ax.set_title(f'{d} (n={n}, p={p})')
        ax.set_xlabel('model size (k)')
        ax.set_ylabel('test AUC')
    for ax in axes.ravel()[len(ds):]:
        ax.axis('off')
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])


# runtime variants: vanilla, hard CFS, and ours (deployed = discovery + auc-cv + fit)
_RT_VARIANTS = ['vanilla', 'cfs_hard', 'ours']
_RT_LABEL = {'vanilla': 'vanilla', 'cfs_hard': 'CFS (hard)', 'ours': 'ours'}


def load_runtime() -> pd.DataFrame | None:
    """per-dataset runtime + AUC table written by experiments/.../runtime_bench.py.
    one row per benchmark; columns t_/auc_ per variant. None if not run yet."""
    return pd.read_csv(_RUNTIME) if _RUNTIME.exists() else None


def runtime_table(rt: pd.DataFrame) -> pd.DataFrame:
    """per-deployment wall-clock seconds. 'ours' is discovery + auc-cv mu tuning +
    fit; the mu-cv stage is broken out (it is a one-time per-deployment cost, not
    per-prediction). speed-matched to hard CFS on the shared discovery."""
    cols = {_RT_LABEL[v]: rt[f't_{v}'].round(1) for v in _RT_VARIANTS}
    out = pd.DataFrame({'dataset': rt['dataset'], **cols,
                        'of which mu-cv (s)': rt['t_mu_cv'].round(1),
                        'ours, no re-tune (s)': rt['t_ours_notune'].round(1)})
    return out.set_index('dataset')


def runtime_auc_table(rt: pd.DataFrame) -> pd.DataFrame:
    """test AUC per variant at the timed split: ours (auc-cv tuned) is at or above
    vanilla and above hard CFS, at a wall-clock that is still seconds."""
    cols = {_RT_LABEL[v]: rt[f'auc_{v}'].round(3) for v in _RT_VARIANTS}
    return pd.DataFrame({'dataset': rt['dataset'], **cols}).set_index('dataset')
