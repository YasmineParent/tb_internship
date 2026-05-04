import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison(records_old: list[dict], records_new: list[dict], metrics: list[str], labels: list[str],
                    label_old: str = 'Gaussian (noise)', label_new: str = 'Logistic (FLXMRglm)'):
    """Grouped bar chart comparing any set of metrics between two models."""
    df_old = pd.DataFrame(records_old)
    df_new = pd.DataFrame(records_new)

    means_old = [df_old[m].mean() for m in metrics]
    means_new = [df_new[m].mean() for m in metrics]
    stds_old  = [df_old[m].std()  for m in metrics]
    stds_new  = [df_new[m].std()  for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(2 + 2 * len(labels), 4))
    ax.bar(x - width / 2, means_old, width, yerr=stds_old, label=label_old, capsize=4, color='steelblue')
    ax.bar(x + width / 2, means_new, width, yerr=stds_new, label=label_new, capsize=4, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_sweep(df: pd.DataFrame, metrics: list[str], metric_labels: dict,
               param_labels: dict | None = None, group_col: str = 'model',
               group_labels: dict | None = None, title: str = '', figsize_scale: float = 4.0):
    """Figure D.2-style grid: rows=metrics, columns=sweep parameters.
    df must have columns: param, value, <group_col>, and all metrics.
    Param order is taken from df['param'].unique(); pass param_labels for nice column titles.
    group_col is 'model' for run_parameter_sweep results, 'method' for run_baselines_sweep results."""
    params = list(dict.fromkeys(df['param']))
    param_labels = param_labels or {}
    group_labels = group_labels or {}
    groups = sorted(df[group_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))

    fig, axes = plt.subplots(len(metrics), len(params),
                              figsize=(figsize_scale * len(params), 3 * len(metrics)))
    if len(metrics) == 1:
        axes = axes[np.newaxis, :]

    for col, param in enumerate(params):
        sub = df[df['param'] == param]
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for group, color in zip(groups, colors):
                g = sub[sub[group_col] == group].groupby('value')[metric]
                means, stds = g.mean(), g.std().fillna(0)
                ax.plot(means.index, means.values, color=color,
                        label=group_labels.get(group, group), marker='o', markersize=3)
                ax.fill_between(means.index, means - stds, means + stds, alpha=0.2, color=color)
            if 'fpr' in metric.lower():
                ax.set_ylim(0, 0.2)
                ax.set_yticks([0, 0.1, 0.2])
            else:
                ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title(param_labels.get(param, param), fontsize=9)
            if col == 0:
                ax.set_ylabel(metric_labels.get(metric, metric))
            if row == len(metrics) - 1:
                ax.set_xlabel(param)

    axes[0, -1].legend(fontsize=8)
    if title:
        plt.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.show()
