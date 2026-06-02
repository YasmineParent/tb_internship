"""Shared experiment configuration for the §6.1 mechanism test.

DEFAULTS is the canonical operating point (the p=30 n=300 headline shape);
each sweep in SWEEPS varies one axis at a time holding the others at
DEFAULTS. The default value is included in every sweep so it acts as a
reference point in cross-sweep plots.

The 'p_edge' sweep produces the headline figures (selectivity-driven
recovery as a function of confounding); the other three are robustness
slices defending the anchor cell against op-point cherry-pick concerns.
CLI overrides (--p, --n, --k-star in stability_selection.py) allow
shooting at alternate op-points without touching DEFAULTS.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULTS = dict(p=30, n=300, k_star=5, p_edge=0.2)

SWEEPS: dict[str, list] = {
    'p_edge':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # headline; 0.7 added for extreme-confounding tail
    'n':       [75, 100, 150, 200, 300, 500, 1000],  # sample-efficiency; 75/100 probe the data-starved regime
    'p':       [10, 15, 20, 30, 50],                  # feature-count robustness; 50 stresses GES
    'k_star':  [3, 5, 7],                             # true-sparsity robustness
}

# per-sweep seed budget; p_edge is the headline so more seeds for tight bands
SEED_COUNTS: dict[str, int] = {
    'p_edge': 20,
    'n':      10,
    'p':      5,
    'k_star': 5,
}

PARAM_LABELS = {
    'p_edge':  'DAG edge density',
    'n':       'sample size',
    'p':       '#features',
    'k_star':  '|S*| (true causes)',
}

DEFAULT_N_SEEDS = 5  # used only when CLI --n-seeds overrides SEED_COUNTS uniformly


@dataclass(frozen=True, order=True)
class Cell:
    p: int
    n: int
    k_star: int
    p_edge: float
    seed: int

    @property
    def filename(self) -> str:
        return (f'seed{self.seed}_p{self.p}_n{self.n}_k{self.k_star}'
                f'_pedge{self.p_edge}.npz')


def build_cells(scope: str, n_seeds: int | None = None,
                overrides: dict | None = None) -> list[Cell]:
    """Return the (deduplicated, sorted) cells for one sweep or all sweeps.

    scope: a sweep name from SWEEPS, or 'all' to take the union across sweeps.
    n_seeds: if None, use per-sweep SEED_COUNTS; if set, override uniformly
        (CLI --n-seeds path).
    overrides: optional dict of {p, n, k_star, p_edge} values to override
        DEFAULTS for the base shape; the sweep axis still varies on top.
    """
    if scope == 'all':
        sweep_names = list(SWEEPS.keys())
    elif scope in SWEEPS:
        sweep_names = [scope]
    else:
        raise ValueError(f'unknown scope {scope!r}; must be one of '
                         f'{list(SWEEPS.keys()) + ["all"]}')

    base = {**DEFAULTS, **(overrides or {})}
    cells: set[Cell] = set()
    for sweep in sweep_names:
        seeds_for_sweep = (n_seeds if n_seeds is not None
                           else SEED_COUNTS.get(sweep, DEFAULT_N_SEEDS))
        for val in SWEEPS[sweep]:
            params = {**base, sweep: val}
            for seed in range(seeds_for_sweep):
                cells.add(Cell(seed=seed, **params))
    return sorted(cells)
