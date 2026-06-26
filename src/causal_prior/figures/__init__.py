"""figure builders: pure functions that load from results/ and return a Figure.

register with @figure(name, kind); add the module to _MODULES so it loads.
`python -m src.causal_prior.figures` renders and saves them; experiment scripts never plot.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def latest_run(parent: Path, prefix: str = '', needs: str | None = None) -> Path:
    """newest run dir under parent matching prefix* (optionally containing `needs`)."""
    runs = [p for p in parent.glob(f'{prefix}*')
            if p.is_dir() and (needs is None or (p / needs).exists())]
    if not runs:
        raise FileNotFoundError(f'{parent}/{prefix}*' + (f' with {needs}' if needs else ''))
    return max(runs, key=lambda p: p.stat().st_mtime)

# builder modules to import so their @figure decorators register (empty: scaffold).
_MODULES: list[str] = [
    'src.causal_prior.figures.mb_recovery',
    'src.causal_prior.figures.theory',
    'src.causal_prior.figures.recovery',
    'src.causal_prior.figures.transport',
    'src.causal_prior.figures.scorecard',
    'src.causal_prior.figures.cfs',
    'src.causal_prior.figures.q_corruption',
]


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    fn: Callable


REGISTRY: dict[str, Spec] = {}


def figure(name: str, kind: str = 'main') -> Callable[[Callable], Callable]:
    if kind not in ('main', 'appendix'):
        raise ValueError(f"kind must be 'main' or 'appendix', got {kind!r}")

    def deco(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f'duplicate figure name: {name!r}')
        REGISTRY[name] = Spec(name, kind, fn)
        return fn

    return deco


def load_builders() -> None:
    for mod in _MODULES:
        importlib.import_module(mod)


def specs() -> list[Spec]:
    load_builders()
    return sorted(REGISTRY.values(), key=lambda s: (s.kind, s.name))
