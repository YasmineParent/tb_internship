"""Shared I/O helpers for experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path


def new_run_dir(base: Path, config: dict | None = None) -> Path:
    """Create a fresh results directory that never overwrites an existing one.

    If `base` already exists, appends `_2`, `_3`, ... until an unused name is
    found. When `config` is given, dumps it to `config.json` inside the new
    directory. Returns the created directory.
    """
    out, i = base, 2
    while out.exists():
        out = base.with_name(base.name + f'_{i}')
        i += 1
    out.mkdir(parents=True)
    if config is not None:
        (out / 'config.json').write_text(json.dumps(config, indent=2))
    return out
