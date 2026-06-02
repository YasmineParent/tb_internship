"""Top-level package for the causally-informed risk scoring project.

Side effects on import:
- Silences rpy2's R console writebacks (warnings, prints).
- Extends ``src.__path__`` to include CMM's vendored ``external/cmm/src/`` so
  CMM modules are importable as ``src.exp.*``, ``src.mixtures.*``, etc. Our
  own subpackages (``src.causal_prior``, ``src.data``, ``src.causal_discovery``)
  take precedence because our src/ comes first on the path.
"""
from pathlib import Path

from rpy2.rinterface_lib import callbacks

callbacks.consolewrite_warnerror = lambda x: None
callbacks.consolewrite_print = lambda x: None

_CMM_SRC = Path(__file__).resolve().parents[1] / 'external' / 'cmm' / 'src'
if str(_CMM_SRC) not in __path__:
    __path__.append(str(_CMM_SRC))
