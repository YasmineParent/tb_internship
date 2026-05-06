import sys
from pathlib import Path
from rpy2.rinterface_lib import callbacks
import logging

callbacks.consolewrite_warnerror = lambda x: None
callbacks.consolewrite_print = lambda x: None

_CMM_PATH = Path(__file__).resolve().parents[1] / "external" / "cmm"
if str(_CMM_PATH) not in sys.path:
    sys.path.insert(0, str(_CMM_PATH))
