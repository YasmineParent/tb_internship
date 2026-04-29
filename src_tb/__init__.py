import sys
from pathlib import Path

_CMM_PATH = Path(__file__).resolve().parents[1] / "external" / "cmm"
if str(_CMM_PATH) not in sys.path:
    sys.path.insert(0, str(_CMM_PATH))
