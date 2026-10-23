"""Support optional project-local dependencies without changing global Python."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / '.packages').exists():
    sys.path.insert(0, str(ROOT / '.packages'))
