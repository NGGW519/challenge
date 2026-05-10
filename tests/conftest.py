"""pytest configuration — repo root을 PYTHONPATH에 추가."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "aic_model_pkg"))
