"""Make every level's modules importable from the tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in (
    "01_foundations",
    "02_local_search",
    "03_metaheuristics",
    "04_adversarial_search",
    "06_applications",
):
    sys.path.append(str(ROOT / sub))
