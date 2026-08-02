from __future__ import annotations

import importlib
from pathlib import Path


def test_metrics_module_is_packaged_and_importable() -> None:
    package_root = Path(__file__).resolve().parents[1] / "psgtr_hf"
    assert (package_root / "metrics.py").is_file()

    module = importlib.import_module("psgtr_hf.metrics")
    assert hasattr(module, "PsgEvaluationAccumulator")
