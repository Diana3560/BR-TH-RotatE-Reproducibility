from __future__ import annotations

from pathlib import Path

import yaml


def test_stage3_public_config_contract():
    root = Path(__file__).resolve().parents[1]
    stage3_cfg = yaml.safe_load((root / "stage3" / "config.yaml").read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load((root / "config" / "multidataset_comparison.yaml").read_text(encoding="utf-8"))
    assert stage3_cfg["stage3_protocol"]["pykeen_version"] == "1.11.1"
    assert base_cfg["training"]["max_steps"] == 3000
    assert base_cfg["training"]["batch_size"] == 1024
    assert base_cfg["training"]["num_negatives"] == 64
    assert all(str(spec["split_dir"]).startswith("external_data/") for spec in base_cfg["datasets"].values())
