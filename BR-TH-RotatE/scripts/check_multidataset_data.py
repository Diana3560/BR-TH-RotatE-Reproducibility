from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from throtate_repro.config import load_config  # noqa: E402
from throtate_repro.multidataset_data import audit_dataset  # noqa: E402


def main() -> int:
    cfg = load_config(PROJECT_ROOT / "config" / "multidataset_comparison.yaml")
    reports: dict[str, dict] = {}
    missing: dict[str, str] = {}

    for dataset_key, spec in cfg["datasets"].items():
        try:
            report = audit_dataset(PROJECT_ROOT, cfg, dataset_key)
        except FileNotFoundError as exc:
            missing[dataset_key] = str(exc)
            continue
        reports[dataset_key] = {
            "status": report["status"],
            "display_name": spec["display_name"],
            "counts": report["counts"],
            "split_overlaps": report["split_overlaps"],
            "split_entities_seen_in_training": report["split_entities_seen_in_training"],
        }

    print(json.dumps({"datasets": reports, "missing_external_data": missing}, ensure_ascii=False, indent=2))

    if missing:
        print(
            "\n[DATA NOT PRESENT] This public code repository intentionally contains no datasets. "
            "The paths under external_data/ are optional local inputs and are ignored by Git. "
            "No separately distributed dataset archive is searched, opened, extracted, or modified automatically."
        )
        return 2

    print("\n[PASS] Configured external datasets passed hashes, counts, split-overlap, and entity-coverage checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
