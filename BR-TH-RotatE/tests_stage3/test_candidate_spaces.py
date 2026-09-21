from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from stage3.candidate_spaces import (
    audit_type_schema_gold_coverage,
    build_candidate_artifacts,
    filtered_candidate_counts,
    nominal_candidate_counts,
    summarize_counts,
)
from throtate_repro.multidataset_data import dataset_paths, read_entity_types


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = yaml.safe_load((ROOT / "config" / "multidataset_comparison.yaml").read_text(encoding="utf-8"))


def _require_dataset(dataset_key: str) -> None:
    paths = dataset_paths(ROOT, BASE_CFG, dataset_key)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        pytest.skip("external dataset files are intentionally absent from the public code release")


def test_ownkg_candidate_counts_and_train_only_schema():
    _require_dataset("ownkg")
    artifacts = build_candidate_artifacts(ROOT, BASE_CFG, "ownkg")
    assert len(artifacts.full_entities) == 13693
    assert len(artifacts.modeled_entities) == 11653

    expected = {
        "APPLIES_TO": ({"TrainModel"}, {"Component"}),
        "APPLIES_UNDER": ({"Standard"}, {"ConditionRule"}),
        "CAUSED_BY": ({"FailureMode"}, {"FaultCause"}),
        "DETECTED_BY": ({"Component", "FailureMode"}, {"ProcessPlan"}),
        "FIXED_BY": ({"FailureMode"}, {"DispositionRule", "ProcessPlan"}),
        "HAS_DISPOSITION": ({"Standard"}, {"DispositionRule"}),
        "HAS_FAILURE": ({"Component"}, {"FailureMode"}),
        "HAS_PART": ({"Component"}, {"Component"}),
        "HAS_STEP": ({"ProcessPlan"}, {"ProcessStep"}),
        "HAS_SYMPTOM": ({"FailureMode"}, {"Symptom"}),
        "INDICATES": ({"Symptom"}, {"FailureMode"}),
        "LIMITED_BY": ({"Component", "FailureMode"}, {"Standard"}),
        "NEXT_STEP": ({"ProcessStep"}, {"ProcessStep"}),
        "REQUIRES": ({"ProcessPlan", "ProcessStep"}, {"ToolMaterial"}),
    }
    actual = {
        relation: (set(spec["head_types"]), set(spec["tail_types"]))
        for relation, spec in artifacts.type_schema.items()
    }
    assert actual == expected

    entity_types = read_entity_types(dataset_paths(ROOT, BASE_CFG, "ownkg")["entity_types"])
    assert audit_type_schema_gold_coverage(artifacts, entity_types)["status"] == "PASS"


def test_ownkg_candidate_space_rank_counts_and_ranges():
    _require_dataset("ownkg")
    artifacts = build_candidate_artifacts(ROOT, BASE_CFG, "ownkg")
    for space in ("full", "modeled_entities", "type_constrained"):
        nominal = nominal_candidate_counts(artifacts, space=space)
        filtered = filtered_candidate_counts(artifacts, space=space)
        assert len(nominal) == 2016
        assert len(filtered) == 2016
        assert min(filtered) >= 1
        assert all(f <= n for f, n in zip(filtered, nominal))

    full = summarize_counts(nominal_candidate_counts(artifacts, space="full"))
    modeled = summarize_counts(nominal_candidate_counts(artifacts, space="modeled_entities"))
    typed = summarize_counts(nominal_candidate_counts(artifacts, space="type_constrained"))
    assert full["min"] == full["max"] == 13693
    assert modeled["min"] == modeled["max"] == 11653
    assert typed["max"] < 11653
    assert typed["min"] >= 1


def test_paper4_train_schema_keeps_all_gold_targets():
    _require_dataset("paper4")
    artifacts = build_candidate_artifacts(ROOT, BASE_CFG, "paper4")
    assert len(artifacts.full_entities) == 8105
    assert len(artifacts.modeled_entities) == 8105
    entity_types = read_entity_types(dataset_paths(ROOT, BASE_CFG, "paper4")["entity_types"])
    assert audit_type_schema_gold_coverage(artifacts, entity_types)["status"] == "PASS"
