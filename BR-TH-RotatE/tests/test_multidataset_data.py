from __future__ import annotations

from pathlib import Path

import pytest

from throtate_repro.config import load_config
from throtate_repro.multidataset_data import (
    audit_dataset,
    compute_training_structure_features,
    dataset_paths,
    read_entity_types,
    read_triples,
    stable_fingerprint,
    validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "multidataset_comparison.yaml"


def _require_dataset_files(dataset_key: str):
    cfg = load_config(CONFIG_PATH)
    paths = dataset_paths(PROJECT_ROOT, cfg, dataset_key)
    missing = [str(path) for path in paths.values() if isinstance(path, Path) and not path.is_file()]
    if missing:
        pytest.skip(f"restricted/private dataset files are not packaged: {missing}")
    return cfg, paths


def test_multidataset_config_contract() -> None:
    cfg = load_config(CONFIG_PATH)
    validate_config(cfg)
    assert tuple(cfg["datasets"]) == ("ownkg", "paper4")
    assert cfg["formal"]["models"] == ["TransH", "RotatE", "D0", "D1", "A0", "D2"]
    assert cfg["formal"]["seeds"] == [42, 43, 44, 45, 46]


@pytest.mark.parametrize(
    ("dataset_key", "entities", "relations", "train", "validation", "test"),
    [
        ("ownkg", 13693, 14, 15413, 1009, 1008),
        ("paper4", 8105, 9, 10982, 1373, 1373),
    ],
)
def test_packaged_dataset_audits(
    dataset_key: str,
    entities: int,
    relations: int,
    train: int,
    validation: int,
    test: int,
) -> None:
    cfg, _ = _require_dataset_files(dataset_key)
    report = audit_dataset(PROJECT_ROOT, cfg, dataset_key)
    assert report["status"] == "PASS"
    assert report["counts"]["candidate_entities"] == entities
    assert report["counts"]["modeled_relations"] == relations
    assert report["counts"]["train_triples"] == train
    assert report["counts"]["validation_triples"] == validation
    assert report["counts"]["test_triples"] == test
    assert not any(report["split_overlaps"].values())
    assert report["split_entities_seen_in_training"] is True


def test_generic_ownkg_feature_builder_covers_all_fourteen_relations() -> None:
    cfg, paths = _require_dataset_files("ownkg")
    spec = cfg["datasets"]["ownkg"]
    computed = compute_training_structure_features(
        read_triples(paths["train"]),
        read_entity_types(paths["entity_types"]),
        spec["modeled_relations"],
        spec["relation_roles"],
    )
    assert len(computed) == 14
    assert {row["relation"] for row in computed} == set(spec["modeled_relations"])
    assert all(int(row["train_triples"]) > 0 for row in computed)

def test_paper4_features_cover_all_nine_training_relations() -> None:
    cfg, paths = _require_dataset_files("paper4")
    spec = cfg["datasets"]["paper4"]
    rows = compute_training_structure_features(
        read_triples(paths["train"]),
        read_entity_types(paths["entity_types"]),
        spec["modeled_relations"],
        spec["relation_roles"],
    )
    assert len(rows) == 9
    assert {row["relation"] for row in rows} == set(spec["modeled_relations"])
    assert all(int(row["train_triples"]) > 0 for row in rows)


def test_fingerprint_is_key_order_independent() -> None:
    assert stable_fingerprint({"b": 2, "a": 1}) == stable_fingerprint({"a": 1, "b": 2})
