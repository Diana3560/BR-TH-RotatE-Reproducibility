from __future__ import annotations

"""Dataset-agnostic validation and Training-only relation features.

This module intentionally has no PyTorch or PyKEEN import. It can therefore be
used to audit both packaged datasets before the training environment is ready.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_TOKEN = "THROTATE_MULTIDATASET_COMPARISON_V1"
ALLOWED_RELATION_ROLES = ("diagnostic_semantic", "procedural_structural")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_triples(path: str | Path) -> list[tuple[str, str, str]]:
    path = Path(path)
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, row in enumerate(csv.reader(file, delimiter="\t"), 1):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(f"{path}:{line_number} must contain exactly three TSV columns")
            if any(not value.strip() for value in row):
                raise ValueError(f"{path}:{line_number} contains an empty triple field")
            if line_number == 1 and tuple(value.lower() for value in row) == (
                "head",
                "relation",
                "tail",
            ):
                raise ValueError(f"Triple files must not contain a header: {path}")
            triples.append((row[0], row[1], row[2]))
    if not triples:
        raise ValueError(f"Triple file is empty: {path}")
    return triples


def read_entity_types(path: str | Path) -> dict[str, str]:
    path = Path(path)
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.reader(file, delimiter="\t")
        first = next(rows, None)
        if first is None:
            raise ValueError(f"Entity type file is empty: {path}")
        if len(first) < 2:
            raise ValueError(f"Malformed first row in entity type file: {path}")
        if first[:2] != ["entity_id", "entity_type"]:
            mapping[first[0]] = first[1]
        for line_number, row in enumerate(rows, 2):
            if not row:
                continue
            if len(row) < 2 or not row[0] or not row[1]:
                raise ValueError(f"{path}:{line_number} must contain entity_id and entity_type")
            old = mapping.get(row[0])
            if old is not None and old != row[1]:
                raise ValueError(f"Conflicting types for entity {row[0]} in {path}")
            mapping[row[0]] = row[1]
    return mapping


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get("protocol", {}).get("token") != PROTOCOL_TOKEN:
        raise ValueError(f"Protocol token must be {PROTOCOL_TOKEN}")
    dataset_map = cfg.get("datasets")
    if not isinstance(dataset_map, Mapping) or set(dataset_map) != {"ownkg", "paper4"}:
        raise ValueError("The comparison config must define exactly ownkg and paper4")
    formal = cfg.get("formal", {})
    if tuple(formal.get("models", ())) != ("TransH", "RotatE", "D0", "D1", "A0", "D2"):
        raise ValueError("Formal models must remain TransH/RotatE/D0/D1/A0/D2")
    seeds = tuple(int(value) for value in formal.get("seeds", ()))
    if seeds != (42, 43, 44, 45, 46):
        raise ValueError("Formal seeds must remain 42/43/44/45/46")

    for dataset_key, spec in dataset_map.items():
        modeled = tuple(str(value) for value in spec.get("modeled_relations", ()))
        roles = dict(spec.get("relation_roles", {}))
        if not modeled or len(modeled) != len(set(modeled)):
            raise ValueError(f"{dataset_key}: modeled_relations must be non-empty and unique")
        if set(roles) != set(modeled):
            raise ValueError(f"{dataset_key}: relation_roles must exactly match modeled_relations")
        bad_roles = sorted(set(roles.values()) - set(ALLOWED_RELATION_ROLES))
        if bad_roles:
            raise ValueError(f"{dataset_key}: unsupported relation roles: {bad_roles}")


def dataset_paths(project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str) -> dict[str, Path]:
    project_root = Path(project_root)
    if dataset_key not in cfg["datasets"]:
        raise KeyError(f"Unknown dataset: {dataset_key}")
    spec = cfg["datasets"][dataset_key]
    split_dir = project_root / spec["split_dir"]
    paths = {
        "full": project_root / spec["full_file"],
        "entity_types": project_root / spec["entity_types_file"],
        "train": split_dir / spec["train_filename"],
        "validation": split_dir / spec["validation_filename"],
        "test": split_dir / spec["test_filename"],
    }
    if spec.get("relation_schema_file"):
        paths["relation_schema"] = project_root / spec["relation_schema_file"]
    if spec.get("dataset_stats_file"):
        paths["dataset_stats"] = project_root / spec["dataset_stats_file"]
    return paths


def _entities(triples: Iterable[tuple[str, str, str]]) -> set[str]:
    return {value for head, _relation, tail in triples for value in (head, tail)}


def _relations(triples: Iterable[tuple[str, str, str]]) -> set[str]:
    return {relation for _head, relation, _tail in triples}


def audit_dataset(project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str) -> dict[str, Any]:
    """Perform strict, deterministic integrity checks without constructing a Test model factory."""
    validate_config(cfg)
    spec = cfg["datasets"][dataset_key]
    paths = dataset_paths(project_root, cfg, dataset_key)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset_key}: missing required files: {missing}")

    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected_hashes = {str(key): str(value) for key, value in spec["sha256"].items()}
    if set(expected_hashes) != set(paths):
        raise ValueError(
            f"{dataset_key}: sha256 keys must exactly match required data files; "
            f"expected_keys={sorted(expected_hashes)}, path_keys={sorted(paths)}"
        )
    for name, expected in expected_hashes.items():
        if actual_hashes.get(name) != expected:
            raise ValueError(
                f"{dataset_key}: {name} SHA-256 mismatch; "
                f"expected={expected}, actual={actual_hashes.get(name)}"
            )

    triples = {name: read_triples(paths[name]) for name in ("full", "train", "validation", "test")}
    triple_sets = {name: set(rows) for name, rows in triples.items()}
    duplicate_counts = {name: len(triples[name]) - len(triple_sets[name]) for name in triples}
    if any(duplicate_counts.values()):
        raise ValueError(f"{dataset_key}: duplicate triples detected: {duplicate_counts}")
    overlaps = {
        "train_validation": len(triple_sets["train"] & triple_sets["validation"]),
        "train_test": len(triple_sets["train"] & triple_sets["test"]),
        "validation_test": len(triple_sets["validation"] & triple_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"{dataset_key}: split overlap detected: {overlaps}")

    split_union = triple_sets["train"] | triple_sets["validation"] | triple_sets["test"]
    if not split_union <= triple_sets["full"]:
        raise ValueError(f"{dataset_key}: split union contains triples absent from full_file")

    full_entities = _entities(triples["full"])
    train_entities = _entities(triples["train"])
    split_entities = _entities(split_union)
    unseen_split_entities = sorted(split_entities - train_entities)
    if unseen_split_entities:
        raise ValueError(
            f"{dataset_key}: Validation/Test entities missing from Training: {unseen_split_entities[:20]}"
        )

    modeled_relations = tuple(str(value) for value in spec["modeled_relations"])
    modeled_set = set(modeled_relations)
    if _relations(triples["train"]) != modeled_set:
        raise ValueError(
            f"{dataset_key}: Training relation set does not match modeled_relations; "
            f"training={sorted(_relations(triples['train']))}, configured={sorted(modeled_set)}"
        )
    for name in ("validation", "test"):
        unexpected = sorted(_relations(triples[name]) - modeled_set)
        if unexpected:
            raise ValueError(f"{dataset_key}: unexpected {name} relations: {unexpected}")

    entity_types = read_entity_types(paths["entity_types"])
    missing_types = sorted(full_entities - set(entity_types))
    if missing_types:
        raise ValueError(f"{dataset_key}: entities without a declared type: {missing_types[:20]}")
    extra_types = sorted(set(entity_types) - full_entities)

    expected = spec["expected"]
    counts = {
        "candidate_entities": len(full_entities),
        "mapping_relations": len(_relations(triples["full"])),
        "modeled_relations": len(modeled_set),
        "full_triples": len(triples["full"]),
        "train_triples": len(triples["train"]),
        "validation_triples": len(triples["validation"]),
        "test_triples": len(triples["test"]),
        "split_union_triples": len(split_union),
        "validation_relations": len(_relations(triples["validation"])),
        "test_relations": len(_relations(triples["test"])),
    }
    count_checks = {key: counts[key] == int(expected[key]) for key in counts}
    if not all(count_checks.values()):
        raise ValueError(
            f"{dataset_key}: count audit failed: "
            f"{dict((key, {'actual': counts[key], 'expected': int(expected[key])}) for key in counts if not count_checks[key])}"
        )

    relation_counts = {
        name: dict(sorted(Counter(relation for _head, relation, _tail in triples[name]).items()))
        for name in triples
    }
    return {
        "status": "PASS",
        "dataset_key": dataset_key,
        "display_name": str(spec["display_name"]),
        "identity_note": str(spec.get("identity_note", "")),
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": actual_hashes,
        "counts": counts,
        "count_checks": count_checks,
        "duplicate_counts": duplicate_counts,
        "split_overlaps": overlaps,
        "split_union_is_subset_of_full": True,
        "split_entities_seen_in_training": True,
        "full_entities_outside_modeled_splits": len(full_entities - split_entities),
        "entity_type_rows": len(entity_types),
        "entity_type_classes": len(set(entity_types.values())),
        "typed_entities_outside_full_graph": extra_types,
        "relation_counts": relation_counts,
        "validation_relations": sorted(_relations(triples["validation"])),
        "test_relations": sorted(_relations(triples["test"])),
        "modeled_relations": list(modeled_relations),
        "relation_roles": dict(spec["relation_roles"]),
    }


def mapping_type(tph: float, hpt: float, threshold: float = 1.5) -> str:
    if tph < threshold and hpt < threshold:
        return "1-1"
    if tph >= threshold and hpt < threshold:
        return "1-N"
    if tph < threshold and hpt >= threshold:
        return "N-1"
    return "N-N"


def _inverse_overlap(
    relation_triples: Mapping[str, set[tuple[str, str]]], relation: str, candidate: str
) -> float:
    pairs = relation_triples[relation]
    reverse_candidate = {(tail, head) for head, tail in relation_triples[candidate]}
    return len(pairs & reverse_candidate) / len(pairs) if pairs else 0.0


def compute_training_structure_features(
    training: Sequence[tuple[str, str, str]],
    entity_types: Mapping[str, str],
    relation_order: Sequence[str],
    relation_roles: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Compute the same 9-D D2 inputs for arbitrary configured relation names.

    Every value is derived from Training only. No Validation/Test metrics or
    triples are accepted by this API.
    """
    relation_order = tuple(str(value) for value in relation_order)
    if set(relation_order) != set(relation_roles):
        raise ValueError("relation_order and relation_roles must contain the same relations")
    bad_roles = sorted(set(relation_roles.values()) - set(ALLOWED_RELATION_ROLES))
    if bad_roles:
        raise ValueError(f"Unsupported relation roles: {bad_roles}")

    by_relation: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for head, relation, tail in training:
        if relation not in relation_roles:
            raise ValueError(f"Training contains unconfigured relation: {relation}")
        by_relation[relation].append((head, tail))
    missing = [relation for relation in relation_order if not by_relation[relation]]
    if missing:
        raise ValueError(f"Configured relations absent from Training: {missing}")

    relation_sets = {relation: set(by_relation[relation]) for relation in relation_order}
    rows: list[dict[str, Any]] = []
    for relation in relation_order:
        pairs = by_relation[relation]
        count = len(pairs)
        heads = {head for head, _tail in pairs}
        tails = {tail for _head, tail in pairs}
        tph = count / len(heads)
        hpt = count / len(tails)

        signatures = Counter(
            (entity_types.get(head, "UNKNOWN"), entity_types.get(tail, "UNKNOWN"))
            for head, tail in pairs
        )
        dominant_signature, dominant_count = signatures.most_common(1)[0]
        entropy = -sum(
            (signature_count / count) * math.log(signature_count / count)
            for signature_count in signatures.values()
        )
        entropy = max(0.0, entropy)

        nonself = {(head, tail) for head, tail in pairs if head != tail}
        reversed_nonself = {(tail, head) for head, tail in nonself}
        symmetry_rate = len(nonself & reversed_nonself) / len(nonself) if nonself else 0.0

        best_inverse_relation = ""
        best_inverse_score = 0.0
        for candidate in relation_order:
            if candidate == relation:
                continue
            score = _inverse_overlap(relation_sets, relation, candidate)
            if score > best_inverse_score:
                best_inverse_score = score
                best_inverse_relation = candidate

        rows.append(
            {
                "relation": relation,
                "role": relation_roles[relation],
                "train_triples": count,
                "unique_heads": len(heads),
                "unique_tails": len(tails),
                "tph": tph,
                "hpt": hpt,
                "mapping_type_1p5": mapping_type(tph, hpt),
                "symmetry_rate_nonself": symmetry_rate,
                "best_inverse_relation": best_inverse_relation,
                "best_inverse_score": best_inverse_score,
                "type_signature_count": len(signatures),
                "dominant_head_type": dominant_signature[0],
                "dominant_tail_type": dominant_signature[1],
                "dominant_type_signature_ratio": dominant_count / count,
                "type_signature_entropy": entropy,
            }
        )
    return rows


def write_structure_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        raise ValueError(f"Refusing to write an empty structure feature file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
