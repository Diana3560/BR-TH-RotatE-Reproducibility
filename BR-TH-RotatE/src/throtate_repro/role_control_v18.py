from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# v1.8 role taxonomy is frozen from the KG business semantics, NOT from Validation performance.
DIAGNOSTIC_SEMANTIC_RELATIONS = (
    "HAS_FAILURE",
    "DETECTED_BY",
    "LIMITED_BY",
    "FIXED_BY",
    "CAUSED_BY",
    "HAS_SYMPTOM",
    "INDICATES",
    "APPLIES_UNDER",
    "HAS_DISPOSITION",
    "APPLIES_TO",
)

PROCEDURAL_STRUCTURAL_RELATIONS = (
    "HAS_PART",
    "HAS_STEP",
    "REQUIRES",
    "NEXT_STEP",
)

PROVENANCE_DOCUMENT_RELATIONS = (
    "EVIDENCED_BY",
    "HAS_CHUNK",
    "HAS_SUBCHUNK",
)

REASONING14_RELATIONS = DIAGNOSTIC_SEMANTIC_RELATIONS + PROCEDURAL_STRUCTURAL_RELATIONS
FULL17_RELATIONS = REASONING14_RELATIONS + PROVENANCE_DOCUMENT_RELATIONS

RELATION_ROLE = {
    **{r: "diagnostic_semantic" for r in DIAGNOSTIC_SEMANTIC_RELATIONS},
    **{r: "procedural_structural" for r in PROCEDURAL_STRUCTURAL_RELATIONS},
    **{r: "provenance_document" for r in PROVENANCE_DOCUMENT_RELATIONS},
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_triples(path: str | Path) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(f"Expected 3 columns, got {len(row)}: {row[:5]}")
            out.append((row[0], row[1], row[2]))
    return out


def write_triples(path: str | Path, triples: Iterable[tuple[str, str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerows(triples)


def dump_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_bytes(src: str | Path, dst: str | Path) -> None:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def split_reasoning14_relation_stratified_entity_coverage(
    triples: list[tuple[str, str, str]], *, seed: int = 42, holdout_ratio: float = 0.06
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]], dict]:
    """Relation-stratified Reasoning14 split with Reasoning14 entity coverage.

    Why 0.06 holdout per side?
    With the frozen batch_size=1024, the resulting Training set has 16 batches/epoch,
    so 125 epochs equals exactly 2000 optimizer steps. This preserves the Paper4/OwnKG
    training-budget control without changing batch size.

    Test is created and hashed, but must remain sealed during development.
    """
    observed = {r for _, r, _ in triples}
    if observed != set(REASONING14_RELATIONS):
        raise ValueError(f"Unexpected Reasoning14 relation set: {sorted(observed)}")

    rng = random.Random(seed)
    train_idx = set(range(len(triples)))
    val_idx: list[int] = []
    test_idx: list[int] = []

    entity_degree = Counter()
    relation_degree = Counter()
    by_relation: dict[str, list[int]] = defaultdict(list)
    for i, (h, r, t) in enumerate(triples):
        entity_degree[h] += 1
        entity_degree[t] += 1
        relation_degree[r] += 1
        by_relation[r].append(i)

    per_relation: dict[str, dict] = {}
    for relation in REASONING14_RELATIONS:
        candidates = list(by_relation[relation])
        rng.shuffle(candidates)
        n = len(candidates)
        # Keep at least a tiny holdout for small-but-important relations.
        target_val = max(1, round(n * holdout_ratio)) if n >= 10 else 0
        target_test = max(1, round(n * holdout_ratio)) if n >= 10 else 0
        got_val = 0
        got_test = 0

        for i in candidates:
            if got_val >= target_val and got_test >= target_test:
                break
            h, r, t = triples[i]
            # A held-out triple must not make either endpoint unseen in Reasoning14 Training.
            if entity_degree[h] <= 1 or entity_degree[t] <= 1 or relation_degree[r] <= 1:
                continue

            val_fill = got_val / max(1, target_val)
            test_fill = got_test / max(1, target_test)
            choose_validation = val_fill <= test_fill
            if got_val >= target_val:
                choose_validation = False
            if got_test >= target_test:
                choose_validation = True

            train_idx.remove(i)
            if choose_validation:
                val_idx.append(i)
                got_val += 1
            else:
                test_idx.append(i)
                got_test += 1
            entity_degree[h] -= 1
            entity_degree[t] -= 1
            relation_degree[r] -= 1

        train_count = sum(1 for i in train_idx if triples[i][1] == relation)
        per_relation[relation] = {
            "role": RELATION_ROLE[relation],
            "total": n,
            "training": train_count,
            "validation": got_val,
            "testing": got_test,
            "target_validation": target_val,
            "target_testing": target_test,
            "coverage_limited": got_val < target_val or got_test < target_test,
        }

    if len(val_idx) < len(test_idx):
        val_idx, test_idx = test_idx, val_idx
        for row in per_relation.values():
            row["validation"], row["testing"] = row["testing"], row["validation"]

    train = [triples[i] for i in sorted(train_idx)]
    valid = [triples[i] for i in sorted(val_idx)]
    test = [triples[i] for i in sorted(test_idx)]

    all_entities = {x for h, _, t in triples for x in (h, t)}
    train_entities = {x for h, _, t in train for x in (h, t)}
    if train_entities != all_entities:
        missing = sorted(all_entities - train_entities)
        raise ValueError(f"Reasoning14 entity coverage failed: missing {missing[:20]}")
    if set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test):
        raise ValueError("Reasoning14 split overlap detected")
    if set(train) | set(valid) | set(test) != set(triples):
        raise ValueError("Reasoning14 split union mismatch")

    manifest = {
        "seed": seed,
        "split_method": "reasoning14_relation_stratified_plus_entity_coverage_v18",
        "holdout_ratio_per_side": holdout_ratio,
        "counts": {"training": len(train), "validation": len(valid), "testing": len(test)},
        "actual_ratios": [len(train) / len(triples), len(valid) / len(triples), len(test) / len(triples)],
        "reasoning14_entity_count": len(all_entities),
        "all_reasoning14_entities_seen_in_training": True,
        "all_14_relations_seen_in_training": {r for _, r, _ in train} == set(REASONING14_RELATIONS),
        "validation_contains_all_14_relations": {r for _, r, _ in valid} == set(REASONING14_RELATIONS),
        "test_contains_all_14_relations": {r for _, r, _ in test} == set(REASONING14_RELATIONS),
        "per_relation": per_relation,
        "note": (
            "APPLIES_UNDER is coverage-limited because many low-degree entities must remain in Training. "
            "The 0.06 holdout ratio is predeclared to preserve exactly 2000 optimizer steps with batch_size=1024."
        ),
    }
    return train, valid, test, manifest


def read_entity_types(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"entity_types row must have >=2 columns: {row}")
            mapping[row[0]] = row[1]
    return mapping


def mapping_type(tph: float, hpt: float, threshold: float = 1.5) -> str:
    if tph < threshold and hpt < threshold:
        return "1-1"
    if tph >= threshold and hpt < threshold:
        return "1-N"
    if tph < threshold and hpt >= threshold:
        return "N-1"
    return "N-N"


def _inverse_overlap(
    relation_triples: dict[str, set[tuple[str, str]]], relation: str, candidate: str
) -> float:
    a = relation_triples[relation]
    b_rev = {(t, h) for h, t in relation_triples[candidate]}
    if not a:
        return 0.0
    return len(a & b_rev) / len(a)


def compute_training_structure_features(
    training: list[tuple[str, str, str]], entity_types: dict[str, str]
) -> list[dict]:
    by_relation: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for h, r, t in training:
        by_relation[r].append((h, t))

    relation_sets = {r: set(pairs) for r, pairs in by_relation.items()}
    rows: list[dict] = []
    for r in REASONING14_RELATIONS:
        pairs = by_relation[r]
        n = len(pairs)
        heads = {h for h, _ in pairs}
        tails = {t for _, t in pairs}
        tph = n / len(heads)
        hpt = n / len(tails)

        sig = Counter((entity_types.get(h, "UNKNOWN"), entity_types.get(t, "UNKNOWN")) for h, t in pairs)
        dominant_sig, dominant_count = sig.most_common(1)[0]
        sig_entropy = 0.0
        for count in sig.values():
            p = count / n
            sig_entropy -= p * math.log(p)
        sig_entropy = max(0.0, sig_entropy)

        self_pairs = {(h, t) for h, t in pairs if h != t}
        reverse_pairs = {(t, h) for h, t in self_pairs}
        symmetry_rate = len(self_pairs & reverse_pairs) / len(self_pairs) if self_pairs else 0.0

        best_inverse = None
        best_inverse_score = 0.0
        for candidate in REASONING14_RELATIONS:
            if candidate == r:
                continue
            score = _inverse_overlap(relation_sets, r, candidate)
            if score > best_inverse_score:
                best_inverse_score = score
                best_inverse = candidate

        rows.append({
            "relation": r,
            "role": RELATION_ROLE[r],
            "train_triples": n,
            "unique_heads": len(heads),
            "unique_tails": len(tails),
            "tph": tph,
            "hpt": hpt,
            "mapping_type_1p5": mapping_type(tph, hpt),
            "symmetry_rate_nonself": symmetry_rate,
            "best_inverse_relation": best_inverse or "",
            "best_inverse_score": best_inverse_score,
            "type_signature_count": len(sig),
            "dominant_head_type": dominant_sig[0],
            "dominant_tail_type": dominant_sig[1],
            "dominant_type_signature_ratio": dominant_count / n,
            "type_signature_entropy": sig_entropy,
        })
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
