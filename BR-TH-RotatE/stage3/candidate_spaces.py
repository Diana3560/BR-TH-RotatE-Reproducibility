from __future__ import annotations

"""Candidate-space construction and audits for Stage 3.

Important leakage guard:
- type constraints are derived from TRAINING triples only;
- Validation/Test are used only to audit that every gold target remains admissible;
- standard filtered evaluation may use Train+Validation+Test truths to remove other
  known-positive candidates, exactly as in the paper's Both-sides Filtered protocol.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from throtate_repro.multidataset_data import dataset_paths, read_entity_types, read_triples


Triple = tuple[str, str, str]


@dataclass(frozen=True)
class CandidateArtifacts:
    full_entities: frozenset[str]
    modeled_entities: frozenset[str]
    type_schema: Mapping[str, Mapping[str, tuple[str, ...]]]
    entities_by_type: Mapping[str, frozenset[str]]
    train: tuple[Triple, ...]
    validation: tuple[Triple, ...]
    test: tuple[Triple, ...]


def _entities(triples: Iterable[Triple]) -> set[str]:
    return {x for h, _r, t in triples for x in (h, t)}


def derive_training_type_schema(
    train: Sequence[Triple],
    entity_types: Mapping[str, str],
    modeled_relations: Sequence[str],
) -> dict[str, dict[str, tuple[str, ...]]]:
    schema: dict[str, dict[str, set[str]]] = {
        relation: {"head_types": set(), "tail_types": set()} for relation in modeled_relations
    }
    for h, r, t in train:
        if r not in schema:
            raise ValueError(f"Training triple contains relation outside modeled set: {r}")
        try:
            ht = entity_types[h]
            tt = entity_types[t]
        except KeyError as exc:
            raise ValueError(f"Missing entity type for training triple {(h, r, t)}") from exc
        schema[r]["head_types"].add(ht)
        schema[r]["tail_types"].add(tt)
    missing = [r for r, x in schema.items() if not x["head_types"] or not x["tail_types"]]
    if missing:
        raise ValueError(f"Modeled relations absent from training schema: {missing}")
    return {
        r: {
            "head_types": tuple(sorted(x["head_types"])),
            "tail_types": tuple(sorted(x["tail_types"])),
        }
        for r, x in sorted(schema.items())
    }


def build_candidate_artifacts(project_root: str | Path, base_cfg: Mapping[str, Any], dataset_key: str) -> CandidateArtifacts:
    root = Path(project_root).resolve()
    paths = dataset_paths(root, base_cfg, dataset_key)
    full = tuple(read_triples(paths["full"]))
    train = tuple(read_triples(paths["train"]))
    validation = tuple(read_triples(paths["validation"]))
    test = tuple(read_triples(paths["test"]))
    entity_types = read_entity_types(paths["entity_types"])
    modeled_relations = tuple(str(x) for x in base_cfg["datasets"][dataset_key]["modeled_relations"])
    type_schema = derive_training_type_schema(train, entity_types, modeled_relations)
    entities_by_type: dict[str, set[str]] = defaultdict(set)
    for entity, entity_type in entity_types.items():
        entities_by_type[entity_type].add(entity)
    return CandidateArtifacts(
        full_entities=frozenset(_entities(full)),
        modeled_entities=frozenset(_entities((*train, *validation, *test))),
        type_schema=type_schema,
        entities_by_type={k: frozenset(v) for k, v in sorted(entities_by_type.items())},
        train=train,
        validation=validation,
        test=test,
    )


def audit_type_schema_gold_coverage(artifacts: CandidateArtifacts, entity_types: Mapping[str, str]) -> dict[str, Any]:
    violations = []
    for split_name, triples in (("validation", artifacts.validation), ("test", artifacts.test)):
        for h, r, t in triples:
            schema = artifacts.type_schema[r]
            ht = entity_types[h]
            tt = entity_types[t]
            if ht not in schema["head_types"] or tt not in schema["tail_types"]:
                violations.append(
                    {
                        "split": split_name,
                        "triple": [h, r, t],
                        "head_type": ht,
                        "tail_type": tt,
                        "allowed_head_types": list(schema["head_types"]),
                        "allowed_tail_types": list(schema["tail_types"]),
                    }
                )
    return {"status": "PASS" if not violations else "FAIL", "violations": violations}


def allowed_entities_for_type_side(
    artifacts: CandidateArtifacts,
    relation: str,
    side: str,
) -> frozenset[str]:
    if side not in {"head", "tail"}:
        raise ValueError(side)
    key = f"{side}_types"
    types = artifacts.type_schema[relation][key]
    values: set[str] = set()
    for entity_type in types:
        values.update(artifacts.entities_by_type.get(entity_type, ()))
    # The full graph defines the actual ranking universe.
    values.intersection_update(artifacts.full_entities)
    return frozenset(values)


def candidate_set_for_query(
    artifacts: CandidateArtifacts,
    *,
    space: str,
    relation: str,
    side: str,
) -> frozenset[str]:
    if space == "full":
        return artifacts.full_entities
    if space == "modeled_entities":
        return artifacts.modeled_entities
    if space == "type_constrained":
        return allowed_entities_for_type_side(artifacts, relation, side)
    raise ValueError(f"Unknown candidate space: {space}")


def _truth_indices(all_true: Sequence[Triple]):
    tails: dict[tuple[str, str], set[str]] = defaultdict(set)
    heads: dict[tuple[str, str], set[str]] = defaultdict(set)
    for h, r, t in all_true:
        tails[(h, r)].add(t)
        heads[(r, t)].add(h)
    return heads, tails


def filtered_candidate_counts(artifacts: CandidateArtifacts, *, space: str) -> list[int]:
    """Exact candidate counts for each Test head/tail query after positive filtering."""
    all_true = (*artifacts.train, *artifacts.validation, *artifacts.test)
    known_heads, known_tails = _truth_indices(all_true)
    counts: list[int] = []
    for h, r, t in artifacts.test:
        head_candidates = candidate_set_for_query(artifacts, space=space, relation=r, side="head")
        tail_candidates = candidate_set_for_query(artifacts, space=space, relation=r, side="tail")
        if h not in head_candidates:
            raise ValueError(f"Gold head excluded by {space}: {(h, r, t)}")
        if t not in tail_candidates:
            raise ValueError(f"Gold tail excluded by {space}: {(h, r, t)}")
        filtered_heads = len((known_heads[(r, t)] & set(head_candidates)) - {h})
        filtered_tails = len((known_tails[(h, r)] & set(tail_candidates)) - {t})
        counts.append(len(head_candidates) - filtered_heads)
        counts.append(len(tail_candidates) - filtered_tails)
    return counts


def nominal_candidate_counts(artifacts: CandidateArtifacts, *, space: str) -> list[int]:
    counts: list[int] = []
    for _h, r, _t in artifacts.test:
        counts.append(len(candidate_set_for_query(artifacts, space=space, relation=r, side="head")))
        counts.append(len(candidate_set_for_query(artifacts, space=space, relation=r, side="tail")))
    return counts


def summarize_counts(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("No candidate counts")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_ranks": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def adjusted_mean_rank_index(*, mean_rank: float, filtered_candidate_counts_values: Sequence[int]) -> float:
    """Candidate-count adjusted arithmetic mean rank index (larger is better).

    For each query with n candidates, a uniform random ranking has expected rank
    (n+1)/2. We average these expected ranks over the evaluated queries and normalize
    observed MR so that 1.0 is perfect and 0.0 is random expectation.
    """
    if not filtered_candidate_counts_values:
        raise ValueError("Candidate counts are required for AMRI")
    expected_mr = float(np.mean([(n + 1.0) / 2.0 for n in filtered_candidate_counts_values]))
    if expected_mr <= 1.0:
        return 1.0 if abs(float(mean_rank) - 1.0) < 1.0e-12 else float("nan")
    return 1.0 - (float(mean_rank) - 1.0) / (expected_mr - 1.0)


def type_schema_rows(artifacts: CandidateArtifacts) -> list[dict[str, Any]]:
    rows = []
    for relation, spec in artifacts.type_schema.items():
        head_entities = allowed_entities_for_type_side(artifacts, relation, "head")
        tail_entities = allowed_entities_for_type_side(artifacts, relation, "tail")
        rows.append(
            {
                "relation": relation,
                "head_types": "|".join(spec["head_types"]),
                "tail_types": "|".join(spec["tail_types"]),
                "head_candidate_entities": len(head_entities),
                "tail_candidate_entities": len(tail_entities),
            }
        )
    return rows
