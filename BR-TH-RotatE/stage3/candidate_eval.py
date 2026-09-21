from __future__ import annotations

"""Exact target-side candidate-restricted Both-sides Filtered evaluator.

This module intentionally has no PyKEEN import. It only assumes the supplied model
implements PyKEEN-compatible ``predict_h`` / ``predict_t`` methods and exposes a
``device`` attribute. Keeping this evaluator PyTorch-only makes the ranking logic easy
to unit-test independently of the full training stack.
"""

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from stage3.candidate_spaces import (
    CandidateArtifacts,
    adjusted_mean_rank_index,
    candidate_set_for_query,
    filtered_candidate_counts,
    nominal_candidate_counts,
    summarize_counts,
)


CANDIDATE_SPACES = ("full", "modeled_entities", "type_constrained")


def metric_values_from_ranks(ranks: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(ranks, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("No ranks were produced")
    if not np.all(np.isfinite(arr)) or np.any(arr < 1.0):
        raise ValueError("Rank vector contains non-finite values or ranks < 1")
    return {
        "mrr": float(np.mean(1.0 / arr)),
        "mr": float(np.mean(arr)),
        "hits_at_1": float(np.mean(arr <= 1.0)),
        "hits_at_3": float(np.mean(arr <= 3.0)),
        "hits_at_10": float(np.mean(arr <= 10.0)),
    }


def realistic_rank_from_scores(scores: torch.Tensor, target_index: int) -> float:
    """Return realistic rank: mean of optimistic and pessimistic ranks."""
    target_score = scores[int(target_index)]
    if not torch.isfinite(target_score):
        raise RuntimeError("Gold target received a non-finite score after candidate/filter masking")
    optimistic = 1 + int(torch.count_nonzero(scores > target_score).item())
    pessimistic = int(torch.count_nonzero(scores >= target_score).item())
    return 0.5 * (optimistic + pessimistic)


def _truth_indices(artifacts: CandidateArtifacts):
    known_tails: dict[tuple[str, str], set[str]] = defaultdict(set)
    known_heads: dict[tuple[str, str], set[str]] = defaultdict(set)
    for h, r, t in (*artifacts.train, *artifacts.validation, *artifacts.test):
        known_tails[(h, r)].add(t)
        known_heads[(r, t)].add(h)
    return known_heads, known_tails


def _relation_test_rows(artifacts: CandidateArtifacts):
    rows: dict[str, list[tuple[int, tuple[str, str, str]]]] = defaultdict(list)
    for i, triple in enumerate(artifacts.test):
        rows[triple[1]].append((i, triple))
    return dict(rows)


def _entities_to_ids(entity_names: Iterable[str], entity_to_id: Mapping[str, int]) -> list[int]:
    values = []
    for name in entity_names:
        if name not in entity_to_id:
            raise KeyError(f"Entity is absent from full mapping: {name}")
        values.append(int(entity_to_id[name]))
    return sorted(set(values))


def _mapped_test_by_relation(context: Mapping[str, Any], artifacts: CandidateArtifacts):
    mapping = context["full"]
    rows: dict[str, list[list[int]]] = defaultdict(list)
    for h, r, t in artifacts.test:
        rows[r].append([
            int(mapping.entity_to_id[h]),
            int(mapping.relation_to_id[r]),
            int(mapping.entity_to_id[t]),
        ])
    return {r: torch.as_tensor(v, dtype=torch.long) for r, v in rows.items()}


def _candidate_mask(
    *,
    artifacts: CandidateArtifacts,
    relation: str,
    side: str,
    space: str,
    entity_to_id: Mapping[str, int],
    num_entities: int,
    device: torch.device,
):
    allowed_names = candidate_set_for_query(
        artifacts,
        space=space,
        relation=relation,
        side=side,
    )
    ids = _entities_to_ids(allowed_names, entity_to_id)
    mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
    if ids:
        mask[torch.as_tensor(ids, dtype=torch.long, device=device)] = True
    return mask, ids


def evaluate_candidate_space(
    *,
    model,
    context: Mapping[str, Any],
    artifacts: CandidateArtifacts,
    space: str,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate exact both-side filtered ranks under one candidate-space definition.

    PyKEEN's global ``restrict_entities_to`` cannot represent a relation whose legal
    head set differs from its legal tail set. Therefore this evaluator obtains scores
    with ``predict_h`` / ``predict_t`` and applies separate target-side masks.

    ``predict_h`` is essential here: for models trained with reciprocal triples, PyKEEN
    automatically turns head prediction into tail prediction under the inverse relation.
    Calling raw ``score_h`` would not reproduce that evaluation behavior.
    """
    if space not in CANDIDATE_SPACES:
        raise ValueError(space)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model.eval()
    device = model.device
    full = context["full"]
    entity_to_id = full.entity_to_id
    num_entities = int(full.num_entities)
    mapped_by_relation = _mapped_test_by_relation(context, artifacts)
    raw_by_relation = _relation_test_rows(artifacts)
    known_heads, known_tails = _truth_indices(artifacts)

    ranks: list[float] = []
    actual_candidate_counts: list[int] = []
    relation_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for relation in sorted(raw_by_relation):
            raw_rows = raw_by_relation[relation]
            mapped = mapped_by_relation[relation]
            if int(mapped.shape[0]) != len(raw_rows):
                raise RuntimeError(f"Mapped/raw Test count mismatch for relation={relation}")

            relation_rank_start = len(ranks)
            for side in ("head", "tail"):
                allowed_mask, allowed_ids = _candidate_mask(
                    artifacts=artifacts,
                    relation=relation,
                    side=side,
                    space=space,
                    entity_to_id=entity_to_id,
                    num_entities=num_entities,
                    device=device,
                )
                if not allowed_ids:
                    raise RuntimeError(f"Empty candidate set: relation={relation}, side={side}, space={space}")

                side_ranks: list[float] = []
                side_counts: list[int] = []
                for start in range(0, int(mapped.shape[0]), batch_size):
                    end = min(start + batch_size, int(mapped.shape[0]))
                    hrt = mapped[start:end].to(device=device)
                    if side == "tail":
                        scores = model.predict_t(hrt[:, :2])
                    else:
                        scores = model.predict_h(hrt[:, 1:])
                    if scores.ndim != 2 or int(scores.shape[1]) != num_entities:
                        raise RuntimeError(
                            f"Unexpected {side} score shape={tuple(scores.shape)}; expected (*,{num_entities})"
                        )
                    scores = scores.clone()
                    scores[:, ~allowed_mask] = -torch.inf

                    for local_i in range(end - start):
                        _raw_index, (h, r, t) = raw_rows[start + local_i]
                        target_name = h if side == "head" else t
                        target_id = int(entity_to_id[target_name])
                        if not bool(allowed_mask[target_id].item()):
                            raise RuntimeError(
                                f"Gold target excluded: space={space}, relation={r}, side={side}, entity={target_name}"
                            )

                        if side == "tail":
                            other_true_names = known_tails[(h, r)] - {t}
                        else:
                            other_true_names = known_heads[(r, t)] - {h}
                        filtered_ids = [
                            int(entity_to_id[name])
                            for name in other_true_names
                            if name in entity_to_id and bool(allowed_mask[int(entity_to_id[name])].item())
                        ]
                        if filtered_ids:
                            scores[local_i, torch.as_tensor(filtered_ids, dtype=torch.long, device=device)] = -torch.inf

                        count = len(allowed_ids) - len(filtered_ids)
                        if count < 1:
                            raise RuntimeError("Filtered candidate count became < 1")
                        rank = realistic_rank_from_scores(scores[local_i], target_id)
                        if rank > count + 1.0e-9:
                            raise RuntimeError(
                                f"Rank exceeds candidate count: rank={rank}, count={count}, relation={r}, side={side}"
                            )
                        side_ranks.append(rank)
                        side_counts.append(count)
                        ranks.append(rank)
                        actual_candidate_counts.append(count)

                side_metrics = metric_values_from_ranks(side_ranks)
                side_amri = adjusted_mean_rank_index(
                    mean_rank=side_metrics["mr"],
                    filtered_candidate_counts_values=side_counts,
                )
                relation_rows.append(
                    {
                        "relation": relation,
                        "side": side,
                        "space": space,
                        "n_ranks": len(side_ranks),
                        **side_metrics,
                        "amri": side_amri,
                        **{
                            f"candidate_{k}": v
                            for k, v in summarize_counts(side_counts).items()
                            if k != "n_ranks"
                        },
                    }
                )

            if len(ranks) - relation_rank_start != 2 * len(raw_rows):
                raise RuntimeError(f"Both-side relation rank-count audit failed for relation={relation}")

    metrics = metric_values_from_ranks(ranks)
    metrics["amri"] = adjusted_mean_rank_index(
        mean_rank=metrics["mr"],
        filtered_candidate_counts_values=actual_candidate_counts,
    )

    expected_counts = filtered_candidate_counts(artifacts, space=space)
    if sorted(actual_candidate_counts) != sorted(expected_counts):
        raise RuntimeError(
            f"Candidate-count audit mismatch for {space}: evaluator and independent data audit differ"
        )
    expected_rank_count = 2 * len(artifacts.test)
    if len(ranks) != expected_rank_count:
        raise RuntimeError(f"Expected {expected_rank_count} ranks, got {len(ranks)}")

    return {
        "status": "PASS",
        "space": space,
        "metrics": metrics,
        "rank_count": len(ranks),
        "nominal_candidate_count_summary": summarize_counts(
            nominal_candidate_counts(artifacts, space=space)
        ),
        "filtered_candidate_count_summary": summarize_counts(actual_candidate_counts),
        "per_relation_side": relation_rows,
    }
