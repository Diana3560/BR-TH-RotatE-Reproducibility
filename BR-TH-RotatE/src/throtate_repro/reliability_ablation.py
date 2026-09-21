from __future__ import annotations

"""Utilities for the q_r=1 no-shrinkage ablation.

The final BR-TH-RotatE D2 model uses Training-only frequency reliability

    q_r = sqrt(N_r / max_s N_s)

outside the router tanh.  The no-shrinkage control keeps every other part of D2
unchanged and sets q_r=1 only for the modeled R14 relations (forward and inverse
internal directions).  Unused provenance-relation rows remain at q=0.
"""

import hashlib
from typing import Any, Iterable, Mapping

import torch

from .reciprocal_v263 import ReciprocalFeatureBundle


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_no_shrinkage_reciprocal_bundle(
    *,
    bundle: ReciprocalFeatureBundle,
    relation_to_id: Mapping[str, int],
    modeled_relations: Iterable[str],
) -> tuple[ReciprocalFeatureBundle, dict[str, Any]]:
    """Return an otherwise identical reciprocal bundle with modeled q_r set to 1.

    PyKEEN 1.11.1 uses internal relation IDs ``2*r`` (forward) and ``2*r+1``
    (inverse) when reciprocal training is enabled.  Only the modeled R14 relation
    rows are changed.  The feature tensor is cloned without modification and the
    remaining reliability rows are left untouched (the provenance-only rows are
    expected to be zero in the current CRH-L4MKG workflow).
    """

    features = bundle.features.detach().clone()
    original = bundle.reliability.detach().clone()
    modified = original.clone()

    modeled = tuple(str(r) for r in modeled_relations)
    active_ids: list[int] = []
    active_rows: list[dict[str, Any]] = []
    for relation in modeled:
        if relation not in relation_to_id:
            raise ValueError(f"Modeled relation missing from relation mapping: {relation}")
        real_id = int(relation_to_id[relation])
        for direction, internal_id in (("forward", 2 * real_id), ("inverse", 2 * real_id + 1)):
            if internal_id >= int(modified.shape[0]):
                raise ValueError(
                    f"Internal reciprocal relation id {internal_id} is outside reliability tensor"
                )
            q_before = float(original[internal_id, 0])
            if not (0.0 < q_before <= 1.0):
                raise ValueError(
                    f"Expected positive Training-only reliability for {relation}/{direction}; got {q_before}"
                )
            modified[internal_id, 0] = 1.0
            active_ids.append(internal_id)
            active_rows.append(
                {
                    "relation": relation,
                    "real_relation_id": real_id,
                    "internal_relation_id": internal_id,
                    "direction": direction,
                    "original_reliability_q": q_before,
                    "no_shrinkage_reliability_q": 1.0,
                }
            )

    active_id_set = set(active_ids)
    if len(active_id_set) != len(active_ids):
        raise ValueError("Duplicate internal relation ids in modeled no-shrinkage rows")

    # All non-modeled rows must remain bitwise/equality unchanged.
    all_ids = set(range(int(original.shape[0])))
    inactive_ids = sorted(all_ids - active_id_set)
    if inactive_ids:
        inactive_index = torch.as_tensor(inactive_ids, dtype=torch.long)
        if not torch.equal(original[inactive_index], modified[inactive_index]):
            raise RuntimeError("No-shrinkage ablation modified non-modeled reliability rows")

    updated_rows: list[dict[str, Any]] = []
    for row in bundle.rows:
        copied = dict(row)
        internal_id = int(copied["internal_relation_id"])
        copied["original_reliability_q"] = float(original[internal_id, 0])
        copied["reliability_q"] = float(modified[internal_id, 0])
        updated_rows.append(copied)

    no_shrinkage = ReciprocalFeatureBundle(
        features=features,
        reliability=modified,
        feature_names=bundle.feature_names,
        rows=updated_rows,
    )

    original_active = [float(original[i, 0]) for i in active_ids]
    audit = {
        "ablation": "D2-noShrinkage",
        "rule": "q_r=1 for every modeled R14 forward/inverse relation; all other rows unchanged",
        "modeled_relation_count": len(modeled),
        "modified_internal_relation_rows": len(active_ids),
        "expected_modified_internal_relation_rows": 2 * len(modeled),
        "modified_internal_relation_ids": active_ids,
        "inactive_internal_relation_rows": len(inactive_ids),
        "original_modeled_q_min": min(original_active),
        "original_modeled_q_max": max(original_active),
        "modified_modeled_q_unique": sorted({float(modified[i, 0]) for i in active_ids}),
        "feature_tensor_sha256_before": bundle.feature_sha256(),
        "feature_tensor_sha256_after": no_shrinkage.feature_sha256(),
        "reliability_sha256_before": bundle.reliability_sha256(),
        "reliability_sha256_after": no_shrinkage.reliability_sha256(),
        "feature_tensor_unchanged": bool(torch.equal(bundle.features, no_shrinkage.features)),
        "non_modeled_reliability_unchanged": True,
        "active_rows": active_rows,
    }
    if audit["feature_tensor_sha256_before"] != audit["feature_tensor_sha256_after"]:
        raise RuntimeError("Feature tensor changed in no-shrinkage ablation")
    if audit["modified_modeled_q_unique"] != [1.0]:
        raise RuntimeError("No-shrinkage modeled reliability rows are not all q_r=1")
    return no_shrinkage, audit
