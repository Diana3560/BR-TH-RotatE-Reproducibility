from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class PerRelationControlBundle:
    """Fixed one-hot relation identity features for the direct-angle control.

    Each modeled forward/inverse relation state owns exactly one independent scalar
    router weight.  The feature matrix itself is fixed and contains no topology,
    frequency, type-signature, business-role, Validation, or Test information.
    """

    features: Tensor
    reliability: Tensor
    rows: list[dict[str, object]]

    @property
    def trainable_angle_parameters(self) -> int:
        return int(self.features.shape[1])


def build_per_relation_control_bundle(
    *,
    relation_to_id: Mapping[str, int],
    modeled_relations: Sequence[str],
    num_internal_relations: int,
) -> PerRelationControlBundle:
    """Build a one-hot direct per-relation control for reciprocal TH-RotatE.

    For each modeled real relation ``r`` we create two independent internal states,
    matching PyKEEN's reciprocal layout::

        forward(r) -> 2 * real_relation_id
        inverse(r) -> 2 * real_relation_id + 1

    Let ``e_(r,d)`` be the one-hot feature for relation/direction state ``(r,d)``.
    The existing zero-bias linear router then becomes a table lookup of one directly
    learned scalar ``delta_(r,d)``.  With active-state reliability fixed to one, the
    bounded fusion angle is exactly::

        theta_(r,d) = theta_global + Delta * tanh(delta_(r,d))

    followed by the same first-quadrant safety clamp used by BR-TH-RotatE.

    Unmodeled mapping relations (the provenance layer in CRH-L4MKG) receive all-zero
    features and reliability zero. They are not training/evaluation targets in R14.
    """

    modeled = tuple(str(r) for r in modeled_relations)
    if not modeled:
        raise ValueError("modeled_relations must not be empty")
    if len(set(modeled)) != len(modeled):
        raise ValueError("modeled_relations contains duplicates")
    if num_internal_relations <= 0 or num_internal_relations % 2 != 0:
        raise ValueError("num_internal_relations must be a positive even integer")

    missing = [r for r in modeled if r not in relation_to_id]
    if missing:
        raise KeyError(f"Missing modeled relations in relation_to_id: {missing}")

    feature_dim = 2 * len(modeled)
    features = torch.zeros((num_internal_relations, feature_dim), dtype=torch.float32)
    reliability = torch.zeros((num_internal_relations, 1), dtype=torch.float32)
    rows: list[dict[str, object]] = []

    column = 0
    used_internal_ids: set[int] = set()
    for relation in modeled:
        real_id = int(relation_to_id[relation])
        for direction, internal_id in (("forward", 2 * real_id), ("inverse", 2 * real_id + 1)):
            if not 0 <= internal_id < num_internal_relations:
                raise ValueError(
                    f"Internal relation id {internal_id} for {relation}/{direction} is out of range "
                    f"for num_internal_relations={num_internal_relations}"
                )
            if internal_id in used_internal_ids:
                raise ValueError(f"Duplicate internal relation id: {internal_id}")
            used_internal_ids.add(internal_id)
            features[internal_id, column] = 1.0
            reliability[internal_id, 0] = 1.0
            rows.append(
                {
                    "relation": relation,
                    "real_relation_id": real_id,
                    "direction": direction,
                    "internal_relation_id": internal_id,
                    "one_hot_column": column,
                    "reliability": 1.0,
                }
            )
            column += 1

    # Exact construction audit: every active state has one 1, every feature column is
    # used once, and provenance/unmodeled rows remain all zero.
    active = features.sum(dim=1)
    if not torch.all(active[list(sorted(used_internal_ids))] == 1):
        raise RuntimeError("Active relation states are not one-hot")
    if not torch.all(features.sum(dim=0) == 1):
        raise RuntimeError("A direct-angle feature column is not uniquely assigned")
    unused = [i for i in range(num_internal_relations) if i not in used_internal_ids]
    if unused and not torch.all(active[unused] == 0):
        raise RuntimeError("Unmodeled relation states must remain all zero")
    if column != feature_dim:
        raise RuntimeError("Unexpected direct-angle feature dimension")

    return PerRelationControlBundle(features=features, reliability=reliability, rows=rows)
