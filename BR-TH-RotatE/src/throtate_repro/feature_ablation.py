from __future__ import annotations

"""Small, auditable feature-ablation helpers.

These helpers deliberately operate on already-built frozen relation-feature tensors.
They do not recompute relation statistics and do not change model architecture,
parameter count, training budget, or candidate-space definitions.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


BUSINESS_ROLE_FEATURES = ("role_diagnostic", "role_procedural")


@dataclass(frozen=True)
class FeatureMaskAudit:
    removed_feature_names: tuple[str, ...]
    removed_feature_indices: tuple[int, ...]
    original_shape: tuple[int, ...]
    masked_shape: tuple[int, ...]
    nonzero_removed_before: int
    nonzero_removed_after: int
    unchanged_other_columns: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "removed_feature_names": list(self.removed_feature_names),
            "removed_feature_indices": list(self.removed_feature_indices),
            "original_shape": list(self.original_shape),
            "masked_shape": list(self.masked_shape),
            "nonzero_removed_before": int(self.nonzero_removed_before),
            "nonzero_removed_after": int(self.nonzero_removed_after),
            "unchanged_other_columns": bool(self.unchanged_other_columns),
        }


def zero_named_feature_columns(
    tensor: Tensor,
    feature_names: Sequence[str],
    names_to_remove: Sequence[str],
) -> tuple[Tensor, FeatureMaskAudit]:
    """Return a clone with selected feature columns zeroed.

    Keeping the tensor width unchanged is intentional: the adapter architecture and
    trainable parameter count remain exactly the same. Only the information carried by
    the selected feature coordinates is removed.
    """
    if tensor.ndim != 2:
        raise ValueError("feature tensor must be two-dimensional")
    if len(feature_names) != int(tensor.shape[1]):
        raise ValueError("feature_names length does not match tensor width")

    name_to_index = {str(name): i for i, name in enumerate(feature_names)}
    missing = [str(name) for name in names_to_remove if str(name) not in name_to_index]
    if missing:
        raise KeyError(f"Missing feature names: {missing}")
    indices = tuple(name_to_index[str(name)] for name in names_to_remove)
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate feature names requested for removal")

    before = tensor.detach().clone()
    masked = before.clone()
    if indices:
        masked[:, list(indices)] = 0.0

    keep = [i for i in range(int(tensor.shape[1])) if i not in set(indices)]
    unchanged = True
    if keep:
        unchanged = bool(torch.equal(masked[:, keep], before[:, keep]))

    audit = FeatureMaskAudit(
        removed_feature_names=tuple(str(name) for name in names_to_remove),
        removed_feature_indices=indices,
        original_shape=tuple(int(v) for v in tensor.shape),
        masked_shape=tuple(int(v) for v in masked.shape),
        nonzero_removed_before=int(torch.count_nonzero(before[:, list(indices)]).item()) if indices else 0,
        nonzero_removed_after=int(torch.count_nonzero(masked[:, list(indices)]).item()) if indices else 0,
        unchanged_other_columns=unchanged,
    )
    return masked, audit


def remove_business_role_features(
    tensor: Tensor,
    feature_names: Sequence[str],
) -> tuple[Tensor, FeatureMaskAudit]:
    """Zero the two manually defined business-role coordinates only."""
    return zero_named_feature_columns(tensor, feature_names, BUSINESS_ROLE_FEATURES)
