from __future__ import annotations

"""Reciprocal-training utilities for v2.6.3.

PyKEEN 1.11.1 represents a real relation r internally as 2*r when inverse triples are
enabled, and its artificial inverse as 2*r+1. We therefore build deterministic frozen
router features/reliability for both directions while keeping the original 13,693-entity
candidate space and the original Validation triples unchanged.
"""

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .bounded_adaptive_v26 import BoundedRelationFeatureBundle

try:
    from pykeen.training import SLCWATrainingLoop
except ImportError as exc:
    SLCWATrainingLoop = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


DIRECTION_FEATURE_NAME = "direction_is_inverse"


@dataclass(frozen=True)
class ReciprocalFeatureBundle:
    features: Tensor
    reliability: Tensor
    feature_names: tuple[str, ...]
    rows: list[dict[str, Any]]

    def feature_sha256(self) -> str:
        return hashlib.sha256(self.features.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def reliability_sha256(self) -> str:
        return hashlib.sha256(self.reliability.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_reciprocal_feature_bundle(
    *,
    base_bundle: BoundedRelationFeatureBundle,
    relation_to_id: Mapping[str, int],
) -> ReciprocalFeatureBundle:
    """Expand frozen v2.6 features to PyKEEN's forward/inverse internal relation IDs.

    For inverse direction we swap the already frozen TPH/HPT coordinates, because head
    and tail roles reverse. All other Training-only structural coordinates are inherited,
    and a binary direction coordinate is appended. Provenance3 rows remain zero except
    for the direction bit; they are outside the R14 training objective.
    """
    base = base_bundle.features.tensor.detach().cpu()
    reliability_base = base_bundle.reliability.detach().cpu()
    real_num_relations = max(int(v) for v in relation_to_id.values()) + 1
    if base.shape[0] != real_num_relations or reliability_base.shape[0] != real_num_relations:
        raise ValueError("Base v2.6 feature rows must match real relation count")

    feature_dim = int(base.shape[1]) + 1
    features = torch.zeros((2 * real_num_relations, feature_dim), dtype=torch.float32)
    reliability = torch.zeros((2 * real_num_relations, 1), dtype=torch.float32)
    id_to_label = {int(v): str(k) for k, v in relation_to_id.items()}
    rows: list[dict[str, Any]] = []

    # v1.9/v2.6 FEATURE_NAMES positions: z_log_tph=3, z_log_hpt=4
    tph_idx, hpt_idx = 3, 4
    for rid in range(real_num_relations):
        label = id_to_label[rid]
        forward = base[rid].clone()
        inverse = base[rid].clone()
        inverse[tph_idx], inverse[hpt_idx] = base[rid, hpt_idx], base[rid, tph_idx]
        fwd_id, inv_id = 2 * rid, 2 * rid + 1
        features[fwd_id, :-1] = forward
        features[fwd_id, -1] = 0.0
        features[inv_id, :-1] = inverse
        features[inv_id, -1] = 1.0
        reliability[fwd_id] = reliability_base[rid]
        reliability[inv_id] = reliability_base[rid]
        rows.extend([
            {
                "relation": label,
                "real_relation_id": rid,
                "internal_relation_id": fwd_id,
                "direction": "forward",
                "reliability_q": float(reliability[fwd_id, 0]),
            },
            {
                "relation": label,
                "real_relation_id": rid,
                "internal_relation_id": inv_id,
                "direction": "inverse",
                "reliability_q": float(reliability[inv_id, 0]),
            },
        ])
    return ReciprocalFeatureBundle(
        features=features,
        reliability=reliability,
        feature_names=tuple(base_bundle.features.feature_names) + (DIRECTION_FEATURE_NAME,),
        rows=rows,
    )


def reciprocal_training_budget(*, num_real_triples: int, batch_size: int, max_steps: int) -> dict[str, int]:
    """Compute an exact optimizer-step plan after PyKEEN duplicates reciprocal instances."""
    effective_instances = 2 * int(num_real_triples)
    steps_per_full_epoch = math.ceil(effective_instances / int(batch_size))
    full_epochs, remainder = divmod(int(max_steps), steps_per_full_epoch)
    num_epochs = full_epochs + (1 if remainder else 0)
    final_epoch_batches = remainder if remainder else steps_per_full_epoch
    if num_epochs <= 0:
        raise ValueError("Invalid reciprocal training budget")
    return {
        "effective_training_instances": effective_instances,
        "steps_per_full_epoch": steps_per_full_epoch,
        "full_epochs": full_epochs,
        "final_epoch_batches": final_epoch_batches,
        "num_epochs": num_epochs,
        "exact_optimizer_steps": int(max_steps),
    }


class _LimitedBatches:
    """Length-aware iterator wrapper used only for the final partial epoch."""

    def __init__(self, source, limit: int):
        self.source = source
        self.limit = int(limit)

    def __iter__(self):
        yield from itertools.islice(iter(self.source), self.limit)

    def __len__(self):
        return min(len(self.source), self.limit)


if SLCWATrainingLoop is not None:

    class ExactStepReciprocalSLCWATrainingLoop(SLCWATrainingLoop):
        """sLCWA loop that truncates only the last reciprocal epoch to hit exactly N steps."""

        def __init__(
            self,
            *,
            exact_max_steps: int,
            reciprocal_steps_per_full_epoch: int,
            reciprocal_final_epoch_batches: int,
            reciprocal_num_epochs: int,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.exact_max_steps = int(exact_max_steps)
            self.reciprocal_steps_per_full_epoch = int(reciprocal_steps_per_full_epoch)
            self.reciprocal_final_epoch_batches = int(reciprocal_final_epoch_batches)
            self.reciprocal_num_epochs = int(reciprocal_num_epochs)
            self.v263_optimizer_steps = 0

        def _train_epoch(self, *, batches, epoch: int, **kwargs):
            # PyKEEN's epoch index is 1-based for a fresh training run.
            if int(epoch) == self.reciprocal_num_epochs:
                batches = _LimitedBatches(batches, self.reciprocal_final_epoch_batches)
            n_batches = len(batches)
            loss = super()._train_epoch(batches=batches, epoch=epoch, **kwargs)
            if not bool(kwargs.get("only_size_probing", False)):
                self.v263_optimizer_steps += int(n_batches)
            return loss

else:
    ExactStepReciprocalSLCWATrainingLoop = None


def build_exact_step_reciprocal_training_loop_class():
    if ExactStepReciprocalSLCWATrainingLoop is None:
        raise RuntimeError("PyKEEN is not installed in this environment") from _PYKEEN_IMPORT_ERROR
    return ExactStepReciprocalSLCWATrainingLoop
