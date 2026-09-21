from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import normalize_last_dim, rotate_distance, transh_distance

try:
    from pykeen.losses import NSSALoss, UnsupportedLabelSmoothingError, prepare_negative_scores_for_softmax
    from pykeen.models import ERModel
    from pykeen.nn.init import PretrainedInitializer, init_phases, xavier_uniform_
    from pykeen.nn.modules import Interaction
    from pykeen.triples.weights import RelationLossWeighter
    from pykeen.utils import complex_normalize
except ImportError as exc:  # Keep feature/router math importable without PyKEEN.
    NSSALoss = ERModel = Interaction = RelationLossWeighter = None
    UnsupportedLabelSmoothingError = prepare_negative_scores_for_softmax = None
    PretrainedInitializer = init_phases = xavier_uniform_ = complex_normalize = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


FEATURE_NAMES = (
    "role_diagnostic",
    "role_procedural",
    "z_log_train_frequency",
    "z_log_tph",
    "z_log_hpt",
    "z_log_type_signature_count",
    "z_type_signature_entropy",
    "z_dominant_type_signature_ratio",
    "z_best_inverse_score",
)

FORBIDDEN_STRUCTURE_COLUMNS = {
    "transh_validation_tail_mrr_mean",
    "rotate_validation_tail_mrr_mean",
    "throtate_validation_tail_mrr_mean",
    "delta_transh_minus_rotate",
    "best_baseline",
}


@dataclass(frozen=True)
class RelationFeatureBundle:
    tensor: Tensor
    feature_names: tuple[str, ...]
    rows: list[dict[str, Any]]
    normalization: dict[str, dict[str, float]]
    relation_to_id: dict[str, int]

    def sha256(self) -> str:
        payload = self.tensor.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _zscore(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(var)
    if std < 1.0e-12:
        return [0.0 for _ in values], mean, 0.0
    return [(x - mean) / std for x in values], mean, std


def build_relation_feature_bundle(
    *,
    structure_csv: str | Path,
    relation_to_id: Mapping[str, int],
    allowed_reasoning_relations: Iterable[str],
) -> RelationFeatureBundle:
    """Build frozen relation features from *Training-only* structural statistics.

    The function deliberately refuses any merged file containing validation-performance
    columns. This makes leakage through ``best_baseline`` / Validation MRR impossible.
    """
    rows = _read_csv(structure_csv)
    if not rows:
        raise ValueError("Empty relation structure CSV")
    present_columns = set(rows[0])
    leakage = sorted(FORBIDDEN_STRUCTURE_COLUMNS & present_columns)
    if leakage:
        raise ValueError(
            "Validation-derived columns are forbidden in v1.9 router inputs: " + ", ".join(leakage)
        )

    allowed = list(allowed_reasoning_relations)
    by_relation = {row["relation"]: row for row in rows}
    if set(by_relation) != set(allowed):
        missing = sorted(set(allowed) - set(by_relation))
        extra = sorted(set(by_relation) - set(allowed))
        raise ValueError(f"Structure relation mismatch; missing={missing}, extra={extra}")

    raw_continuous: dict[str, list[float]] = {
        "log_train_frequency": [],
        "log_tph": [],
        "log_hpt": [],
        "log_type_signature_count": [],
        "type_signature_entropy": [],
        "dominant_type_signature_ratio": [],
        "best_inverse_score": [],
    }
    for relation in allowed:
        row = by_relation[relation]
        raw_continuous["log_train_frequency"].append(math.log1p(float(row["train_triples"])))
        raw_continuous["log_tph"].append(math.log1p(float(row["tph"])))
        raw_continuous["log_hpt"].append(math.log1p(float(row["hpt"])))
        raw_continuous["log_type_signature_count"].append(math.log1p(float(row["type_signature_count"])))
        raw_continuous["type_signature_entropy"].append(float(row["type_signature_entropy"]))
        raw_continuous["dominant_type_signature_ratio"].append(float(row["dominant_type_signature_ratio"]))
        raw_continuous["best_inverse_score"].append(float(row["best_inverse_score"]))

    normalized: dict[str, list[float]] = {}
    normalization: dict[str, dict[str, float]] = {}
    for name, values in raw_continuous.items():
        z, mean, std = _zscore(values)
        normalized[name] = z
        normalization[name] = {"mean": mean, "std_population": std}

    num_relations = max(relation_to_id.values()) + 1
    matrix = torch.zeros((num_relations, len(FEATURE_NAMES)), dtype=torch.float32)
    output_rows: list[dict[str, Any]] = []
    for i, relation in enumerate(allowed):
        role = by_relation[relation]["role"]
        if role == "diagnostic_semantic":
            role_diag, role_proc = 1.0, 0.0
        elif role == "procedural_structural":
            role_diag, role_proc = 0.0, 1.0
        else:
            raise ValueError(f"Unexpected Reasoning14 role for {relation}: {role}")
        vector = [
            role_diag,
            role_proc,
            normalized["log_train_frequency"][i],
            normalized["log_tph"][i],
            normalized["log_hpt"][i],
            normalized["log_type_signature_count"][i],
            normalized["type_signature_entropy"][i],
            normalized["dominant_type_signature_ratio"][i],
            normalized["best_inverse_score"][i],
        ]
        rid = int(relation_to_id[relation])
        matrix[rid] = torch.tensor(vector, dtype=torch.float32)
        output_rows.append(
            {
                "relation": relation,
                "relation_id": rid,
                "role": role,
                **{name: float(value) for name, value in zip(FEATURE_NAMES, vector, strict=True)},
            }
        )

    # Full17 provenance IDs intentionally remain all-zero: they are outside v1.9 primary training/evaluation.
    return RelationFeatureBundle(
        tensor=matrix,
        feature_names=FEATURE_NAMES,
        rows=output_rows,
        normalization=normalization,
        relation_to_id=dict(relation_to_id),
    )


def write_relation_feature_artifacts(bundle: RelationFeatureBundle, *, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "relation_router_features_v19.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bundle.rows[0]))
        writer.writeheader()
        writer.writerows(bundle.rows)
    manifest = {
        "feature_names": list(bundle.feature_names),
        "feature_tensor_shape": list(bundle.tensor.shape),
        "feature_tensor_sha256": bundle.sha256(),
        "normalization": bundle.normalization,
        "leakage_guard": "Training-only structure features; Validation/Test performance is forbidden as router input.",
        "provenance_feature_policy": "Full17 provenance relation rows are zero because Provenance3 is outside the v1.9 primary Reasoning14 objective.",
    }
    (out / "relation_router_features_v19.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalized_positive_pair(raw: Tensor, eps: float = 1.0e-12) -> Tensor:
    positive = F.softplus(raw) + eps
    return positive / torch.linalg.vector_norm(positive, ord=2, dim=-1, keepdim=True)


class RRSMoGECore(nn.Module):
    """PyTorch-only core for the v1.9 role/structure-aware soft expert router."""

    def __init__(
        self,
        *,
        feature_dim: int = len(FEATURE_NAMES),
        transh_norm: int = 2,
        transh_power_norm: bool = False,
        rotate_norm: int = 2,
    ) -> None:
        super().__init__()
        self.transh_norm = transh_norm
        self.transh_power_norm = transh_power_norm
        self.rotate_norm = rotate_norm
        self.router = nn.Linear(feature_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Zero init reproduces Original TH-RotatE's equal 1/sqrt(2) start for every relation.
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def fusion_weights(self, relation_features: Tensor) -> Tensor:
        return normalized_positive_pair(self.router(relation_features))

    def forward(self, h, r, t) -> Tensor:
        h_h, h_r = h
        d_r, w_r, r_r, x_r = r
        t_h, t_r = t
        d_h = transh_distance(
            h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
        )
        d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
        alpha_r, beta_r = self.fusion_weights(x_r).unbind(dim=-1)
        return -(alpha_r * d_h + beta_r * d_rot)


if Interaction is not None:

    class RRSMoGEInteraction(Interaction):
        """Role/structure-aware soft fusion of TransH and RotatE experts."""

        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d", "f")

        def __init__(
            self,
            *,
            feature_dim: int = len(FEATURE_NAMES),
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
        ) -> None:
            super().__init__()
            self.transh_norm = transh_norm
            self.transh_power_norm = transh_power_norm
            self.rotate_norm = rotate_norm
            self.router = nn.Linear(feature_dim, 2)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)

        def relation_fusion_weights(self, relation_features: Tensor) -> Tensor:
            return normalized_positive_pair(self.router(relation_features))

        def component_distances(self, h, r, t) -> tuple[Tensor, Tensor, Tensor]:
            h_h, h_r = h
            d_r, w_r, r_r, x_r = r
            t_h, t_r = t
            d_h = transh_distance(
                h_h,
                d_r,
                w_r,
                t_h,
                p=self.transh_norm,
                power_norm=self.transh_power_norm,
            )
            d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
            weights = self.relation_fusion_weights(x_r)
            return d_h, d_rot, weights

        def forward(self, h, r, t):
            d_h, d_rot, weights = self.component_distances(h=h, r=r, t=t)
            alpha_r, beta_r = weights.unbind(dim=-1)
            return -(alpha_r * d_h + beta_r * d_rot)


    class RRSMoGE(ERModel):
        """v1.9 RRS-MoGE: low-capacity Training-only structure-aware two-expert model."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            relation_feature_tensor: Tensor,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
            **kwargs: Any,
        ) -> None:
            if relation_feature_tensor.ndim != 2:
                raise ValueError("relation_feature_tensor must have shape (num_relations, feature_dim)")
            feature_dim = int(relation_feature_tensor.shape[1])
            feature_initializer = PretrainedInitializer(tensor=relation_feature_tensor.detach().cpu())
            super().__init__(
                interaction=RRSMoGEInteraction,
                interaction_kwargs=dict(
                    feature_dim=feature_dim,
                    transh_norm=transh_norm,
                    transh_power_norm=transh_power_norm,
                    rotate_norm=rotate_norm,
                ),
                entity_representations_kwargs=[
                    dict(shape=embedding_dim, initializer=nn.init.xavier_normal_),
                    dict(shape=embedding_dim, initializer=xavier_uniform_, dtype=torch.cfloat),
                ],
                relation_representations_kwargs=[
                    dict(shape=embedding_dim, initializer=nn.init.xavier_normal_),
                    dict(
                        shape=embedding_dim,
                        initializer=nn.init.xavier_normal_,
                        constrainer=normalize_last_dim,
                    ),
                    dict(
                        shape=embedding_dim,
                        initializer=init_phases,
                        constrainer=complex_normalize,
                        dtype=torch.cfloat,
                    ),
                    dict(
                        shape=feature_dim,
                        initializer=feature_initializer,
                        trainable=False,
                    ),
                ],
                **kwargs,
            )

        def get_all_relation_fusion_weights(self) -> Tensor:
            features = self.relation_representations[-1](indices=None)
            return self.interaction.relation_fusion_weights(features).detach().cpu()

        def score_components_hrt(self, hrt_batch: Tensor, *, mode=None) -> dict[str, Tensor]:
            h, r, t = self._get_representations(
                h=hrt_batch[:, 0],
                r=hrt_batch[:, 1],
                t=hrt_batch[:, 2],
                mode=mode,
            )
            d_h, d_rot, weights = self.interaction.component_distances(h=h, r=r, t=t)
            alpha_r, beta_r = weights.unbind(dim=-1)
            return {
                "transh_distance": d_h,
                "rotate_distance": d_rot,
                "fusion_alpha": alpha_r,
                "fusion_beta": beta_r,
                "fused_score": -(alpha_r * d_h + beta_r * d_rot),
            }

else:
    RRSMoGEInteraction = None
    RRSMoGE = None




if NSSALoss is not None:

    class RelationBalancedNSSALoss(NSSALoss):
        """NSSA with conservative relation-frequency weights for sLCWA.

        PyKEEN 1.11.1's base adversarial loss intentionally rejects sample weights.
        This subclass keeps the same NSSA score terms and self-adversarial negative
        weighting, but reduces the per-positive loss sets with the supplied relation weight.
        """

        def process_slcwa_scores(
            self,
            positive_scores: Tensor,
            negative_scores: Tensor,
            label_smoothing: float | None = None,
            batch_filter: Tensor | None = None,
            num_entities: int | None = None,
            pos_weights: Tensor | None = None,
            neg_weights: Tensor | None = None,
        ) -> Tensor:
            del num_entities  # NSSA does not use it without label smoothing.
            if label_smoothing:
                raise UnsupportedLabelSmoothingError(self)

            negative_scores = prepare_negative_scores_for_softmax(
                batch_filter=batch_filter,
                negative_scores=negative_scores,
                no_inf_rows=True,
            )

            # RelationLossWeighter supplies both positive and negative weights. Bernoulli
            # corruption preserves r, so all negatives spawned from a positive have the
            # same relation weight. We use the positive-set weight once for the full set.
            # `neg_weights` is therefore deliberately not multiplied a second time.
            if pos_weights is None and neg_weights is not None:
                # Defensive fallback. For dense negatives all values in a row should be
                # the same relation weight, hence row mean recovers the positive weight.
                if neg_weights.ndim == 1:
                    raise ValueError("Cannot infer per-positive relation weights from flattened neg_weights")
                pos_weights = neg_weights.reshape(neg_weights.shape[0], -1).mean(dim=-1, keepdim=True)

            return relation_balanced_nssa_from_dense_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                sample_weights=pos_weights,
                margin=float(self.margin),
                adversarial_temperature=float(self.inverse_softmax_temperature),
                reduction=self.reduction,
            )

else:
    RelationBalancedNSSALoss = None

def relation_balanced_nssa_from_dense_scores(
    *,
    positive_scores: Tensor,
    negative_scores: Tensor,
    sample_weights: Tensor | None,
    margin: float,
    adversarial_temperature: float,
    reduction: str = "mean",
) -> Tensor:
    """Compute per-positive relation-balanced NSSA on dense negative sets.

    This preserves the PyKEEN 1.11.1 NSSA semantics when ``sample_weights`` are all one:
    the positive term is ``-logsigmoid(margin + positive_score)``, negative scores are
    self-adversarially softmax-weighted, and the mean reduction uses the original 0.5 factor.

    Relation weighting is applied once per positive triple to the *whole* positive+negative
    training set generated from it. This is appropriate for the current Bernoulli corruption
    sampler because corruption changes head/tail only and preserves the relation identifier.
    """
    if positive_scores.ndim == 1:
        positive_scores = positive_scores.unsqueeze(-1)
    if positive_scores.ndim != 2 or positive_scores.shape[1] != 1:
        raise ValueError(f"positive_scores must have shape (batch, 1), got {tuple(positive_scores.shape)}")
    if negative_scores.ndim != 2 or negative_scores.shape[0] != positive_scores.shape[0]:
        raise ValueError(
            "negative_scores must have shape (batch, num_negatives) aligned to positive_scores; "
            f"got positive={tuple(positive_scores.shape)}, negative={tuple(negative_scores.shape)}"
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction={reduction!r}")

    batch_size = positive_scores.shape[0]
    if sample_weights is None:
        weights = torch.ones(batch_size, device=positive_scores.device, dtype=positive_scores.dtype)
    else:
        weights = sample_weights.to(device=positive_scores.device, dtype=positive_scores.dtype).reshape(-1)
        if weights.numel() != batch_size:
            raise ValueError(f"sample_weights must contain {batch_size} values, got {weights.numel()}")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weights must be finite and non-negative")

    positive_per_example = -F.logsigmoid(float(margin) + positive_scores.squeeze(-1))

    # Identical to PyKEEN's self-adversarial weighting: gradients do not flow through
    # the softmax weights themselves. Non-finite values can arise only after negative
    # filtering; their adversarial weight is zero and their loss input is replaced by 0.
    adversarial = negative_scores.detach().mul(float(adversarial_temperature)).softmax(dim=-1)
    finite_negative_scores = torch.masked_fill(negative_scores, mask=~torch.isfinite(negative_scores), value=0.0)
    negative_unreduced = -F.logsigmoid(-finite_negative_scores - float(margin))
    negative_per_example = (adversarial * negative_unreduced).sum(dim=-1)

    if reduction == "mean":
        denom = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        positive_loss = (weights * positive_per_example).sum() / denom
        negative_loss = (weights * negative_per_example).sum() / denom
        return 0.5 * (positive_loss + negative_loss)

    positive_loss = (weights * positive_per_example).sum()
    negative_loss = (weights * negative_per_example).sum()
    return positive_loss + negative_loss

def build_sqrt_inverse_relation_weights(
    *,
    mapped_training_triples: Tensor,
    num_relations: int,
    exponent: float = 0.5,
    max_weight_before_renorm: float = 3.0,
    min_weight_before_renorm: float = 0.5,
) -> tuple[Tensor, dict[str, Any]]:
    """Conservative relation balancing: inverse-frequency^0.5, clipped, weighted-mean normalized to 1.

    Full inverse frequency would make the rarest 21/34-triple relations dominate each example.
    The square-root transform reduces imbalance while keeping the intervention conservative.
    """
    if mapped_training_triples.ndim != 2 or mapped_training_triples.shape[1] != 3:
        raise ValueError("mapped_training_triples must have shape (n, 3)")
    relation_ids, counts = mapped_training_triples[:, 1].unique(return_counts=True)
    count_by_id = {int(r): int(c) for r, c in zip(relation_ids.tolist(), counts.tolist(), strict=True)}
    weights = torch.ones(num_relations, dtype=torch.float32)
    raw: dict[int, float] = {}
    for rid, count in count_by_id.items():
        value = count ** (-float(exponent))
        raw[rid] = min(max(value, 0.0), float("inf"))

    # First normalize relative to the training-triple-weighted average.
    denom = sum(count_by_id[rid] * raw[rid] for rid in raw) / sum(count_by_id.values())
    normalized = {rid: raw[rid] / denom for rid in raw}
    clipped = {
        rid: min(max(value, min_weight_before_renorm), max_weight_before_renorm)
        for rid, value in normalized.items()
    }
    # Re-normalize so average positive-example weight over Training equals exactly one.
    denom2 = sum(count_by_id[rid] * clipped[rid] for rid in clipped) / sum(count_by_id.values())
    final = {rid: clipped[rid] / denom2 for rid in clipped}
    for rid, value in final.items():
        weights[rid] = float(value)

    manifest = {
        "formula": "w_r proportional to N_r^(-0.5), clip before final normalization, training-triple-weighted mean = 1",
        "exponent": exponent,
        "clip_before_final_normalization": [min_weight_before_renorm, max_weight_before_renorm],
        "weighted_mean_over_training_triples": (
            sum(count_by_id[rid] * final[rid] for rid in final) / sum(count_by_id.values())
        ),
        "counts_by_relation_id": count_by_id,
        "weights_by_relation_id": final,
    }
    return weights, manifest


def make_relation_loss_weighter(weights: Tensor):
    if RelationLossWeighter is None:
        raise RuntimeError("PyKEEN is required to create RelationLossWeighter") from _PYKEEN_IMPORT_ERROR
    return RelationLossWeighter(weights=weights)
