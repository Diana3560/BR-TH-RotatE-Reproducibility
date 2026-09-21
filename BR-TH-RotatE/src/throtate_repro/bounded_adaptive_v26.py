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

from .model import normalize_last_dim, normalized_positive_weights, rotate_distance, transh_distance
from .rrs_moge_v19 import FEATURE_NAMES, RelationFeatureBundle, build_relation_feature_bundle

try:
    from pykeen.losses import NSSALoss
    from pykeen.models import ERModel
    from pykeen.nn.init import PretrainedInitializer, init_phases, xavier_uniform_
    from pykeen.nn.modules import Interaction
    from pykeen.utils import complex_normalize
except ImportError as exc:  # Keep PyTorch-only core importable for unit tests.
    NSSALoss = ERModel = Interaction = None
    PretrainedInitializer = init_phases = xavier_uniform_ = complex_normalize = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


ANGLE_EPS = 1.0e-5


@dataclass(frozen=True)
class BoundedRelationFeatureBundle:
    """Frozen Training-only router features plus parameter-free frequency reliability."""

    features: RelationFeatureBundle
    reliability: Tensor
    reliability_rows: list[dict[str, Any]]
    reliability_rule: str

    def reliability_sha256(self) -> str:
        payload = self.reliability.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()


def _read_structure_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_bounded_relation_feature_bundle(
    *,
    structure_csv: str | Path,
    relation_to_id: Mapping[str, int],
    allowed_reasoning_relations: Iterable[str],
) -> BoundedRelationFeatureBundle:
    """Build v2.6 Training-only features and low-frequency shrinkage reliability.

    The structural feature vector is exactly the v1.9 Training-only 9-D vector. The
    reliability is deliberately parameter-free:

        q_r = sqrt(N_r / max_s N_s)

    Hence q_r is in (0, 1] for Reasoning14 relations. Rare relations are forced closer
    to the global TH-RotatE anchor without adding a tunable frequency threshold.
    Provenance3 relation IDs (outside the primary R14 objective) retain q=0.
    """
    allowed = tuple(allowed_reasoning_relations)
    feature_bundle = build_relation_feature_bundle(
        structure_csv=structure_csv,
        relation_to_id=relation_to_id,
        allowed_reasoning_relations=allowed,
    )
    rows = _read_structure_rows(structure_csv)
    by_relation = {row["relation"]: row for row in rows}
    if set(by_relation) != set(allowed):
        raise ValueError("Training structure relation set does not match Reasoning14")
    frequencies = {r: int(by_relation[r]["train_triples"]) for r in allowed}
    if any(n <= 0 for n in frequencies.values()):
        raise ValueError("Every Reasoning14 relation must have positive Training frequency")
    max_frequency = max(frequencies.values())
    reliability = torch.zeros((max(relation_to_id.values()) + 1, 1), dtype=torch.float32)
    reliability_rows: list[dict[str, Any]] = []
    for relation in allowed:
        q = math.sqrt(frequencies[relation] / max_frequency)
        rid = int(relation_to_id[relation])
        reliability[rid, 0] = float(q)
        reliability_rows.append(
            {
                "relation": relation,
                "relation_id": rid,
                "train_triples": frequencies[relation],
                "max_train_triples": max_frequency,
                "reliability_q": float(q),
            }
        )
    return BoundedRelationFeatureBundle(
        features=feature_bundle,
        reliability=reliability,
        reliability_rows=reliability_rows,
        reliability_rule="sqrt(train_relation_frequency / max_reasoning14_train_frequency)",
    )


def write_bounded_relation_feature_artifacts(
    bundle: BoundedRelationFeatureBundle, *, output_dir: str | Path
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_rows = bundle.features.rows
    if feature_rows:
        with (out / "relation_features_v26.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)
    with (out / "relation_reliability_v26.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bundle.reliability_rows[0]))
        writer.writeheader()
        writer.writerows(bundle.reliability_rows)
    manifest = {
        "feature_names": list(bundle.features.feature_names),
        "feature_tensor_shape": list(bundle.features.tensor.shape),
        "feature_tensor_sha256": bundle.features.sha256(),
        "feature_normalization": bundle.features.normalization,
        "reliability_shape": list(bundle.reliability.shape),
        "reliability_sha256": bundle.reliability_sha256(),
        "reliability_rule": bundle.reliability_rule,
        "leakage_guard": "All relation features and q_r values are computed from Reasoning14 Training only.",
    }
    (out / "bounded_relation_feature_manifest_v26.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def global_pair_to_angle(weights: Tensor) -> Tensor:
    """Convert positive L2-normalized (alpha,beta) to theta in the first quadrant."""
    alpha, beta = weights.unbind(dim=-1)
    return torch.atan2(beta, alpha)


def bounded_relation_weights(
    *,
    global_weights: Tensor,
    router_logit: Tensor,
    reliability: Tensor,
    max_angle_deviation: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a bounded angular relation correction around the global TH-RotatE anchor.

    theta_r = theta_g + q_r * Delta * tanh(router(z_r))
    alpha_r = cos(theta_r), beta_r = sin(theta_r)

    The angular representation preserves alpha_r^2 + beta_r^2 = 1 exactly (up to
    floating-point error), while ``Delta`` explicitly caps the relation-specific freedom.
    """
    if not 0.0 <= float(max_angle_deviation) < math.pi / 2:
        raise ValueError("max_angle_deviation must be in [0, pi/2)")
    theta_global = global_pair_to_angle(global_weights)
    shift = reliability.squeeze(dim=-1) * float(max_angle_deviation) * torch.tanh(router_logit.squeeze(dim=-1))
    theta_relation = (theta_global + shift).clamp(min=ANGLE_EPS, max=math.pi / 2 - ANGLE_EPS)
    weights = torch.stack([torch.cos(theta_relation), torch.sin(theta_relation)], dim=-1)
    return weights, theta_global, shift


class BoundedAdaptiveTHRotatECore(nn.Module):
    """PyTorch-only v2.6 core: global TH-RotatE anchor + bounded relation deviation."""

    def __init__(
        self,
        *,
        feature_dim: int = len(FEATURE_NAMES),
        max_angle_deviation: float = 0.10,
        transh_norm: int = 2,
        transh_power_norm: bool = False,
        rotate_norm: int = 2,
    ) -> None:
        super().__init__()
        self.max_angle_deviation = float(max_angle_deviation)
        self.transh_norm = transh_norm
        self.transh_power_norm = transh_power_norm
        self.rotate_norm = rotate_norm
        # Exact same learnable global fusion parameterization as corrected TH-RotatE.
        self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
        # One shared low-capacity direction predictor; zero init => no relation deviation at start.
        self.relation_router = nn.Linear(feature_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.raw_fusion_weights.zero_()
            nn.init.zeros_(self.relation_router.weight)

    @property
    def global_fusion_weights(self) -> Tensor:
        return normalized_positive_weights(self.raw_fusion_weights)

    def relation_fusion_weights(self, relation_features: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return bounded_relation_weights(
            global_weights=self.global_fusion_weights,
            router_logit=self.relation_router(relation_features),
            reliability=reliability,
            max_angle_deviation=self.max_angle_deviation,
        )

    def forward(self, h, r, t) -> Tensor:
        h_h, h_r = h
        d_r, w_r, r_r, x_r, q_r = r
        t_h, t_r = t
        d_h = transh_distance(h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm)
        d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
        weights, _theta_g, _shift = self.relation_fusion_weights(x_r, q_r)
        alpha_r, beta_r = weights.unbind(dim=-1)
        return -(alpha_r * d_h + beta_r * d_rot)


if Interaction is not None:

    class BoundedAdaptiveTHRotatEInteraction(Interaction):
        """Corrected TH-RotatE with globally anchored, bounded relation adaptation."""

        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d", "f", "q")

        def __init__(
            self,
            *,
            feature_dim: int = len(FEATURE_NAMES),
            max_angle_deviation: float = 0.10,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
        ) -> None:
            super().__init__()
            self.max_angle_deviation = float(max_angle_deviation)
            if not 0.0 <= self.max_angle_deviation < math.pi / 2:
                raise ValueError("max_angle_deviation must be in [0, pi/2)")
            self.transh_norm = transh_norm
            self.transh_power_norm = transh_power_norm
            self.rotate_norm = rotate_norm
            self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
            self.relation_router = nn.Linear(feature_dim, 1, bias=False)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            # PyKEEN recursively resets modules. Restore the predeclared equal global anchor
            # and zero relation deviation after any such reset.
            with torch.no_grad():
                self.raw_fusion_weights.zero_()
                nn.init.zeros_(self.relation_router.weight)

        @property
        def global_fusion_weights(self) -> Tensor:
            return normalized_positive_weights(self.raw_fusion_weights)

        def relation_fusion_weights(self, relation_features: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            return bounded_relation_weights(
                global_weights=self.global_fusion_weights,
                router_logit=self.relation_router(relation_features),
                reliability=reliability,
                max_angle_deviation=self.max_angle_deviation,
            )

        def component_distances(self, h, r, t) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            h_h, h_r = h
            d_r, w_r, r_r, x_r, q_r = r
            t_h, t_r = t
            d_h = transh_distance(
                h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
            )
            d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
            weights, theta_global, shift = self.relation_fusion_weights(x_r, q_r)
            return d_h, d_rot, weights, theta_global, shift

        def forward(self, h, r, t):
            d_h, d_rot, weights, _theta_global, _shift = self.component_distances(h=h, r=r, t=t)
            alpha_r, beta_r = weights.unbind(dim=-1)
            return -(alpha_r * d_h + beta_r * d_rot)


    class BoundedAdaptiveTHRotatE(ERModel):
        """v2.6 A1/A3 model built directly on corrected TH-RotatE gamma=3 geometry."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            relation_feature_tensor: Tensor,
            relation_reliability_tensor: Tensor,
            max_angle_deviation: float,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
            **kwargs: Any,
        ) -> None:
            if relation_feature_tensor.ndim != 2:
                raise ValueError("relation_feature_tensor must have shape (num_relations, feature_dim)")
            if relation_reliability_tensor.ndim != 2 or relation_reliability_tensor.shape[1] != 1:
                raise ValueError("relation_reliability_tensor must have shape (num_relations, 1)")
            if relation_feature_tensor.shape[0] != relation_reliability_tensor.shape[0]:
                raise ValueError("relation feature/reliability row count mismatch")
            feature_dim = int(relation_feature_tensor.shape[1])
            feature_initializer = PretrainedInitializer(tensor=relation_feature_tensor.detach().cpu())
            reliability_initializer = PretrainedInitializer(tensor=relation_reliability_tensor.detach().cpu())
            super().__init__(
                interaction=BoundedAdaptiveTHRotatEInteraction,
                interaction_kwargs=dict(
                    feature_dim=feature_dim,
                    max_angle_deviation=float(max_angle_deviation),
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
                    dict(shape=embedding_dim, initializer=nn.init.xavier_normal_, constrainer=normalize_last_dim),
                    dict(
                        shape=embedding_dim,
                        initializer=init_phases,
                        constrainer=complex_normalize,
                        dtype=torch.cfloat,
                    ),
                    dict(shape=feature_dim, initializer=feature_initializer, trainable=False),
                    dict(shape=1, initializer=reliability_initializer, trainable=False),
                ],
                **kwargs,
            )

        def get_global_fusion_weights(self) -> tuple[float, float]:
            weights = self.interaction.global_fusion_weights.detach().cpu().tolist()
            return float(weights[0]), float(weights[1])

        def get_all_relation_fusion_state(self) -> dict[str, Tensor]:
            features = self.relation_representations[-2](indices=None)
            reliability = self.relation_representations[-1](indices=None)
            weights, theta_global, shift = self.interaction.relation_fusion_weights(features, reliability)
            return {
                "weights": weights.detach().cpu(),
                "global_weights": self.interaction.global_fusion_weights.detach().cpu(),
                "global_angle": theta_global.detach().cpu(),
                "angle_shift": shift.detach().cpu(),
                "reliability": reliability.detach().cpu(),
            }

        def score_components_hrt(self, hrt_batch: Tensor, *, mode=None) -> dict[str, Tensor]:
            h, r, t = self._get_representations(
                h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode
            )
            d_h, d_rot, weights, theta_global, shift = self.interaction.component_distances(h=h, r=r, t=t)
            alpha_r, beta_r = weights.unbind(dim=-1)
            return {
                "transh_distance": d_h,
                "rotate_distance": d_rot,
                "fusion_alpha": alpha_r,
                "fusion_beta": beta_r,
                "global_angle": theta_global,
                "angle_shift": shift,
                "fused_score": -(alpha_r * d_h + beta_r * d_rot),
            }

else:
    BoundedAdaptiveTHRotatEInteraction = None
    BoundedAdaptiveTHRotatE = None


def build_bounded_adaptive_model_class():
    if BoundedAdaptiveTHRotatE is None:
        raise RuntimeError("PyKEEN is not installed. Run `pip install -r requirements.txt` first.") from _PYKEEN_IMPORT_ERROR
    return BoundedAdaptiveTHRotatE
