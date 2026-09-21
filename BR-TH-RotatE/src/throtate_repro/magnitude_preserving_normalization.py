from __future__ import annotations

"""Magnitude-preserving branch-scale sensitivity for bounded relation-adaptive TH-RotatE.

This module is isolated from the frozen main implementation. It implements a
secondary/post-hoc sensitivity model that equalizes the *relative* TransH and
RotatE branch scales while keeping the overall fused-distance magnitude near
its original training scale.

Let c_H and c_R be fixed positive branch-distance means estimated from Training
positives only after a frozen-budget raw-D2 calibration run. Define

    c_* = sqrt(c_H * c_R)

and transform the branch distances as

    d_H^mp = (c_* / c_H) * d_H
    d_R^mp = (c_* / c_R) * d_R

Thus, on the calibration set, both transformed branch means equal c_*. Unlike
naive d/c normalization, this does not collapse both branch means to 1 and
therefore avoids a large change in absolute score magnitude relative to the
fixed NSSA margin gamma.

The constants are not trainable and are not selected using Validation/Test.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn

from .bounded_adaptive_v26 import bounded_relation_weights
from .model import normalize_last_dim, normalized_positive_weights, rotate_distance, transh_distance
from .rrs_moge_v19 import FEATURE_NAMES

try:
    from pykeen.losses import NSSALoss
    from pykeen.models import ERModel
    from pykeen.nn.init import PretrainedInitializer, init_phases, xavier_uniform_
    from pykeen.nn.modules import Interaction
    from pykeen.utils import complex_normalize
except ImportError as exc:  # Keep math-only tests importable without PyKEEN.
    NSSALoss = ERModel = Interaction = None
    PretrainedInitializer = init_phases = xavier_uniform_ = complex_normalize = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def magnitude_preserving_constants(transh_mean: float, rotate_mean: float) -> dict[str, float]:
    """Return c_* and the two multiplicative scale factors."""
    c_h = _validate_positive(transh_mean, "transh_mean")
    c_r = _validate_positive(rotate_mean, "rotate_mean")
    c_star = math.sqrt(c_h * c_r)
    return {
        "transh_mean_cH": c_h,
        "rotate_mean_cR": c_r,
        "common_scale_cStar": c_star,
        "transh_multiplier": c_star / c_h,
        "rotate_multiplier": c_star / c_r,
    }


class BoundedAdaptiveMagnitudePreservingTHRotatECore(nn.Module):
    """PyTorch-only core for magnitude-preserving branch-scale sensitivity."""

    def __init__(
        self,
        *,
        transh_mean: float,
        rotate_mean: float,
        feature_dim: int = len(FEATURE_NAMES),
        max_angle_deviation: float = 0.15,
        transh_norm: int = 2,
        transh_power_norm: bool = False,
        rotate_norm: int = 2,
    ) -> None:
        super().__init__()
        self.max_angle_deviation = float(max_angle_deviation)
        self.transh_norm = int(transh_norm)
        self.transh_power_norm = bool(transh_power_norm)
        self.rotate_norm = int(rotate_norm)
        c = magnitude_preserving_constants(transh_mean, rotate_mean)
        self.register_buffer("transh_mean", torch.tensor(c["transh_mean_cH"]))
        self.register_buffer("rotate_mean", torch.tensor(c["rotate_mean_cR"]))
        self.register_buffer("common_scale", torch.tensor(c["common_scale_cStar"]))
        self.register_buffer("transh_multiplier", torch.tensor(c["transh_multiplier"]))
        self.register_buffer("rotate_multiplier", torch.tensor(c["rotate_multiplier"]))
        self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
        self.relation_router = nn.Linear(int(feature_dim), 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.raw_fusion_weights.zero_()
            nn.init.zeros_(self.relation_router.weight)

    @property
    def global_fusion_weights(self) -> Tensor:
        return normalized_positive_weights(self.raw_fusion_weights)

    def relation_fusion_weights(self, relation_features: Tensor, reliability: Tensor):
        return bounded_relation_weights(
            global_weights=self.global_fusion_weights,
            router_logit=self.relation_router(relation_features),
            reliability=reliability,
            max_angle_deviation=self.max_angle_deviation,
        )

    def rescale_distances(self, d_h: Tensor, d_rot: Tensor) -> tuple[Tensor, Tensor]:
        return d_h * self.transh_multiplier, d_rot * self.rotate_multiplier

    def forward(self, h, r, t) -> Tensor:
        h_h, h_r = h
        d_r, w_r, r_r, x_r, q_r = r
        t_h, t_r = t
        d_h = transh_distance(
            h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
        )
        d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
        d_h_mp, d_rot_mp = self.rescale_distances(d_h, d_rot)
        weights, _theta_g, _shift = self.relation_fusion_weights(x_r, q_r)
        alpha_r, beta_r = weights.unbind(dim=-1)
        return -(alpha_r * d_h_mp + beta_r * d_rot_mp)


if Interaction is not None:

    class BoundedAdaptiveMagnitudePreservingTHRotatEInteraction(Interaction):
        """D2 interaction with fixed magnitude-preserving branch rescaling."""

        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d", "f", "q")

        def __init__(
            self,
            *,
            transh_mean: float,
            rotate_mean: float,
            feature_dim: int = len(FEATURE_NAMES),
            max_angle_deviation: float = 0.15,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
        ) -> None:
            super().__init__()
            self.max_angle_deviation = float(max_angle_deviation)
            if not 0.0 <= self.max_angle_deviation < math.pi / 2:
                raise ValueError("max_angle_deviation must be in [0, pi/2)")
            self.transh_norm = int(transh_norm)
            self.transh_power_norm = bool(transh_power_norm)
            self.rotate_norm = int(rotate_norm)
            c = magnitude_preserving_constants(transh_mean, rotate_mean)
            self.register_buffer("transh_mean", torch.tensor(c["transh_mean_cH"]))
            self.register_buffer("rotate_mean", torch.tensor(c["rotate_mean_cR"]))
            self.register_buffer("common_scale", torch.tensor(c["common_scale_cStar"]))
            self.register_buffer("transh_multiplier", torch.tensor(c["transh_multiplier"]))
            self.register_buffer("rotate_multiplier", torch.tensor(c["rotate_multiplier"]))
            self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
            self.relation_router = nn.Linear(int(feature_dim), 1, bias=False)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            with torch.no_grad():
                self.raw_fusion_weights.zero_()
                nn.init.zeros_(self.relation_router.weight)

        @property
        def global_fusion_weights(self) -> Tensor:
            return normalized_positive_weights(self.raw_fusion_weights)

        def relation_fusion_weights(self, relation_features: Tensor, reliability: Tensor):
            return bounded_relation_weights(
                global_weights=self.global_fusion_weights,
                router_logit=self.relation_router(relation_features),
                reliability=reliability,
                max_angle_deviation=self.max_angle_deviation,
            )

        def component_distances(self, h, r, t):
            h_h, h_r = h
            d_r, w_r, r_r, x_r, q_r = r
            t_h, t_r = t
            d_h = transh_distance(
                h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
            )
            d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
            d_h_mp = d_h * self.transh_multiplier
            d_rot_mp = d_rot * self.rotate_multiplier
            weights, theta_global, shift = self.relation_fusion_weights(x_r, q_r)
            return d_h, d_rot, d_h_mp, d_rot_mp, weights, theta_global, shift

        def forward(self, h, r, t):
            _d_h, _d_rot, d_h_mp, d_rot_mp, weights, _theta_global, _shift = self.component_distances(
                h=h, r=r, t=t
            )
            alpha_r, beta_r = weights.unbind(dim=-1)
            return -(alpha_r * d_h_mp + beta_r * d_rot_mp)


    class BoundedAdaptiveMagnitudePreservingTHRotatE(ERModel):
        """Secondary D2-MPNorm sensitivity model; main raw D2 remains frozen."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            relation_feature_tensor: Tensor,
            relation_reliability_tensor: Tensor,
            max_angle_deviation: float,
            transh_mean: float,
            rotate_mean: float,
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
                interaction=BoundedAdaptiveMagnitudePreservingTHRotatEInteraction,
                interaction_kwargs=dict(
                    transh_mean=float(transh_mean),
                    rotate_mean=float(rotate_mean),
                    feature_dim=feature_dim,
                    max_angle_deviation=float(max_angle_deviation),
                    transh_norm=int(transh_norm),
                    transh_power_norm=bool(transh_power_norm),
                    rotate_norm=int(rotate_norm),
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

        def get_magnitude_preserving_constants(self) -> dict[str, float]:
            return {
                "transh_mean_cH": float(self.interaction.transh_mean.detach().cpu()),
                "rotate_mean_cR": float(self.interaction.rotate_mean.detach().cpu()),
                "common_scale_cStar": float(self.interaction.common_scale.detach().cpu()),
                "transh_multiplier": float(self.interaction.transh_multiplier.detach().cpu()),
                "rotate_multiplier": float(self.interaction.rotate_multiplier.detach().cpu()),
            }

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
            d_h, d_rot, d_h_mp, d_rot_mp, weights, theta_global, shift = self.interaction.component_distances(
                h=h, r=r, t=t
            )
            alpha_r, beta_r = weights.unbind(dim=-1)
            return {
                "transh_distance": d_h,
                "rotate_distance": d_rot,
                "magnitude_preserved_transh_distance": d_h_mp,
                "magnitude_preserved_rotate_distance": d_rot_mp,
                "fusion_alpha": alpha_r,
                "fusion_beta": beta_r,
                "global_angle": theta_global,
                "angle_shift": shift,
                "fused_score": -(alpha_r * d_h_mp + beta_r * d_rot_mp),
            }

else:
    BoundedAdaptiveMagnitudePreservingTHRotatEInteraction = None
    BoundedAdaptiveMagnitudePreservingTHRotatE = None


def build_bounded_adaptive_magnitude_preserving_model_class():
    if BoundedAdaptiveMagnitudePreservingTHRotatE is None:
        raise RuntimeError("PyKEEN is not installed. Use the same environment as the main experiments.") from _PYKEEN_IMPORT_ERROR
    return BoundedAdaptiveMagnitudePreservingTHRotatE
