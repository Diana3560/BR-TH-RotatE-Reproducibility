from __future__ import annotations

"""v2.6.3: joint-training bounded relation adaptation with a tiny relation residual.

This module intentionally preserves the v2.6 geometry and global anchor. The only
algorithmic extension is a single scalar u_r per relation inside the already bounded
angular adapter:

    theta_r = theta_g + q_r * Delta * tanh(w^T z_r + u_r)

All parameters (embeddings, global fusion, shared router, u_r) are optimized jointly.
The external angular bound remains unchanged, so u_r cannot recover the unconstrained
free-router behaviour observed in older negative ablations.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn

from .bounded_adaptive_v26 import ANGLE_EPS, global_pair_to_angle
from .model import normalize_last_dim, normalized_positive_weights, rotate_distance, transh_distance

try:
    from pykeen.losses import NSSALoss
    from pykeen.models import ERModel
    from pykeen.nn.init import PretrainedInitializer, init_phases, xavier_uniform_
    from pykeen.nn.modules import Interaction
    from pykeen.utils import complex_normalize
except ImportError as exc:  # keep torch-only math importable in build/test environments
    NSSALoss = ERModel = Interaction = None
    PretrainedInitializer = init_phases = xavier_uniform_ = complex_normalize = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


def bounded_relation_weights_with_residual(
    *,
    global_weights: Tensor,
    router_logit: Tensor,
    relation_residual: Tensor,
    reliability: Tensor,
    max_angle_deviation: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return bounded relation weights with a one-scalar relation residual.

    The relation residual is *inside* tanh, while Delta and q_r remain outside. Thus
    the hard invariant |shift_r| <= q_r * Delta is preserved for every relation.
    """
    if not 0.0 <= float(max_angle_deviation) < math.pi / 2:
        raise ValueError("max_angle_deviation must be in [0, pi/2)")
    if relation_residual.shape[-1] != 1:
        raise ValueError("relation_residual must have final dimension 1")
    theta_global = global_pair_to_angle(global_weights)
    combined_logit = router_logit.squeeze(dim=-1) + relation_residual.squeeze(dim=-1)
    shift = reliability.squeeze(dim=-1) * float(max_angle_deviation) * torch.tanh(combined_logit)
    theta_relation = (theta_global + shift).clamp(min=ANGLE_EPS, max=math.pi / 2 - ANGLE_EPS)
    weights = torch.stack([torch.cos(theta_relation), torch.sin(theta_relation)], dim=-1)
    return weights, theta_global, shift, combined_logit


class ResidualBoundedAdaptiveCore(nn.Module):
    """Torch-only core for regression tests."""

    def __init__(
        self,
        *,
        feature_dim: int,
        max_angle_deviation: float = 0.10,
        transh_norm: int = 2,
        transh_power_norm: bool = False,
        rotate_norm: int = 2,
    ) -> None:
        super().__init__()
        self.max_angle_deviation = float(max_angle_deviation)
        self.transh_norm = int(transh_norm)
        self.transh_power_norm = bool(transh_power_norm)
        self.rotate_norm = int(rotate_norm)
        self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
        self.relation_router = nn.Linear(feature_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.raw_fusion_weights.zero_()
            nn.init.zeros_(self.relation_router.weight)

    @property
    def global_fusion_weights(self) -> Tensor:
        return normalized_positive_weights(self.raw_fusion_weights)

    def relation_fusion_weights(
        self, relation_features: Tensor, reliability: Tensor, relation_residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return bounded_relation_weights_with_residual(
            global_weights=self.global_fusion_weights,
            router_logit=self.relation_router(relation_features),
            relation_residual=relation_residual,
            reliability=reliability,
            max_angle_deviation=self.max_angle_deviation,
        )

    def forward(self, h, r, t) -> Tensor:
        h_h, h_r = h
        d_r, w_r, r_r, x_r, q_r, u_r = r
        t_h, t_r = t
        d_h = transh_distance(h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm)
        d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
        weights, _, _, _ = self.relation_fusion_weights(x_r, q_r, u_r)
        alpha_r, beta_r = weights.unbind(dim=-1)
        return -(alpha_r * d_h + beta_r * d_rot)


if Interaction is not None:

    class ResidualBoundedAdaptiveTHRotatEInteraction(Interaction):
        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d", "f", "q", "u")

        def __init__(
            self,
            *,
            feature_dim: int,
            max_angle_deviation: float = 0.10,
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

        def relation_fusion_weights(
            self, relation_features: Tensor, reliability: Tensor, relation_residual: Tensor
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            return bounded_relation_weights_with_residual(
                global_weights=self.global_fusion_weights,
                router_logit=self.relation_router(relation_features),
                relation_residual=relation_residual,
                reliability=reliability,
                max_angle_deviation=self.max_angle_deviation,
            )

        def component_distances(self, h, r, t):
            h_h, h_r = h
            d_r, w_r, r_r, x_r, q_r, u_r = r
            t_h, t_r = t
            d_h = transh_distance(
                h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
            )
            d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
            weights, theta_global, shift, combined_logit = self.relation_fusion_weights(x_r, q_r, u_r)
            return d_h, d_rot, weights, theta_global, shift, combined_logit

        def forward(self, h, r, t):
            d_h, d_rot, weights, _, _, _ = self.component_distances(h=h, r=r, t=t)
            alpha_r, beta_r = weights.unbind(dim=-1)
            return -(alpha_r * d_h + beta_r * d_rot)


    class ResidualBoundedAdaptiveTHRotatE(ERModel):
        """v2.6.3 V2/V3 model. Supports ordinary or reciprocal TriplesFactory mappings."""

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
                raise ValueError("relation_reliability_tensor must have shape (num_relations,1)")
            if relation_feature_tensor.shape[0] != relation_reliability_tensor.shape[0]:
                raise ValueError("relation feature/reliability row count mismatch")
            feature_dim = int(relation_feature_tensor.shape[1])
            feature_initializer = PretrainedInitializer(tensor=relation_feature_tensor.detach().cpu())
            reliability_initializer = PretrainedInitializer(tensor=relation_reliability_tensor.detach().cpu())
            super().__init__(
                interaction=ResidualBoundedAdaptiveTHRotatEInteraction,
                interaction_kwargs=dict(
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
                    # one tiny trainable scalar per *internal* relation; zero-init => exactly V1 at start
                    dict(shape=1, initializer=nn.init.zeros_),
                ],
                **kwargs,
            )

        def get_global_fusion_weights(self) -> tuple[float, float]:
            weights = self.interaction.global_fusion_weights.detach().cpu().tolist()
            return float(weights[0]), float(weights[1])

        def get_all_relation_fusion_state(self) -> dict[str, Tensor]:
            features = self.relation_representations[-3](indices=None)
            reliability = self.relation_representations[-2](indices=None)
            residual = self.relation_representations[-1](indices=None)
            weights, theta_global, shift, combined_logit = self.interaction.relation_fusion_weights(
                features, reliability, residual
            )
            return {
                "weights": weights.detach().cpu(),
                "global_weights": self.interaction.global_fusion_weights.detach().cpu(),
                "global_angle": theta_global.detach().cpu(),
                "angle_shift": shift.detach().cpu(),
                "reliability": reliability.detach().cpu(),
                "relation_residual": residual.detach().cpu(),
                "combined_router_logit": combined_logit.detach().cpu(),
            }

        def score_components_hrt(self, hrt_batch: Tensor, *, mode=None) -> dict[str, Tensor]:
            h, r, t = self._get_representations(
                h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode
            )
            d_h, d_rot, weights, theta_global, shift, combined_logit = self.interaction.component_distances(
                h=h, r=r, t=t
            )
            alpha_r, beta_r = weights.unbind(dim=-1)
            return {
                "transh_distance": d_h,
                "rotate_distance": d_rot,
                "fusion_alpha": alpha_r,
                "fusion_beta": beta_r,
                "global_angle": theta_global,
                "angle_shift": shift,
                "combined_router_logit": combined_logit,
                "fused_score": -(alpha_r * d_h + beta_r * d_rot),
            }

else:
    ResidualBoundedAdaptiveTHRotatEInteraction = None
    ResidualBoundedAdaptiveTHRotatE = None


def build_residual_bounded_adaptive_model_class():
    if ResidualBoundedAdaptiveTHRotatE is None:
        raise RuntimeError("PyKEEN is not installed in this environment") from _PYKEEN_IMPORT_ERROR
    return ResidualBoundedAdaptiveTHRotatE
