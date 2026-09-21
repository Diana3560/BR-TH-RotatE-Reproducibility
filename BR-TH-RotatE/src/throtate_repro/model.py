from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from pykeen.losses import NSSALoss
    from pykeen.models import ERModel
    from pykeen.nn.init import init_phases, xavier_uniform_
    from pykeen.nn.modules import Interaction
    from pykeen.utils import complex_normalize
except ImportError as exc:  # Keep the PyTorch-only math helpers importable without PyKEEN.
    NSSALoss = ERModel = Interaction = None
    init_phases = xavier_uniform_ = complex_normalize = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


def normalized_positive_weights(raw_weights: Tensor, eps: float = 1.0e-12) -> Tensor:
    """Map unconstrained parameters to alpha,beta >= 0 with alpha^2+beta^2=1."""
    positive = F.softplus(raw_weights) + eps
    return positive / torch.linalg.vector_norm(positive, ord=2)


def normalize_last_dim(value: Tensor) -> Tensor:
    """Module-level TransH normal-vector constrainer (keeps saved models pickle-safe)."""
    return F.normalize(value, p=2, dim=-1)


def transh_distance(
    h: Tensor,
    relation_translation: Tensor,
    relation_normal: Tensor,
    t: Tensor,
    *,
    p: int = 2,
    power_norm: bool = True,
) -> Tensor:
    """Paper Eq. (1)-(3): relation-specific hyperplane projection and distance."""
    w = F.normalize(relation_normal, p=2, dim=-1)
    h_perp = h - (h * w).sum(dim=-1, keepdim=True) * w
    t_perp = t - (t * w).sum(dim=-1, keepdim=True) * w
    residual = h_perp + relation_translation - t_perp
    distance = torch.linalg.vector_norm(residual, ord=p, dim=-1)
    if power_norm:
        distance = distance.pow(p)
    return distance


def rotate_distance(h: Tensor, r: Tensor, t: Tensor, *, p: int = 2) -> Tensor:
    """Paper Eq. (4)-(5): complex rotation distance."""
    residual = h * r - t
    return torch.linalg.vector_norm(residual, ord=p, dim=-1)


class THRotatECore(nn.Module):
    """PyTorch-only fusion core used for unit tests and math verification."""

    def __init__(self, *, transh_norm: int = 2, transh_power_norm: bool = True, rotate_norm: int = 2) -> None:
        super().__init__()
        self.transh_norm = transh_norm
        self.transh_power_norm = transh_power_norm
        self.rotate_norm = rotate_norm
        # equal raw values -> alpha=beta=1/sqrt(2) after normalization
        self.raw_fusion_weights = nn.Parameter(torch.zeros(2))

    @property
    def fusion_weights(self) -> Tensor:
        return normalized_positive_weights(self.raw_fusion_weights)

    def forward(self, h, r, t) -> Tensor:
        h_h, h_r = h
        d_r, w_r, r_r = r
        t_h, t_r = t
        d_h = transh_distance(
            h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
        )
        d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
        alpha, beta = self.fusion_weights.unbind()
        # PyKEEN uses larger=better scores, whereas the paper writes lower=better distances.
        return -(alpha * d_h + beta * d_rot)


class RelationAdaptiveTHRotatECore(nn.Module):
    """PyTorch-only A1 core with relation-conditioned fusion weights.

    The gate receives the TransH relation-translation representation and predicts two
    positive fusion coefficients. The pair is L2-normalized per triple/relation so that
    alpha_r^2 + beta_r^2 = 1. Zero initialization makes every relation start from the
    Original TH-RotatE equal-weight state (1/sqrt(2), 1/sqrt(2)).
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        transh_norm: int = 2,
        transh_power_norm: bool = False,
        rotate_norm: int = 2,
    ) -> None:
        super().__init__()
        self.transh_norm = transh_norm
        self.transh_power_norm = transh_power_norm
        self.rotate_norm = rotate_norm
        self.relation_gate = nn.Linear(embedding_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.relation_gate.weight)
        nn.init.zeros_(self.relation_gate.bias)

    def relation_fusion_weights(self, relation_translation: Tensor) -> Tensor:
        raw = self.relation_gate(relation_translation)
        positive = F.softplus(raw) + 1.0e-12
        return positive / torch.linalg.vector_norm(
            positive, ord=2, dim=-1, keepdim=True
        )

    def forward(self, h, r, t) -> Tensor:
        h_h, h_r = h
        d_r, w_r, r_r = r
        t_h, t_r = t
        d_h = transh_distance(
            h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
        )
        d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
        weights = self.relation_fusion_weights(d_r)
        alpha_r, beta_r = weights.unbind(dim=-1)
        return -(alpha_r * d_h + beta_r * d_rot)


class ScaleCalibratedTHRotatECore(nn.Module):
    """PyTorch-only score-scale calibration core.

    This ablation keeps the Original TH-RotatE global fusion weights, but divides
    each branch distance by a non-trainable running mean estimated only while the
    module is in training mode. No Validation/Test triples are required to estimate
    the scales. The cumulative running mean introduces no momentum hyperparameter.
    """

    def __init__(
        self,
        *,
        transh_norm: int = 2,
        transh_power_norm: bool = False,
        rotate_norm: int = 2,
        scale_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.transh_norm = transh_norm
        self.transh_power_norm = transh_power_norm
        self.rotate_norm = rotate_norm
        self.scale_epsilon = float(scale_epsilon)
        self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
        self.register_buffer("transh_running_mean", torch.tensor(1.0))
        self.register_buffer("rotate_running_mean", torch.tensor(1.0))
        self.register_buffer("scale_update_count", torch.tensor(0, dtype=torch.long))

    @property
    def fusion_weights(self) -> Tensor:
        return normalized_positive_weights(self.raw_fusion_weights)

    def reset_scale_statistics(self) -> None:
        self.transh_running_mean.fill_(1.0)
        self.rotate_running_mean.fill_(1.0)
        self.scale_update_count.zero_()

    @torch.no_grad()
    def _update_scale_statistics(self, d_h: Tensor, d_rot: Tensor) -> None:
        batch_h = d_h.detach().float().mean().to(self.transh_running_mean.device)
        batch_r = d_rot.detach().float().mean().to(self.rotate_running_mean.device)
        self.scale_update_count.add_(1)
        weight = self.scale_update_count.to(dtype=self.transh_running_mean.dtype).reciprocal()
        self.transh_running_mean.add_(weight * (batch_h - self.transh_running_mean))
        self.rotate_running_mean.add_(weight * (batch_r - self.rotate_running_mean))

    def calibrated_distances(self, d_h: Tensor, d_rot: Tensor) -> tuple[Tensor, Tensor]:
        if self.training:
            self._update_scale_statistics(d_h=d_h, d_rot=d_rot)
        scale_h = self.transh_running_mean.detach().clamp_min(self.scale_epsilon)
        scale_r = self.rotate_running_mean.detach().clamp_min(self.scale_epsilon)
        return d_h / scale_h, d_rot / scale_r

    def forward(self, h, r, t) -> Tensor:
        h_h, h_r = h
        d_r, w_r, r_r = r
        t_h, t_r = t
        d_h = transh_distance(
            h_h, d_r, w_r, t_h, p=self.transh_norm, power_norm=self.transh_power_norm
        )
        d_rot = rotate_distance(h_r, r_r, t_r, p=self.rotate_norm)
        d_h_cal, d_rot_cal = self.calibrated_distances(d_h=d_h, d_rot=d_rot)
        alpha, beta = self.fusion_weights.unbind()
        return -(alpha * d_h_cal + beta * d_rot_cal)


class PositiveStepScaleCalibratedTHRotatECore(nn.Module):
    """PyTorch-only v1.2 A1 scale-calibration core.

    Unlike :class:`ScaleCalibratedTHRotatECore` from v1.1, scoring is pure: a
    forward call never mutates scale statistics. The caller must explicitly
    update the cumulative scale once from a Training-positive batch, then every
    positive/negative score in that optimizer step uses the same frozen scale.
    """

    def __init__(
        self,
        *,
        transh_norm: int = 2,
        transh_power_norm: bool = False,
        rotate_norm: int = 2,
        scale_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.transh_norm = transh_norm
        self.transh_power_norm = transh_power_norm
        self.rotate_norm = rotate_norm
        self.scale_epsilon = float(scale_epsilon)
        self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
        self.register_buffer("transh_running_mean", torch.tensor(1.0))
        self.register_buffer("rotate_running_mean", torch.tensor(1.0))
        self.register_buffer("scale_update_count", torch.tensor(0, dtype=torch.long))

    @property
    def fusion_weights(self) -> Tensor:
        return normalized_positive_weights(self.raw_fusion_weights)

    def reset_scale_statistics(self) -> None:
        self.transh_running_mean.fill_(1.0)
        self.rotate_running_mean.fill_(1.0)
        self.scale_update_count.zero_()

    def raw_component_distances(self, h, r, t) -> tuple[Tensor, Tensor]:
        h_h, h_r = h
        d_r, w_r, r_r = r
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
        return d_h, d_rot

    @torch.no_grad()
    def update_scale_from_positive_representations(self, h, r, t) -> dict[str, float | int]:
        """Update the cumulative scale exactly once from a positive batch."""
        d_h, d_rot = self.raw_component_distances(h=h, r=r, t=t)
        batch_h = d_h.detach().float().mean().to(self.transh_running_mean.device)
        batch_r = d_rot.detach().float().mean().to(self.rotate_running_mean.device)
        self.scale_update_count.add_(1)
        weight = self.scale_update_count.to(dtype=self.transh_running_mean.dtype).reciprocal()
        self.transh_running_mean.add_(weight * (batch_h - self.transh_running_mean))
        self.rotate_running_mean.add_(weight * (batch_r - self.rotate_running_mean))
        return {
            "positive_batch_transh_mean": float(batch_h.detach().cpu()),
            "positive_batch_rotate_mean": float(batch_r.detach().cpu()),
            "transh_running_mean": float(self.transh_running_mean.detach().cpu()),
            "rotate_running_mean": float(self.rotate_running_mean.detach().cpu()),
            "scale_update_count": int(self.scale_update_count.detach().cpu()),
        }

    def calibrated_distances(self, d_h: Tensor, d_rot: Tensor) -> tuple[Tensor, Tensor]:
        scale_h = self.transh_running_mean.detach().clamp_min(self.scale_epsilon)
        scale_r = self.rotate_running_mean.detach().clamp_min(self.scale_epsilon)
        return d_h / scale_h, d_rot / scale_r

    def forward(self, h, r, t) -> Tensor:
        d_h, d_rot = self.raw_component_distances(h=h, r=r, t=t)
        d_h_cal, d_rot_cal = self.calibrated_distances(d_h=d_h, d_rot=d_rot)
        alpha, beta = self.fusion_weights.unbind()
        return -(alpha * d_h_cal + beta * d_rot_cal)


if Interaction is not None:

    class PaperTransHInteraction(Interaction):
        """TransH interaction with an explicit ``(translation, normal)`` relation order.

        PyKEEN 1.11.1's public ``TransH`` model and ``TransHInteraction`` disagree about
        the order of the two relation representations. Keeping the order explicit here
        makes the standalone baseline geometrically identical to the TransH branch used
        by :class:`THRotatE`.
        """

        entity_shape = ("d",)
        relation_shape = ("d", "d")

        def __init__(
            self,
            *,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
        ) -> None:
            super().__init__()
            self.transh_norm = transh_norm
            self.transh_power_norm = transh_power_norm

        def forward(self, h, r, t):
            relation_translation, relation_normal = r
            distance = transh_distance(
                h,
                relation_translation,
                relation_normal,
                t,
                p=self.transh_norm,
                power_norm=self.transh_power_norm,
            )
            # PyKEEN ranks larger scores as more plausible.
            return -distance


    class PaperTransH(ERModel):
        """Auditable TransH baseline matching the TH-RotatE TransH branch."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                interaction=PaperTransHInteraction,
                interaction_kwargs=dict(
                    transh_norm=transh_norm,
                    transh_power_norm=transh_power_norm,
                ),
                entity_representations_kwargs=dict(
                    shape=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                ),
                relation_representations_kwargs=[
                    # 1) relation translation d_r
                    dict(shape=embedding_dim, initializer=nn.init.xavier_normal_),
                    # 2) relation-specific hyperplane normal w_r
                    dict(
                        shape=embedding_dim,
                        initializer=nn.init.xavier_normal_,
                        constrainer=normalize_last_dim,
                    ),
                ],
                **kwargs,
            )

    class THRotatEInteraction(Interaction):
        """Score-level fusion interaction from Paper4, defined at module scope for serialization."""

        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d")

        def __init__(
            self,
            *,
            transh_norm: int = 2,
            transh_power_norm: bool = True,
            rotate_norm: int = 2,
        ) -> None:
            super().__init__()
            self.transh_norm = transh_norm
            self.transh_power_norm = transh_power_norm
            self.rotate_norm = rotate_norm
            self.raw_fusion_weights = nn.Parameter(torch.zeros(2))

        @property
        def fusion_weights(self) -> Tensor:
            return normalized_positive_weights(self.raw_fusion_weights)

        def component_distances(self, h, r, t) -> tuple[Tensor, Tensor]:
            """Return unfused TransH and RotatE distances for scale auditing."""
            h_h, h_r = h
            d_r, w_r, r_r = r
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
            return d_h, d_rot

        def forward(self, h, r, t):
            d_h, d_rot = self.component_distances(h=h, r=r, t=t)
            alpha, beta = self.fusion_weights.unbind()
            return -(alpha * d_h + beta * d_rot)


    class ScaleCalibratedTHRotatEInteraction(Interaction):
        """A1-SC: Original TH-RotatE with training-only running-mean branch calibration."""

        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d")

        def __init__(
            self,
            *,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
            scale_epsilon: float = 1.0e-6,
        ) -> None:
            super().__init__()
            self.transh_norm = transh_norm
            self.transh_power_norm = transh_power_norm
            self.rotate_norm = rotate_norm
            self.scale_epsilon = float(scale_epsilon)
            self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
            self.register_buffer("transh_running_mean", torch.tensor(1.0))
            self.register_buffer("rotate_running_mean", torch.tensor(1.0))
            self.register_buffer("scale_update_count", torch.tensor(0, dtype=torch.long))

        @property
        def fusion_weights(self) -> Tensor:
            return normalized_positive_weights(self.raw_fusion_weights)

        def reset_parameters(self) -> None:
            with torch.no_grad():
                self.raw_fusion_weights.zero_()
                self.transh_running_mean.fill_(1.0)
                self.rotate_running_mean.fill_(1.0)
                self.scale_update_count.zero_()

        @torch.no_grad()
        def _update_scale_statistics(self, d_h: Tensor, d_rot: Tensor) -> None:
            batch_h = d_h.detach().float().mean().to(self.transh_running_mean.device)
            batch_r = d_rot.detach().float().mean().to(self.rotate_running_mean.device)
            self.scale_update_count.add_(1)
            weight = self.scale_update_count.to(dtype=self.transh_running_mean.dtype).reciprocal()
            self.transh_running_mean.add_(weight * (batch_h - self.transh_running_mean))
            self.rotate_running_mean.add_(weight * (batch_r - self.rotate_running_mean))

        def component_distances(self, h, r, t) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            h_h, h_r = h
            d_r, w_r, r_r = r
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
            if self.training:
                self._update_scale_statistics(d_h=d_h, d_rot=d_rot)
            scale_h = self.transh_running_mean.detach().clamp_min(self.scale_epsilon)
            scale_r = self.rotate_running_mean.detach().clamp_min(self.scale_epsilon)
            return d_h, d_rot, d_h / scale_h, d_rot / scale_r

        def forward(self, h, r, t):
            _d_h, _d_rot, d_h_cal, d_rot_cal = self.component_distances(h=h, r=r, t=t)
            alpha, beta = self.fusion_weights.unbind()
            return -(alpha * d_h_cal + beta * d_rot_cal)


    class ScaleCalibratedTHRotatE(ERModel):
        """A1-SC: score-scale calibrated TH-RotatE with Original global fusion weights."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
            scale_epsilon: float = 1.0e-6,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                interaction=ScaleCalibratedTHRotatEInteraction,
                interaction_kwargs=dict(
                    transh_norm=transh_norm,
                    transh_power_norm=transh_power_norm,
                    rotate_norm=rotate_norm,
                    scale_epsilon=scale_epsilon,
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
                ],
                **kwargs,
            )

        def get_fusion_weights(self) -> tuple[float, float]:
            weights = self.interaction.fusion_weights.detach().cpu().tolist()
            return float(weights[0]), float(weights[1])

        def get_calibration_state(self) -> dict[str, float | int]:
            return {
                "transh_running_mean": float(self.interaction.transh_running_mean.detach().cpu()),
                "rotate_running_mean": float(self.interaction.rotate_running_mean.detach().cpu()),
                "scale_ratio_transh_to_rotate": float(
                    self.interaction.transh_running_mean.detach().cpu()
                    / self.interaction.rotate_running_mean.detach().cpu().clamp_min(self.interaction.scale_epsilon)
                ),
                "scale_update_count": int(self.interaction.scale_update_count.detach().cpu()),
            }

        def score_components_hrt(self, hrt_batch: Tensor, *, mode=None) -> dict[str, Tensor]:
            h, r, t = self._get_representations(
                h=hrt_batch[:, 0],
                r=hrt_batch[:, 1],
                t=hrt_batch[:, 2],
                mode=mode,
            )
            d_h, d_rot, d_h_cal, d_rot_cal = self.interaction.component_distances(h=h, r=r, t=t)
            alpha, beta = self.interaction.fusion_weights.unbind()
            return {
                "transh_distance": d_h,
                "rotate_distance": d_rot,
                "calibrated_transh_distance": d_h_cal,
                "calibrated_rotate_distance": d_rot_cal,
                "fused_score": -(alpha * d_h_cal + beta * d_rot_cal),
            }


    class PositiveStepScaleCalibratedTHRotatEInteraction(Interaction):
        """v1.2 A1-SC: positive-only, optimizer-step-frozen scale calibration.

        Important: ``forward()`` and ``component_distances()`` are deliberately
        side-effect free. Scale updates are explicit and are invoked only by the
        v1.2 custom sLCWA training loop from the Training-positive batch.
        """

        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d")

        def __init__(
            self,
            *,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
            scale_epsilon: float = 1.0e-6,
        ) -> None:
            super().__init__()
            self.transh_norm = transh_norm
            self.transh_power_norm = transh_power_norm
            self.rotate_norm = rotate_norm
            self.scale_epsilon = float(scale_epsilon)
            self.raw_fusion_weights = nn.Parameter(torch.zeros(2))
            self.register_buffer("transh_running_mean", torch.tensor(1.0))
            self.register_buffer("rotate_running_mean", torch.tensor(1.0))
            self.register_buffer("scale_update_count", torch.tensor(0, dtype=torch.long))

        @property
        def fusion_weights(self) -> Tensor:
            return normalized_positive_weights(self.raw_fusion_weights)

        def reset_parameters(self) -> None:
            with torch.no_grad():
                self.raw_fusion_weights.zero_()
                self.transh_running_mean.fill_(1.0)
                self.rotate_running_mean.fill_(1.0)
                self.scale_update_count.zero_()

        def raw_component_distances(self, h, r, t) -> tuple[Tensor, Tensor]:
            h_h, h_r = h
            d_r, w_r, r_r = r
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
            return d_h, d_rot

        @torch.no_grad()
        def update_scale_from_positive_distances(self, d_h: Tensor, d_rot: Tensor) -> dict[str, float | int]:
            """Update cumulative means from one Training-positive batch only."""
            batch_h = d_h.detach().float().mean().to(self.transh_running_mean.device)
            batch_r = d_rot.detach().float().mean().to(self.rotate_running_mean.device)
            self.scale_update_count.add_(1)
            weight = self.scale_update_count.to(dtype=self.transh_running_mean.dtype).reciprocal()
            self.transh_running_mean.add_(weight * (batch_h - self.transh_running_mean))
            self.rotate_running_mean.add_(weight * (batch_r - self.rotate_running_mean))
            return {
                "positive_batch_transh_mean": float(batch_h.detach().cpu()),
                "positive_batch_rotate_mean": float(batch_r.detach().cpu()),
                "transh_running_mean": float(self.transh_running_mean.detach().cpu()),
                "rotate_running_mean": float(self.rotate_running_mean.detach().cpu()),
                "scale_update_count": int(self.scale_update_count.detach().cpu()),
            }

        def frozen_scale_tensors(self) -> tuple[Tensor, Tensor]:
            return (
                self.transh_running_mean.detach().clamp_min(self.scale_epsilon),
                self.rotate_running_mean.detach().clamp_min(self.scale_epsilon),
            )

        def component_distances(self, h, r, t) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            """Return raw and calibrated distances without changing calibration state."""
            d_h, d_rot = self.raw_component_distances(h=h, r=r, t=t)
            scale_h, scale_r = self.frozen_scale_tensors()
            return d_h, d_rot, d_h / scale_h, d_rot / scale_r

        def forward(self, h, r, t):
            _d_h, _d_rot, d_h_cal, d_rot_cal = self.component_distances(h=h, r=r, t=t)
            alpha, beta = self.fusion_weights.unbind()
            return -(alpha * d_h_cal + beta * d_rot_cal)


    class PositiveStepScaleCalibratedTHRotatE(ERModel):
        """v1.2 A1-SC with explicit Training-positive scale updates."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
            scale_epsilon: float = 1.0e-6,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                interaction=PositiveStepScaleCalibratedTHRotatEInteraction,
                interaction_kwargs=dict(
                    transh_norm=transh_norm,
                    transh_power_norm=transh_power_norm,
                    rotate_norm=rotate_norm,
                    scale_epsilon=scale_epsilon,
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
                ],
                **kwargs,
            )

        def get_fusion_weights(self) -> tuple[float, float]:
            weights = self.interaction.fusion_weights.detach().cpu().tolist()
            return float(weights[0]), float(weights[1])

        def get_calibration_state(self) -> dict[str, float | int | str]:
            scale_h = self.interaction.transh_running_mean.detach().cpu()
            scale_r = self.interaction.rotate_running_mean.detach().cpu()
            return {
                "transh_running_mean": float(scale_h),
                "rotate_running_mean": float(scale_r),
                "scale_ratio_transh_to_rotate": float(
                    scale_h / scale_r.clamp_min(self.interaction.scale_epsilon)
                ),
                "scale_update_count": int(self.interaction.scale_update_count.detach().cpu()),
                "scale_update_source": "training_positive_batch_only",
                "scoring_mutates_scale": "false",
            }

        @torch.no_grad()
        def update_scale_from_positive_batch(self, hrt_batch: Tensor, *, mode=None) -> dict[str, float | int]:
            """Update scale once from mapped Training-positive triples.

            This is intentionally a model-level explicit API so the training loop
            controls *when* state changes. Normal score calls never update scale.
            """
            if not self.training:
                raise RuntimeError("Positive-only scale updates are allowed only while the model is in training mode")
            if hrt_batch.ndim != 2 or hrt_batch.shape[-1] != 3:
                raise ValueError(f"Expected mapped positive triples with shape (n,3), got {tuple(hrt_batch.shape)}")
            h, r, t = self._get_representations(
                h=hrt_batch[:, 0],
                r=hrt_batch[:, 1],
                t=hrt_batch[:, 2],
                mode=mode,
            )
            d_h, d_rot = self.interaction.raw_component_distances(h=h, r=r, t=t)
            return self.interaction.update_scale_from_positive_distances(d_h=d_h, d_rot=d_rot)

        def get_frozen_scale_tensors(self) -> tuple[Tensor, Tensor]:
            return self.interaction.frozen_scale_tensors()

        def score_components_hrt(self, hrt_batch: Tensor, *, mode=None) -> dict[str, Tensor]:
            """Pure diagnostic scoring; this method never updates scale state."""
            h, r, t = self._get_representations(
                h=hrt_batch[:, 0],
                r=hrt_batch[:, 1],
                t=hrt_batch[:, 2],
                mode=mode,
            )
            d_h, d_rot, d_h_cal, d_rot_cal = self.interaction.component_distances(h=h, r=r, t=t)
            alpha, beta = self.interaction.fusion_weights.unbind()
            return {
                "transh_distance": d_h,
                "rotate_distance": d_rot,
                "calibrated_transh_distance": d_h_cal,
                "calibrated_rotate_distance": d_rot_cal,
                "fused_score": -(alpha * d_h_cal + beta * d_rot_cal),
            }


    class RelationAdaptiveTHRotatEInteraction(Interaction):
        """A1 relation-adaptive score fusion for heterogeneous railway KG relations."""

        entity_shape = ("d", "d")
        relation_shape = ("d", "d", "d")

        def __init__(
            self,
            *,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
        ) -> None:
            super().__init__()
            self.transh_norm = transh_norm
            self.transh_power_norm = transh_power_norm
            self.rotate_norm = rotate_norm
            self.relation_gate = nn.Linear(embedding_dim, 2)
            # Keep the gate at equal weights even when PyKEEN recursively resets modules.
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.zeros_(self.relation_gate.weight)
            nn.init.zeros_(self.relation_gate.bias)

        def relation_fusion_weights(self, relation_translation: Tensor) -> Tensor:
            raw = self.relation_gate(relation_translation)
            positive = F.softplus(raw) + 1.0e-12
            return positive / torch.linalg.vector_norm(
                positive, ord=2, dim=-1, keepdim=True
            )

        def component_distances(self, h, r, t) -> tuple[Tensor, Tensor, Tensor]:
            h_h, h_r = h
            d_r, w_r, r_r = r
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
            weights = self.relation_fusion_weights(d_r)
            return d_h, d_rot, weights

        def forward(self, h, r, t):
            d_h, d_rot, weights = self.component_distances(h=h, r=r, t=t)
            alpha_r, beta_r = weights.unbind(dim=-1)
            return -(alpha_r * d_h + beta_r * d_rot)


    class RelationAdaptiveTHRotatE(ERModel):
        """A1: Original TH-RotatE with relation-conditioned fusion instead of global alpha/beta."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = False,
            rotate_norm: int = 2,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                interaction=RelationAdaptiveTHRotatEInteraction,
                interaction_kwargs=dict(
                    embedding_dim=embedding_dim,
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
                ],
                **kwargs,
            )

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


    class THRotatE(ERModel):
        """Score-level fusion of independent TransH and RotatE embedding streams."""

        loss_default = NSSALoss

        def __init__(
            self,
            *,
            embedding_dim: int = 200,
            transh_norm: int = 2,
            transh_power_norm: bool = True,
            rotate_norm: int = 2,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                interaction=THRotatEInteraction,
                interaction_kwargs=dict(
                    transh_norm=transh_norm,
                    transh_power_norm=transh_power_norm,
                    rotate_norm=rotate_norm,
                ),
                entity_representations_kwargs=[
                    # TransH entity stream (real-valued)
                    dict(shape=embedding_dim, initializer=nn.init.xavier_normal_),
                    # RotatE entity stream (complex-valued)
                    dict(shape=embedding_dim, initializer=xavier_uniform_, dtype=torch.cfloat),
                ],
                relation_representations_kwargs=[
                    # TransH translation vector d_r
                    dict(shape=embedding_dim, initializer=nn.init.xavier_normal_),
                    # TransH relation-specific hyperplane normal w_r
                    dict(
                        shape=embedding_dim,
                        initializer=nn.init.xavier_normal_,
                        constrainer=normalize_last_dim,
                    ),
                    # RotatE phase relation
                    dict(
                        shape=embedding_dim,
                        initializer=init_phases,
                        constrainer=complex_normalize,
                        dtype=torch.cfloat,
                    ),
                ],
                **kwargs,
            )

        def get_fusion_weights(self) -> tuple[float, float]:
            weights = self.interaction.fusion_weights.detach().cpu().tolist()
            return float(weights[0]), float(weights[1])

        def score_components_hrt(self, hrt_batch: Tensor, *, mode=None) -> dict[str, Tensor]:
            """Score triples while exposing both branch distances for diagnostics."""
            h, r, t = self._get_representations(
                h=hrt_batch[:, 0],
                r=hrt_batch[:, 1],
                t=hrt_batch[:, 2],
                mode=mode,
            )
            d_h, d_rot = self.interaction.component_distances(h=h, r=r, t=t)
            alpha, beta = self.interaction.fusion_weights.unbind()
            return {
                "transh_distance": d_h,
                "rotate_distance": d_rot,
                "fused_score": -(alpha * d_h + beta * d_rot),
            }


else:
    PaperTransHInteraction = None
    PaperTransH = None
    THRotatEInteraction = None
    THRotatE = None
    ScaleCalibratedTHRotatEInteraction = None
    ScaleCalibratedTHRotatE = None
    PositiveStepScaleCalibratedTHRotatEInteraction = None
    PositiveStepScaleCalibratedTHRotatE = None
    RelationAdaptiveTHRotatEInteraction = None
    RelationAdaptiveTHRotatE = None


def build_pykeen_model_class():
    """Return the module-level PyKEEN model class, with a clear optional-dependency error."""
    if THRotatE is None:
        raise RuntimeError(
            "PyKEEN is not installed. Run `pip install -r requirements.txt` first."
        ) from _PYKEEN_IMPORT_ERROR
    return THRotatE


def build_paper_transh_model_class():
    """Return the explicit-order TransH class used for auditable baselines."""
    if PaperTransH is None:
        raise RuntimeError(
            "PyKEEN is not installed. Run `pip install -r requirements.txt` first."
        ) from _PYKEEN_IMPORT_ERROR
    return PaperTransH


def build_relation_adaptive_model_class():
    """Return the A1 relation-adaptive TH-RotatE model class."""
    if RelationAdaptiveTHRotatE is None:
        raise RuntimeError(
            "PyKEEN is not installed. Run `pip install -r requirements.txt` first."
        ) from _PYKEEN_IMPORT_ERROR
    return RelationAdaptiveTHRotatE


def build_scale_calibrated_model_class():
    """Return the A1-SC running-mean scale-calibrated TH-RotatE model class."""
    if ScaleCalibratedTHRotatE is None:
        raise RuntimeError(
            "PyKEEN is not installed. Run `pip install -r requirements.txt` first."
        ) from _PYKEEN_IMPORT_ERROR
    return ScaleCalibratedTHRotatE

def build_positive_step_scale_calibrated_model_class():
    """Return the v1.2 positive-only, step-frozen scale-calibrated model class."""
    if PositiveStepScaleCalibratedTHRotatE is None:
        raise RuntimeError(
            "PyKEEN is not installed. Run `pip install -r requirements.txt` first."
        ) from _PYKEEN_IMPORT_ERROR
    return PositiveStepScaleCalibratedTHRotatE

