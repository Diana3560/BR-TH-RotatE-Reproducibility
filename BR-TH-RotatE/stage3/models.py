from __future__ import annotations

"""Stage-3 external baseline models.

This module is additive. It does not modify any Stage-2 implementation.

RatE
----
Implements the relation-adaptive weighted complex product scoring architecture from
Huang et al. (COLING 2020), Eq. (3.2)-(3.4):
    score(h,r,t) = -|| h \\odot_{W(r)} r - t ||_1
with eight relation-specific scalar weights. The Stage-3 experiment deliberately uses
this scoring architecture under the *same* NSSA/Bernoulli/optimizer-step protocol as
our other baselines so that the comparison isolates model architecture rather than a
change in training budget/sampler.

CompoundE
---------
Implements the translation + rotation + scaling compound scoring architecture from
Ge et al. (ACL 2023). The model is adapted to PyKEEN's ERModel interface and uses the
same unified Stage-3 training protocol. The geometric operation follows the public
CompoundE implementation: normalize entities, rotate 2-D coordinate pairs of the
candidate tail, translate, scale, then use negative L1 distance to the head.
"""

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
except ImportError as exc:  # allow PyTorch-only unit tests without PyKEEN installed
    NSSALoss = ERModel = Interaction = None
    init_phases = xavier_uniform_ = complex_normalize = None
    _PYKEEN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKEEN_IMPORT_ERROR = None


def rate_standard_weight_tensor(*, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    """Return the 2x4 matrix that makes RatE's weighted product equal complex multiplication."""
    return torch.tensor(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )


def rate_weight_initializer(tensor: Tensor) -> Tensor:
    """Initialize each relation's eight weights to the ordinary complex-product matrix.

    Starting from RotatE's complex product gives a stable, interpretable initialization;
    the eight scalars remain fully learnable and relation-specific after initialization.
    """
    base = rate_standard_weight_tensor(dtype=tensor.dtype, device=tensor.device).reshape(-1)
    with torch.no_grad():
        tensor.copy_(base.expand_as(tensor))
    return tensor


def rate_weighted_product(h: Tensor, r: Tensor, relation_weights: Tensor) -> Tensor:
    """Element-wise relation-adaptive complex weighted product.

    Parameters
    ----------
    h, r:
        Complex tensors with a shared final embedding dimension.
    relation_weights:
        Real tensor with final dimension 8, reshaped to a relation-specific 2x4 matrix.
        Leading dimensions follow PyKEEN broadcasting conventions.
    """
    if not torch.is_complex(h) or not torch.is_complex(r):
        raise TypeError("RatE h and r must be complex tensors")
    if relation_weights.shape[-1] != 8:
        raise ValueError(f"RatE relation_weights must end in 8, got {tuple(relation_weights.shape)}")

    a, b = h.real, h.imag
    c, d = r.real, r.imag
    # s(u,v) = [ac, ad, bc, bd], shape (..., embedding_dim, 4)
    features = torch.stack((a * c, a * d, b * c, b * d), dim=-1)
    weights = relation_weights.reshape(*relation_weights.shape[:-1], 2, 4)
    # (..., embedding_dim, 4) @ (..., 4, 2) -> (..., embedding_dim, 2)
    output = torch.matmul(features, weights.transpose(-1, -2))
    return torch.complex(output[..., 0], output[..., 1])


def rate_score(h: Tensor, r: Tensor, relation_weights: Tensor, t: Tensor) -> Tensor:
    """RatE plausibility score (larger is better): negative complex L1 distance."""
    translated = rate_weighted_product(h=h, r=r, relation_weights=relation_weights)
    return -torch.linalg.vector_norm(translated - t, ord=1, dim=-1)


def compounde_transform_tail(
    tail: Tensor,
    scale: Tensor,
    translation: Tensor,
    theta: Tensor,
) -> Tensor:
    """Apply CompoundE tail-side rotation -> translation -> scaling.

    Entity coordinates are paired as (x, y). ``theta`` therefore has d/2 values while
    scale and translation have d values. This is mathematically equivalent to the public
    CompoundE forward operation while avoiding the public implementation's unused second
    half of its theta chunk.
    """
    if tail.shape[-1] % 2:
        raise ValueError("CompoundE requires an even entity embedding dimension")
    if scale.shape[-1] != tail.shape[-1] or translation.shape[-1] != tail.shape[-1]:
        raise ValueError("CompoundE scale/translation dimensions must equal the entity dimension")
    if theta.shape[-1] * 2 != tail.shape[-1]:
        raise ValueError("CompoundE theta dimension must be half the entity dimension")

    tail = F.normalize(tail, p=2, dim=-1)
    pairs = tail.reshape(*tail.shape[:-1], -1, 2)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    x = pairs[..., 0:1]
    y = pairs[..., 1:2]
    rotated = torch.cat((cos_theta * x - sin_theta * y, sin_theta * x + cos_theta * y), dim=-1)
    # PyKEEN score_t() broadcasts relation parameters with all candidate tails. In that
    # case ``tail`` may be (1, num_candidates, d) while theta/scale/translation are
    # (batch_size, 1, ...), so ``rotated`` becomes
    # (batch_size, num_candidates, d/2, 2). Reshaping back to ``tail.shape`` incorrectly
    # discards the broadcasted query batch dimension and raises a size mismatch.
    # Flatten only the final pair coordinates, preserving all broadcasted leading dims.
    rotated = rotated.flatten(start_dim=-2)
    return (rotated + translation) * scale


def compounde_score(
    h: Tensor,
    scale: Tensor,
    translation: Tensor,
    theta: Tensor,
    t: Tensor,
) -> Tensor:
    """CompoundE-style plausibility score under the unified protocol: negative L1 distance."""
    h = F.normalize(h, p=2, dim=-1)
    transformed_tail = compounde_transform_tail(
        tail=t,
        scale=scale,
        translation=translation,
        theta=theta,
    )
    return -torch.linalg.vector_norm(h - transformed_tail, ord=1, dim=-1)


def _angle_initializer(tensor: Tensor) -> Tensor:
    return nn.init.uniform_(tensor, a=-math.pi, b=math.pi)


def _scale_initializer(tensor: Tensor) -> Tensor:
    # Identity-centered initialization makes the initial transformation well-conditioned.
    with torch.no_grad():
        tensor.fill_(1.0)
        tensor.add_(0.01 * torch.randn_like(tensor))
    return tensor


if Interaction is not None:

    class RatEInteraction(Interaction):
        entity_shape = ("d",)
        relation_shape = ("d", "w")

        def forward(self, h, r, t):
            relation_embedding, relation_weights = r
            return rate_score(
                h=h,
                r=relation_embedding,
                relation_weights=relation_weights,
                t=t,
            )


    class RatE(ERModel):
        """RatE scoring architecture adapted to the project's unified PyKEEN protocol."""

        loss_default = NSSALoss

        def __init__(self, *, embedding_dim: int = 200, **kwargs: Any) -> None:
            super().__init__(
                interaction=RatEInteraction,
                entity_representations_kwargs=dict(
                    shape=embedding_dim,
                    initializer=xavier_uniform_,
                    dtype=torch.cfloat,
                ),
                relation_representations_kwargs=[
                    dict(
                        shape=embedding_dim,
                        initializer=init_phases,
                        constrainer=complex_normalize,
                        dtype=torch.cfloat,
                    ),
                    dict(shape=8, initializer=rate_weight_initializer),
                ],
                **kwargs,
            )


    class CompoundEInteraction(Interaction):
        entity_shape = ("d",)
        relation_shape = ("d", "d", "a")

        def forward(self, h, r, t):
            scale, translation, theta = r
            return compounde_score(
                h=h,
                scale=scale,
                translation=translation,
                theta=theta,
                t=t,
            )


    class CompoundE(ERModel):
        """CompoundE geometric scoring architecture adapted to the unified protocol."""

        loss_default = NSSALoss

        def __init__(self, *, embedding_dim: int = 200, **kwargs: Any) -> None:
            if embedding_dim % 2:
                raise ValueError("CompoundE requires an even embedding_dim")
            super().__init__(
                interaction=CompoundEInteraction,
                entity_representations_kwargs=dict(
                    shape=embedding_dim,
                    initializer=nn.init.xavier_normal_,
                ),
                relation_representations_kwargs=[
                    dict(shape=embedding_dim, initializer=_scale_initializer),
                    dict(shape=embedding_dim, initializer=nn.init.zeros_),
                    dict(shape=embedding_dim // 2, initializer=_angle_initializer),
                ],
                **kwargs,
            )

else:
    RatEInteraction = RatE = None
    CompoundEInteraction = CompoundE = None


def build_rate_model_class():
    if RatE is None:
        raise RuntimeError("PyKEEN is not installed. Run `python -m pip install -r requirements.txt`.") from _PYKEEN_IMPORT_ERROR
    return RatE


def build_compounde_model_class():
    if CompoundE is None:
        raise RuntimeError("PyKEEN is not installed. Run `python -m pip install -r requirements.txt`.") from _PYKEEN_IMPORT_ERROR
    return CompoundE
