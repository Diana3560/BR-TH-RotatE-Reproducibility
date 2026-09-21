import math

import pytest
import torch

from throtate_repro.magnitude_preserving_normalization import (
    BoundedAdaptiveMagnitudePreservingTHRotatECore,
    magnitude_preserving_constants,
)


def test_means_must_be_positive():
    with pytest.raises(ValueError):
        magnitude_preserving_constants(0.0, 1.0)
    with pytest.raises(ValueError):
        magnitude_preserving_constants(1.0, float("nan"))


def test_geometric_mean_common_scale_and_multipliers():
    c = magnitude_preserving_constants(2.0, 8.0)
    assert math.isclose(c["common_scale_cStar"], 4.0)
    assert math.isclose(c["transh_multiplier"], 2.0)
    assert math.isclose(c["rotate_multiplier"], 0.5)


def test_rescaling_equalizes_calibration_means_without_collapsing_to_one():
    model = BoundedAdaptiveMagnitudePreservingTHRotatECore(
        transh_mean=2.0,
        rotate_mean=8.0,
        feature_dim=3,
        max_angle_deviation=0.15,
        transh_power_norm=False,
    )
    d_h = torch.tensor([1.0, 3.0])  # mean=2
    d_r = torch.tensor([4.0, 12.0])  # mean=8
    n_h, n_r = model.rescale_distances(d_h, d_r)
    assert math.isclose(float(n_h.mean()), 4.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(n_r.mean()), 4.0, rel_tol=0.0, abs_tol=1e-6)
    assert not math.isclose(float(n_h.mean()), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_equal_router_weights_at_zero_initialization():
    model = BoundedAdaptiveMagnitudePreservingTHRotatECore(
        transh_mean=1.0,
        rotate_mean=1.0,
        feature_dim=3,
        max_angle_deviation=0.15,
        transh_power_norm=False,
    )
    features = torch.zeros(5, 3)
    reliability = torch.ones(5, 1)
    weights, theta_global, shift = model.relation_fusion_weights(features, reliability)
    expected = 1.0 / math.sqrt(2.0)
    assert torch.allclose(weights[:, 0], torch.full((5,), expected), atol=1e-6)
    assert torch.allclose(weights[:, 1], torch.full((5,), expected), atol=1e-6)
    assert torch.allclose(shift, torch.zeros_like(shift))
    assert math.isclose(float(theta_global.detach()), math.pi / 4, rel_tol=0.0, abs_tol=1e-6)
