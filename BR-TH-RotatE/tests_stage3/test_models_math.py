from __future__ import annotations

import torch

from stage3.models import (
    compounde_score,
    compounde_transform_tail,
    rate_standard_weight_tensor,
    rate_weight_initializer,
    rate_weighted_product,
)


def test_rate_standard_weights_reproduce_complex_multiplication():
    generator = torch.Generator().manual_seed(7)
    h = torch.complex(torch.randn(3, 5, generator=generator), torch.randn(3, 5, generator=generator))
    r = torch.complex(torch.randn(3, 5, generator=generator), torch.randn(3, 5, generator=generator))
    weights = rate_standard_weight_tensor().reshape(1, 8).expand(3, 8).clone()
    actual = rate_weighted_product(h=h, r=r, relation_weights=weights)
    expected = h * r
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rate_initializer_broadcasts_to_relation_table():
    table = torch.empty(11, 8)
    rate_weight_initializer(table)
    expected = rate_standard_weight_tensor().reshape(1, 8).expand_as(table)
    assert torch.equal(table, expected)


def test_compounde_identity_transform_is_normalized_tail():
    generator = torch.Generator().manual_seed(11)
    tail = torch.randn(4, 8, generator=generator)
    scale = torch.ones_like(tail)
    translation = torch.zeros_like(tail)
    theta = torch.zeros(4, 4)
    actual = compounde_transform_tail(tail, scale, translation, theta)
    expected = torch.nn.functional.normalize(tail, p=2, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_compounde_identity_scores_zero_for_same_normalized_entity():
    generator = torch.Generator().manual_seed(13)
    entity = torch.randn(4, 8, generator=generator)
    normalized = torch.nn.functional.normalize(entity, p=2, dim=-1)
    score = compounde_score(
        h=normalized,
        scale=torch.ones_like(normalized),
        translation=torch.zeros_like(normalized),
        theta=torch.zeros(4, 4),
        t=normalized,
    )
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-6, rtol=1e-6)


def test_compounde_transform_supports_pykeen_score_t_broadcasting():
    """Regression test for batched queries scored against all candidate tails."""
    generator = torch.Generator().manual_seed(17)
    batch_size, num_candidates, dim = 10, 37, 8
    tail = torch.randn(1, num_candidates, dim, generator=generator)
    scale = torch.ones(batch_size, 1, dim)
    translation = torch.zeros(batch_size, 1, dim)
    theta = torch.zeros(batch_size, 1, dim // 2)

    actual = compounde_transform_tail(tail, scale, translation, theta)
    expected_tail = torch.nn.functional.normalize(tail, p=2, dim=-1)
    expected = expected_tail.expand(batch_size, num_candidates, dim)

    assert actual.shape == (batch_size, num_candidates, dim)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_compounde_score_supports_pykeen_score_t_broadcasting():
    """Regression test for the exact score_t-style broadcast that previously crashed."""
    generator = torch.Generator().manual_seed(19)
    batch_size, num_candidates, dim = 10, 37, 8
    h = torch.randn(batch_size, 1, dim, generator=generator)
    tail = torch.randn(1, num_candidates, dim, generator=generator)
    scale = torch.ones(batch_size, 1, dim)
    translation = torch.zeros(batch_size, 1, dim)
    theta = torch.zeros(batch_size, 1, dim // 2)

    score = compounde_score(h=h, scale=scale, translation=translation, theta=theta, t=tail)

    assert score.shape == (batch_size, num_candidates)
    assert torch.isfinite(score).all()
