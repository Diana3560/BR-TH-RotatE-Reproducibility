from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from throtate_repro.model import (  # noqa: E402
    THRotatECore,
    build_paper_transh_model_class,
    build_pykeen_model_class,
    normalized_positive_weights,
    rotate_distance,
    transh_distance,
)


def test_equal_initial_weights():
    weights = normalized_positive_weights(torch.zeros(2))
    target = 1.0 / math.sqrt(2.0)
    assert torch.allclose(weights, torch.tensor([target, target]), atol=1e-6)
    assert torch.allclose((weights**2).sum(), torch.tensor(1.0), atol=1e-6)


def test_transh_zero_distance_on_exact_translation():
    h = torch.tensor([[1.0, 0.0]])
    w = torch.tensor([[0.0, 1.0]])
    d = torch.tensor([[2.0, 0.0]])
    t = torch.tensor([[3.0, 0.0]])
    dist = transh_distance(h, d, w, t, p=2, power_norm=True)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-7)


def test_rotate_zero_distance_on_exact_rotation():
    h = torch.tensor([[1.0 + 0.0j]])
    r = torch.tensor([[0.0 + 1.0j]])
    t = torch.tensor([[0.0 + 1.0j]])
    dist = rotate_distance(h, r, t, p=2)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-7)


def test_fused_score_prefers_true_triple():
    model = THRotatECore()
    h_h = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    d_r = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    w_r = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    t_h = torch.tensor([[3.0, 0.0], [7.0, 0.0]])
    h_r = torch.tensor([[1.0 + 0j], [1.0 + 0j]])
    r_r = torch.tensor([[0.0 + 1j], [0.0 + 1j]])
    t_r = torch.tensor([[0.0 + 1j], [1.0 + 0j]])
    scores = model((h_h, h_r), (d_r, w_r, r_r), (t_h, t_r))
    assert scores[0] > scores[1]


def test_pykeen_model_class_is_pickle_addressable_when_available():
    try:
        model_class = build_pykeen_model_class()
    except RuntimeError as exc:
        if "PyKEEN is not installed" in str(exc):
            return
        raise
    assert "<locals>" not in model_class.__qualname__
    assert model_class.__module__ == "throtate_repro.model"


def test_paper_transh_class_is_pickle_addressable_when_available():
    try:
        model_class = build_paper_transh_model_class()
    except RuntimeError as exc:
        if "PyKEEN is not installed" in str(exc):
            return
        raise
    assert "<locals>" not in model_class.__qualname__
    assert model_class.__module__ == "throtate_repro.model"


def test_squared_and_unsquared_transh_distance_are_explicitly_distinct():
    h = torch.tensor([[0.0, 0.0]])
    w = torch.tensor([[0.0, 1.0]])
    d = torch.tensor([[3.0, 0.0]])
    t = torch.tensor([[1.0, 0.0]])
    unsquared = transh_distance(h, d, w, t, p=2, power_norm=False)
    squared = transh_distance(h, d, w, t, p=2, power_norm=True)
    assert torch.allclose(unsquared, torch.tensor([2.0]))
    assert torch.allclose(squared, torch.tensor([4.0]))
