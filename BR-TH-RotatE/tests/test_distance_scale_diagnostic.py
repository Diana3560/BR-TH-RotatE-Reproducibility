from __future__ import annotations

import pytest
import torch

from throtate_repro.distance_scale_diagnostic import component_summary, translation_normal_alignment


def test_component_summary_reports_raw_and_weighted_ratios() -> None:
    row = component_summary(
        transh_distance=torch.tensor([2.0, 4.0]),
        rotate_distance=torch.tensor([1.0, 2.0]),
        fusion_alpha=torch.tensor([0.5, 0.5]),
        fusion_beta=torch.tensor([1.0, 1.0]),
    )
    assert row["raw_mean_ratio_transh_over_rotate"] == pytest.approx(2.0)
    assert row["weighted_mean_ratio_transh_over_rotate"] == pytest.approx(1.0)
    assert row["mean_transh_share_of_fused_distance"] == pytest.approx(0.5)


def test_translation_normal_alignment_does_not_assume_orthogonality() -> None:
    d = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    w = torch.tensor([[1.0, 0.0], [1.0, -1.0]])
    row = translation_normal_alignment(d, w)
    assert row["constraint_enforced"] is False
    assert row["absolute_cosine"]["max"] == pytest.approx(1.0)
    assert row["absolute_cosine"]["min"] == pytest.approx(0.0)
