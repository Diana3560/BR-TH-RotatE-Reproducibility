from __future__ import annotations

from dataclasses import dataclass

import torch

from stage3.candidate_eval import realistic_rank_from_scores, metric_values_from_ranks


def test_realistic_rank_tie_handling():
    # Target at index 1 has score 2.0. One score is strictly larger, one ties.
    scores = torch.tensor([3.0, 2.0, 2.0, 0.0])
    # optimistic = 2; pessimistic = 3; realistic = 2.5
    assert realistic_rank_from_scores(scores, 1) == 2.5


def test_rank_metrics_known_vector():
    metrics = metric_values_from_ranks([1.0, 2.0, 4.0, 10.0])
    assert metrics["mr"] == 4.25
    assert abs(metrics["mrr"] - ((1 + 0.5 + 0.25 + 0.1) / 4)) < 1e-12
    assert metrics["hits_at_1"] == 0.25
    assert metrics["hits_at_3"] == 0.5
    assert metrics["hits_at_10"] == 1.0
