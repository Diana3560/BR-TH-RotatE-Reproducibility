from __future__ import annotations

import torch

from supplementary.per_relation_model import build_per_relation_control_bundle


def test_per_relation_control_is_exact_one_hot_lookup():
    relation_to_id = {f"r{i}": i for i in range(17)}
    modeled = [f"r{i}" for i in range(14)]
    bundle = build_per_relation_control_bundle(
        relation_to_id=relation_to_id,
        modeled_relations=modeled,
        num_internal_relations=34,
    )
    assert bundle.features.shape == (34, 28)
    assert bundle.reliability.shape == (34, 1)
    assert bundle.trainable_angle_parameters == 28
    assert len(bundle.rows) == 28
    assert torch.all(bundle.features[:28].sum(dim=1) == 1)
    assert torch.all(bundle.features[28:].sum(dim=1) == 0)
    assert torch.all(bundle.features.sum(dim=0) == 1)
    assert torch.all(bundle.reliability[:28] == 1)
    assert torch.all(bundle.reliability[28:] == 0)


def test_per_relation_control_handles_noncontiguous_modeled_relation_ids():
    relation_to_id = {"p0": 0, "a": 1, "p1": 2, "b": 3}
    bundle = build_per_relation_control_bundle(
        relation_to_id=relation_to_id,
        modeled_relations=["a", "b"],
        num_internal_relations=8,
    )
    active = torch.nonzero(bundle.features.sum(dim=1), as_tuple=False).flatten().tolist()
    assert active == [2, 3, 6, 7]
    assert bundle.trainable_angle_parameters == 4
