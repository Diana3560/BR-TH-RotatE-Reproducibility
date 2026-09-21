from __future__ import annotations

import pytest
import torch

from throtate_repro.reciprocal_v263 import ReciprocalFeatureBundle
from throtate_repro.reliability_ablation import build_no_shrinkage_reciprocal_bundle


def _bundle(reliability_values: list[float]) -> ReciprocalFeatureBundle:
    reliability = torch.tensor(reliability_values, dtype=torch.float32).reshape(-1, 1)
    features = torch.arange(len(reliability_values) * 3, dtype=torch.float32).reshape(
        len(reliability_values), 3
    )
    rows = [
        {
            "relation": f"r{i // 2}",
            "real_relation_id": i // 2,
            "internal_relation_id": i,
            "direction": "forward" if i % 2 == 0 else "inverse",
            "reliability_q": float(reliability[i, 0]),
        }
        for i in range(len(reliability_values))
    ]
    return ReciprocalFeatureBundle(
        features=features,
        reliability=reliability,
        feature_names=("f1", "f2", "direction_is_inverse"),
        rows=rows,
    )


def test_no_shrinkage_sets_only_modeled_forward_inverse_q_to_one() -> None:
    source = _bundle([0.25, 0.25, 0.80, 0.80, 0.0, 0.0])
    modified, audit = build_no_shrinkage_reciprocal_bundle(
        bundle=source,
        relation_to_id={"R_A": 0, "R_B": 1, "P3": 2},
        modeled_relations=("R_A", "R_B"),
    )

    assert modified.reliability[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    assert source.reliability[:, 0].tolist() == pytest.approx([0.25, 0.25, 0.80, 0.80, 0.0, 0.0])
    assert torch.equal(modified.features, source.features)
    assert audit["modified_internal_relation_rows"] == 4
    assert audit["feature_tensor_unchanged"] is True
    assert audit["modified_modeled_q_unique"] == [1.0]
    assert audit["non_modeled_reliability_unchanged"] is True


def test_no_shrinkage_rejects_zero_q_for_modeled_relation() -> None:
    source = _bundle([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Expected positive Training-only reliability"):
        build_no_shrinkage_reciprocal_bundle(
            bundle=source,
            relation_to_id={"R_A": 0, "P3": 1},
            modeled_relations=("R_A",),
        )
