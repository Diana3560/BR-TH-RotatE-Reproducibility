from __future__ import annotations

import torch

from throtate_repro.feature_ablation import remove_business_role_features


def test_remove_business_role_features_only_zeros_two_named_columns():
    names = (
        "role_diagnostic",
        "role_procedural",
        "z_log_train_frequency",
        "z_log_tph",
        "direction_is_inverse",
    )
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5, 2.0, 0.0],
            [0.0, 1.0, -0.5, 3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y, audit = remove_business_role_features(x, names)
    assert torch.equal(y[:, :2], torch.zeros_like(y[:, :2]))
    assert torch.equal(y[:, 2:], x[:, 2:])
    assert audit.removed_feature_indices == (0, 1)
    assert audit.nonzero_removed_before == 2
    assert audit.nonzero_removed_after == 0
    assert audit.unchanged_other_columns is True
