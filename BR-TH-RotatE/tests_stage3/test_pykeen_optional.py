from __future__ import annotations

import pytest

pykeen = pytest.importorskip("pykeen")
import torch
from pykeen.triples import TriplesFactory

from stage3.models import build_compounde_model_class, build_rate_model_class


def _tiny_factory():
    triples = [
        ["a", "r", "b"],
        ["b", "r", "c"],
        ["c", "s", "a"],
    ]
    return TriplesFactory.from_labeled_triples(__import__("numpy").asarray(triples, dtype=str))


@pytest.mark.parametrize("builder", [build_rate_model_class, build_compounde_model_class])
def test_stage3_model_scores_have_expected_shapes(builder):
    tf = _tiny_factory()
    cls = builder()
    model = cls(triples_factory=tf, embedding_dim=8, random_seed=42)
    triples = tf.mapped_triples[:2]
    with torch.inference_mode():
        hrt = model.score_hrt(triples)
        tails = model.score_t(triples[:, :2])
        heads = model.score_h(triples[:, 1:])
    assert hrt.shape[0] == 2
    assert tails.shape == (2, tf.num_entities)
    assert heads.shape == (2, tf.num_entities)
