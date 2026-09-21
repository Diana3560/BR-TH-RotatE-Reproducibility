from __future__ import annotations

from pathlib import Path

import yaml

from throtate_repro.v266_protocol import (
    SEEDS,
    VARIANTS,
    build_progress_summary,
    exact_step_budget,
    seed42_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/multidataset_comparison.yaml"


def _row(variant: str, seed: int, mrr: float, hits1: float) -> dict:
    return {
        "status": "PASS",
        "variant": variant,
        "seed": seed,
        "actual_optimizer_steps": 3000,
        "test_loaded": False,
        "validation_metrics": {
            "mrr": mrr,
            "mr": 250.0,
            "hits_at_1": hits1,
            "hits_at_3": 0.57,
            "hits_at_10": 0.71,
        },
    }


def test_equal_full_batch_exposure_budget() -> None:
    ordinary = exact_step_budget(
        num_real_triples=15413,
        batch_size=1024,
        max_steps=3000,
        reciprocal=False,
        drop_last=True,
    )
    reciprocal = exact_step_budget(
        num_real_triples=15413,
        batch_size=1024,
        max_steps=3000,
        reciprocal=True,
        drop_last=True,
    )
    assert ordinary["steps_per_full_epoch"] == 15
    assert ordinary["num_epochs"] == 200
    assert ordinary["discarded_incomplete_batch_instances_per_full_epoch"] == 53
    assert reciprocal["steps_per_full_epoch"] == 30
    assert reciprocal["num_epochs"] == 100
    assert reciprocal["discarded_incomplete_batch_instances_per_full_epoch"] == 106
    assert ordinary["exact_positive_exposure"] == reciprocal["exact_positive_exposure"] == 3_072_000


def test_seed42_gate_requires_D2_to_beat_both() -> None:
    rows = {
        "D0": _row("D0", 42, 0.510, 0.420),
        "D1": _row("D1", 42, 0.515, 0.425),
        "D2": _row("D2", 42, 0.526, 0.435),
    }
    result = seed42_gate(rows, max_steps=3000, relative_target=0.02)
    assert result["decision"] == "CONTINUE"
    assert result["checks"]["direction_gate_pass"] is True
    rows["D2"] = _row("D2", 42, 0.514, 0.436)
    assert seed42_gate(rows, max_steps=3000, relative_target=0.02)["decision"] == "STOP"


def test_complete_five_seed_summary() -> None:
    rows = []
    for seed in SEEDS:
        rows.extend(
            [
                _row("D0", seed, 0.500, 0.400),
                _row("D1", seed, 0.505, 0.405),
                _row("D2", seed, 0.515, 0.415),
            ]
        )
    summary = build_progress_summary(rows, max_steps=3000, relative_target=0.02)
    assert summary["status"] == "COMPLETE"
    assert summary["completed_run_count"] == 15
    assert summary["formal_success_checks"]["formal_success"] is True
    assert summary["paired_comparisons"]["D2_vs_D1"]["wins_both_primary_metrics"] == 5


def test_current_config_preserves_equal_budget_and_validation_only_selection() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["training"]["max_steps"] == 3000
    assert cfg["training"]["batch_size"] == 1024
    assert cfg["training"]["num_negatives"] == 64
    assert cfg["training"]["drop_last"] is True
    assert cfg["training"]["negative_sampler_filtered"] is False
    assert cfg["protocol"]["validation_only_parameter_selection"] is True
    assert cfg["protocol"]["no_test_parameter_selection"] is True
    assert cfg["formal"]["seeds"] == [42, 43, 44, 45, 46]
