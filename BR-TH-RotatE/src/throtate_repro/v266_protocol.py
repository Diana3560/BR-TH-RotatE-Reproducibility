from __future__ import annotations

"""Pure protocol logic for the staged v2.6.6 equal-budget ablation.

This module deliberately has no PyTorch/PyKEEN import.  It defines the frozen
experiment grid, exact-step budgets, the seed-42 direction gate, and the final
five-seed aggregation used by the Windows runners.
"""

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("D0", "D1", "D2")
PRIMARY_METRICS = ("mrr", "hits_at_1")
ALL_METRICS = ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10")
VARIANT_LABELS = {
    "D0": "Corrected TH-RotatE; ordinary training",
    "D1": "Corrected TH-RotatE; reciprocal training",
    "D2": "bounded relation adapter; reciprocal training",
}


def exact_step_budget(
    *,
    num_real_triples: int,
    batch_size: int,
    max_steps: int,
    reciprocal: bool,
    drop_last: bool,
) -> dict[str, Any]:
    """Return an exact optimizer-step plan.

    With ``drop_last=True`` every optimizer update contains exactly ``batch_size``
    positives.  Thus ordinary and reciprocal variants have identical positive and
    negative exposure, not merely the same nominal number of optimizer updates.
    """
    num_real_triples = int(num_real_triples)
    batch_size = int(batch_size)
    max_steps = int(max_steps)
    if num_real_triples <= 0 or batch_size <= 0 or max_steps <= 0:
        raise ValueError("num_real_triples, batch_size, and max_steps must be positive")
    multiplier = 2 if reciprocal else 1
    effective_instances = multiplier * num_real_triples
    if drop_last:
        steps_per_full_epoch = effective_instances // batch_size
    else:
        steps_per_full_epoch = math.ceil(effective_instances / batch_size)
    if steps_per_full_epoch <= 0:
        raise ValueError("batch_size is larger than the effective training set while drop_last=True")
    full_epochs, remainder = divmod(max_steps, steps_per_full_epoch)
    num_epochs = full_epochs + (1 if remainder else 0)
    final_epoch_batches = remainder if remainder else steps_per_full_epoch
    discarded_per_full_epoch = effective_instances % batch_size if drop_last else 0
    exact_positive_exposure = max_steps * batch_size if drop_last else None
    return {
        "num_real_training_triples": num_real_triples,
        "reciprocal_multiplier": multiplier,
        "effective_training_instances": effective_instances,
        "batch_size": batch_size,
        "drop_last": bool(drop_last),
        "discarded_incomplete_batch_instances_per_full_epoch": discarded_per_full_epoch,
        "steps_per_full_epoch": steps_per_full_epoch,
        "full_epochs": full_epochs,
        "final_epoch_batches": final_epoch_batches,
        "num_epochs": num_epochs,
        "exact_optimizer_steps": max_steps,
        "exact_positive_exposure": exact_positive_exposure,
    }


def _require_passed_result(row: Mapping[str, Any], *, variant: str, seed: int, max_steps: int) -> None:
    if row.get("status") != "PASS":
        raise ValueError(f"{variant}/seed{seed} is not PASS")
    if str(row.get("variant")) != variant or int(row.get("seed", -1)) != int(seed):
        raise ValueError(f"result identity mismatch for {variant}/seed{seed}")
    if int(row.get("actual_optimizer_steps", -1)) != int(max_steps):
        raise ValueError(f"optimizer-step audit mismatch for {variant}/seed{seed}")
    if row.get("test_loaded") is not False:
        raise ValueError(f"Test seal audit failed for {variant}/seed{seed}")
    metrics = row.get("validation_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Validation metrics missing for {variant}/seed{seed}")
    for metric in ALL_METRICS:
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric} for {variant}/seed{seed}")


def seed42_gate(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    max_steps: int,
    relative_target: float,
) -> dict[str, Any]:
    """Apply the pre-declared seed-42 direction gate without running more training."""
    for variant in VARIANTS:
        if variant not in rows:
            raise ValueError(f"missing seed42 result: {variant}")
        _require_passed_result(rows[variant], variant=variant, seed=42, max_steps=max_steps)
    metrics = {
        variant: {key: float(rows[variant]["validation_metrics"][key]) for key in ALL_METRICS}
        for variant in VARIANTS
    }
    d2_minus_d0 = {key: metrics["D2"][key] - metrics["D0"][key] for key in PRIMARY_METRICS}
    d2_minus_d1 = {key: metrics["D2"][key] - metrics["D1"][key] for key in PRIMARY_METRICS}
    relative_d2_vs_d0 = {
        key: metrics["D2"][key] / metrics["D0"][key] - 1.0 for key in PRIMARY_METRICS
    }
    beats_d0_both = all(d2_minus_d0[key] > 0.0 for key in PRIMARY_METRICS)
    beats_d1_both = all(d2_minus_d1[key] > 0.0 for key in PRIMARY_METRICS)
    direction_pass = beats_d0_both and beats_d1_both
    reaches_target = all(relative_d2_vs_d0[key] >= float(relative_target) for key in PRIMARY_METRICS)
    return {
        "status": "PASS",
        "phase": "v2.6.6_seed42_direction_gate",
        "seed": 42,
        "decision": "CONTINUE" if direction_pass else "STOP",
        "decision_rule": "D2 must beat both D0 and D1 on both MRR and Hits@1",
        "metrics": metrics,
        "D2_minus_D0": d2_minus_d0,
        "D2_minus_D1": d2_minus_d1,
        "relative_D2_vs_D0": relative_d2_vs_d0,
        "checks": {
            "D2_beats_D0_on_both_primary_metrics": beats_d0_both,
            "D2_beats_D1_on_both_primary_metrics": beats_d1_both,
            "direction_gate_pass": direction_pass,
            "seed42_reaches_relative_target_on_both_primary_metrics": reaches_target,
        },
        "relative_target": float(relative_target),
        "interpretation": (
            "Direction is positive; seeds 43-46 are permitted but remain manual."
            if direction_pass
            else "Direction is not supported; do not spend compute on seeds 43-46."
        ),
        "next_stage_is_manual": True,
        "test_loaded": False,
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "std_sample": None, "values": []}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _paired_comparison(
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    deltas: dict[str, list[float]] = {key: [] for key in ALL_METRICS}
    relative: dict[str, list[float]] = {key: [] for key in PRIMARY_METRICS}
    both_primary_wins = 0
    by_seed = []
    for seed in SEEDS:
        c = lookup[(candidate, seed)]["validation_metrics"]
        r = lookup[(reference, seed)]["validation_metrics"]
        seed_delta = {key: float(c[key]) - float(r[key]) for key in ALL_METRICS}
        for key, value in seed_delta.items():
            deltas[key].append(value)
        seed_relative = {
            key: float(c[key]) / float(r[key]) - 1.0 for key in PRIMARY_METRICS
        }
        for key, value in seed_relative.items():
            relative[key].append(value)
        wins_both = all(seed_delta[key] > 0.0 for key in PRIMARY_METRICS)
        both_primary_wins += int(wins_both)
        by_seed.append(
            {
                "seed": seed,
                "delta": seed_delta,
                "relative_primary_gain": seed_relative,
                "wins_both_primary_metrics": wins_both,
            }
        )
    return {
        "candidate": candidate,
        "reference": reference,
        "absolute_delta": {key: _summary(values) for key, values in deltas.items()},
        "relative_primary_gain": {key: _summary(values) for key, values in relative.items()},
        "wins_both_primary_metrics": both_primary_wins,
        "required_wins_both_primary_metrics": 4,
        "by_seed": by_seed,
    }


def build_progress_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
    relative_target: float,
) -> dict[str, Any]:
    """Build a partial or complete summary from independently produced run files."""
    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    invalid = []
    for row in rows:
        key = (str(row.get("variant")), int(row.get("seed", -1)))
        if key in lookup:
            invalid.append({"key": list(key), "reason": "duplicate result"})
            continue
        try:
            _require_passed_result(row, variant=key[0], seed=key[1], max_steps=max_steps)
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append({"key": list(key), "reason": str(exc)})
            continue
        lookup[key] = row
    expected = [(variant, seed) for seed in SEEDS for variant in VARIANTS]
    missing = [list(key) for key in expected if key not in lookup]
    completed = [list(key) for key in expected if key in lookup]
    aggregated: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_rows = [lookup[(variant, seed)] for seed in SEEDS if (variant, seed) in lookup]
        aggregated[variant] = {
            "label": VARIANT_LABELS[variant],
            "n": len(variant_rows),
            "seeds": [int(row["seed"]) for row in variant_rows],
            "metrics": {
                metric: _summary([float(row["validation_metrics"][metric]) for row in variant_rows])
                for metric in ALL_METRICS
            },
        }
    complete = not missing and not invalid
    comparisons: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    if complete:
        comparisons = {
            "D1_vs_D0": _paired_comparison(lookup, candidate="D1", reference="D0"),
            "D2_vs_D0": _paired_comparison(lookup, candidate="D2", reference="D0"),
            "D2_vs_D1": _paired_comparison(lookup, candidate="D2", reference="D1"),
        }
        means = {
            variant: {
                metric: float(aggregated[variant]["metrics"][metric]["mean"])
                for metric in PRIMARY_METRICS
            }
            for variant in VARIANTS
        }
        relative_mean = {
            metric: means["D2"][metric] / means["D0"][metric] - 1.0
            for metric in PRIMARY_METRICS
        }
        checks = {
            "D2_mean_beats_D0_on_both_primary_metrics": all(
                means["D2"][metric] > means["D0"][metric] for metric in PRIMARY_METRICS
            ),
            "D2_mean_beats_D1_on_both_primary_metrics": all(
                means["D2"][metric] > means["D1"][metric] for metric in PRIMARY_METRICS
            ),
            "D2_relative_mean_gain_vs_D0_reaches_target_on_both": all(
                relative_mean[metric] >= float(relative_target) for metric in PRIMARY_METRICS
            ),
            "D2_beats_D0_on_both_in_at_least_4_of_5_seeds": (
                comparisons["D2_vs_D0"]["wins_both_primary_metrics"] >= 4
            ),
            "D2_beats_D1_on_both_in_at_least_4_of_5_seeds": (
                comparisons["D2_vs_D1"]["wins_both_primary_metrics"] >= 4
            ),
            "relative_D2_mean_vs_D0": relative_mean,
        }
        checks["formal_success"] = all(
            value for key, value in checks.items() if key != "relative_D2_mean_vs_D0"
        )
    return {
        "status": "COMPLETE" if complete else "PARTIAL",
        "phase": "v2.6.6_equal_budget_ablation_validation",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "completed_runs": completed,
        "missing_runs": missing,
        "invalid_runs": invalid,
        "expected_run_count": len(expected),
        "completed_run_count": len(completed),
        "aggregated": aggregated,
        "paired_comparisons": comparisons,
        "formal_success_checks": checks,
        "relative_target": float(relative_target),
        "test_policy": "Validation-only. Test is never constructed or scored by v2.6.6.",
        "test_loaded": False,
    }
