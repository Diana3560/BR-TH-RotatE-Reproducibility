from __future__ import annotations

"""Validation-only runtime for the staged v2.6.6 equal-budget ablation."""

import copy
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from pykeen.sampling import BernoulliNegativeSampler

from .bounded_adaptive_v26 import build_bounded_adaptive_model_class
from .evaluation import metric_summary
from .exact_step_v266 import build_exact_step_training_loop_class
from .gamma3_canonical_v251 import build_explicit_split_report, runtime_loss_audit
from .model import build_pykeen_model_class
from .role_control_v18 import REASONING14_RELATIONS
from .v263_development import load_context as load_v263_context
from .v266_protocol import (
    SEEDS,
    VARIANTS,
    VARIANT_LABELS,
    build_progress_summary,
    exact_step_budget,
)


EXPECTED_TOKEN = "OWNKGC_V266_EQUAL_BUDGET_ABLATION_VALIDATION_ONLY"


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["protocol"]["token"] != EXPECTED_TOKEN:
        raise ValueError("v2.6.6 protocol token mismatch")
    if str(cfg["protocol"]["pykeen_version"]) != "1.11.1":
        raise ValueError("This experiment is frozen for PyKEEN 1.11.1")
    if tuple(cfg["development"]["variants"]) != VARIANTS:
        raise ValueError(f"variants must remain {VARIANTS}")
    if tuple(int(seed) for seed in cfg["development"]["seeds"]) != SEEDS:
        raise ValueError(f"seeds must remain {SEEDS}")
    training = cfg["training"]
    frozen = {
        "max_steps": 3000,
        "batch_size": 1024,
        "num_negatives": 64,
    }
    for key, expected in frozen.items():
        if int(training[key]) != expected:
            raise ValueError(f"{key} must remain {expected}")
    if float(training["gamma"]) != 3.0:
        raise ValueError("gamma must remain 3")
    if float(training["learning_rate"]) != 0.001:
        raise ValueError("Adam learning rate must remain constant at 0.001")
    if training.get("drop_last_for_equal_exposure") is not True:
        raise ValueError("drop_last_for_equal_exposure must remain true")
    if training.get("negative_sampler_filtered") is not False:
        raise ValueError("ordinary unfiltered Bernoulli sampling must be retained")
    if abs(float(cfg["model"]["frozen_delta_angle_radians"]) - 0.10) > 1.0e-12:
        raise ValueError("bounded-adapter Delta must remain 0.10")
    required_true = (
        "equal_optimizer_steps",
        "equal_full_batch_exposure",
        "constant_learning_rate",
        "ordinary_bernoulli_sampling",
        "validation_only",
        "no_test_parameter_selection",
        "staged_manual_execution",
    )
    if not all(cfg["protocol"].get(key) is True for key in required_true):
        raise ValueError("v2.6.6 protocol guard mismatch")
    if cfg["protocol"].get("test_loaded") is not False:
        raise ValueError("Test must remain sealed")


def _v263_compatible_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the audited Training/Validation loader without ever constructing Test."""
    legacy = copy.deepcopy(dict(cfg))
    legacy["training"].update(max_steps=2000, negative_sampler_filtered=False)
    legacy["development"] = {
        "formal_seeds": [42, 43, 44, 45, 46],
        "variants": ["V0", "V1", "V2", "V3"],
    }
    legacy["protocol"].update(
        token="OWNKGC_V263_R14_VALIDATION_ONLY_RESIDUAL_RECIPROCAL",
        typed_negative_sampling_removed=True,
        test_loaded=False,
    )
    return legacy


def load_context(project_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(cfg)
    context = load_v263_context(project_root, _v263_compatible_config(cfg))
    train_cfg = cfg["training"]
    ordinary = exact_step_budget(
        num_real_triples=int(context["training"].num_triples),
        batch_size=int(train_cfg["batch_size"]),
        max_steps=int(train_cfg["max_steps"]),
        reciprocal=False,
        drop_last=True,
    )
    reciprocal = exact_step_budget(
        num_real_triples=int(context["training"].num_triples),
        batch_size=int(train_cfg["batch_size"]),
        max_steps=int(train_cfg["max_steps"]),
        reciprocal=True,
        drop_last=True,
    )
    if ordinary["exact_positive_exposure"] != reciprocal["exact_positive_exposure"]:
        raise ValueError("ordinary/reciprocal positive-exposure audit failed")
    context["v266_budgets"] = {"ordinary": ordinary, "reciprocal": reciprocal}
    return context


class AuditedBernoulliNegativeSampler(BernoulliNegativeSampler):
    """Count the exact number of positives and negatives presented to the loss."""

    requested_negatives = 0
    accepted_negatives = 0
    sampled_batches = 0

    @classmethod
    def reset_audit(cls) -> None:
        cls.requested_negatives = 0
        cls.accepted_negatives = 0
        cls.sampled_batches = 0

    @classmethod
    def audit_snapshot(cls, *, num_negatives_per_positive: int) -> dict[str, Any]:
        requested = int(cls.requested_negatives)
        k = int(num_negatives_per_positive)
        return {
            "requested_negatives": requested,
            "accepted_negatives": int(cls.accepted_negatives),
            "sampled_batches": int(cls.sampled_batches),
            "num_negatives_per_positive": k,
            "actual_positive_instances": requested // k if k else None,
            "all_negatives_accepted": requested == int(cls.accepted_negatives),
        }

    def sample(self, positive_batch):  # noqa: D102
        negative_batch, mask = super().sample(positive_batch=positive_batch)
        requested = int(negative_batch.shape[0] * negative_batch.shape[1])
        type(self).requested_negatives += requested
        type(self).sampled_batches += 1
        type(self).accepted_negatives += requested if mask is None else int(mask.sum().item())
        return negative_batch, mask


def _common_model_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = cfg["model"]
    return {
        "embedding_dim": int(model_cfg["embedding_dim"]),
        "transh_norm": int(model_cfg["transh_norm"]),
        "transh_power_norm": bool(model_cfg["transh_power_norm"]),
        "rotate_norm": int(model_cfg["rotate_norm"]),
    }


def variant_spec(cfg: Mapping[str, Any], context: Mapping[str, Any], variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    common = _common_model_kwargs(cfg)
    if variant == "D0":
        return {
            "model": build_pykeen_model_class(),
            "model_kwargs": common,
            "training": context["training"],
            "pipeline_validation": context["validation"],
            "reciprocal": False,
            "adapter": False,
            "budget": context["v266_budgets"]["ordinary"],
        }
    if variant == "D1":
        return {
            "model": build_pykeen_model_class(),
            "model_kwargs": common,
            "training": context["reciprocal_training"],
            "pipeline_validation": context["reciprocal_validation"],
            "reciprocal": True,
            "adapter": False,
            "budget": context["v266_budgets"]["reciprocal"],
        }
    bundle = context["reciprocal_bundle"]
    return {
        "model": build_bounded_adaptive_model_class(),
        "model_kwargs": {
            **common,
            "relation_feature_tensor": bundle.features,
            "relation_reliability_tensor": bundle.reliability,
            "max_angle_deviation": float(cfg["model"]["frozen_delta_angle_radians"]),
        },
        "training": context["reciprocal_training"],
        "pipeline_validation": context["reciprocal_validation"],
        "reciprocal": True,
        "adapter": True,
        "budget": context["v266_budgets"]["reciprocal"],
    }


def _directional_metrics(*, model, mapped_triples, truth_triples, batch_size: int, filtered: bool):
    from pykeen.evaluation import RankBasedEvaluator

    output = {}
    for name, targets, target_name in (
        ("head", ("head",), "head"),
        ("tail", ("tail",), "tail"),
        ("both", ("head", "tail"), "both"),
    ):
        result = RankBasedEvaluator(filtered=filtered).evaluate(
            model=model,
            mapped_triples=mapped_triples,
            additional_filter_triples=truth_triples if filtered else None,
            batch_size=int(batch_size),
            targets=targets,
        )
        output[name] = metric_summary(result, target=target_name)
    return output


def _relation_rows(model, relation_id_to_label: Mapping[int, str]) -> list[dict[str, Any]]:
    if not hasattr(model, "get_all_relation_fusion_state"):
        return []
    state = model.get_all_relation_fusion_state()
    weights = state["weights"]
    shift = state["angle_shift"]
    reliability = state["reliability"]
    rows = []
    for rid, label in sorted(relation_id_to_label.items()):
        if label not in REASONING14_RELATIONS:
            continue
        for direction, internal_id in (("forward", 2 * rid), ("inverse", 2 * rid + 1)):
            rows.append(
                {
                    "relation": label,
                    "real_relation_id": int(rid),
                    "internal_relation_id": int(internal_id),
                    "direction": direction,
                    "alpha": float(weights[internal_id, 0]),
                    "beta": float(weights[internal_id, 1]),
                    "angle_shift_radians": float(shift[internal_id]),
                    "reliability_q": float(reliability[internal_id, 0]),
                }
            )
    return rows


def run_variant(
    *,
    project_root: Path,
    cfg: Mapping[str, Any],
    context: Mapping[str, Any],
    variant: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Train exactly one independently initialized Validation-only run."""
    from pykeen.pipeline import pipeline

    validate_config(cfg)
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    spec = variant_spec(cfg, context, variant)
    train_cfg = cfg["training"]
    budget = spec["budget"]
    device_cfg = str(train_cfg.get("device", "auto"))
    device = None if device_cfg == "auto" else device_cfg
    expected_positive = int(train_cfg["max_steps"]) * int(train_cfg["batch_size"])
    expected_negative = expected_positive * int(train_cfg["num_negatives"])
    row: dict[str, Any] = {
        "status": "RUNNING",
        "phase": "v2.6.6_equal_budget_ablation_validation",
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "seed": seed,
        "requested_optimizer_steps": int(train_cfg["max_steps"]),
        "training_budget": budget,
        "reciprocal_training": bool(spec["reciprocal"]),
        "bounded_relation_adapter": bool(spec["adapter"]),
        "gamma": float(train_cfg["gamma"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "learning_rate_schedule": "constant",
        "negative_sampling": "unfiltered Bernoulli",
        "drop_last_for_equal_exposure": True,
        "expected_positive_instances": expected_positive,
        "expected_negative_instances": expected_negative,
        "candidate_entity_space": int(context["full"].num_entities),
        "validation_only": True,
        "test_loaded": False,
        "test_used_for_selection": False,
    }
    result = None
    AuditedBernoulliNegativeSampler.reset_audit()
    try:
        start = time.perf_counter()
        result = pipeline(
            training=spec["training"],
            validation=spec["pipeline_validation"],
            testing=spec["pipeline_validation"],
            model=spec["model"],
            model_kwargs=spec["model_kwargs"],
            loss=str(train_cfg["loss"]),
            loss_kwargs={
                "margin": float(train_cfg["gamma"]),
                "adversarial_temperature": float(train_cfg["adversarial_temperature"]),
            },
            optimizer=str(train_cfg["optimizer"]),
            optimizer_kwargs={"lr": float(train_cfg["learning_rate"])},
            training_loop=build_exact_step_training_loop_class(),
            training_loop_kwargs={
                "automatic_memory_optimization": False,
                "exact_max_steps": int(budget["exact_optimizer_steps"]),
                "steps_per_full_epoch": int(budget["steps_per_full_epoch"]),
                "final_epoch_batches": int(budget["final_epoch_batches"]),
                "exact_num_epochs": int(budget["num_epochs"]),
            },
            training_kwargs={
                "num_epochs": int(budget["num_epochs"]),
                "batch_size": int(train_cfg["batch_size"]),
                "sub_batch_size": int(train_cfg["batch_size"]),
                "drop_last": True,
                "use_tqdm_batch": True,
            },
            negative_sampler=AuditedBernoulliNegativeSampler,
            negative_sampler_kwargs={
                "num_negs_per_pos": int(train_cfg["num_negatives"]),
                "filtered": False,
            },
            evaluator="RankBasedEvaluator",
            evaluator_kwargs={"filtered": bool(train_cfg["filtered_evaluation"])},
            evaluation_kwargs={"batch_size": int(train_cfg["validation_batch_size"])},
            use_testing_data=False,
            filter_validation_when_testing=False,
            random_seed=seed,
            device=device,
        )
        wall_seconds = time.perf_counter() - start
        actual_steps = int(getattr(result.training_loop, "v266_optimizer_steps", -1))
        if actual_steps != int(train_cfg["max_steps"]):
            raise RuntimeError(f"exact-step audit failed: requested={train_cfg['max_steps']}, actual={actual_steps}")
        sampling_audit = AuditedBernoulliNegativeSampler.audit_snapshot(
            num_negatives_per_positive=int(train_cfg["num_negatives"])
        )
        if int(sampling_audit["actual_positive_instances"]) != expected_positive:
            raise RuntimeError(f"positive-exposure audit failed: {sampling_audit}")
        if int(sampling_audit["requested_negatives"]) != expected_negative:
            raise RuntimeError(f"negative-exposure audit failed: {sampling_audit}")
        if sampling_audit["all_negatives_accepted"] is not True:
            raise RuntimeError("unfiltered Bernoulli unexpectedly rejected negatives")
        loss_audit = runtime_loss_audit(result.model, 3.0)
        if not loss_audit["pass"]:
            raise RuntimeError(f"NSSA gamma audit failed: {loss_audit}")

        truth_factories = [context["training"], context["validation"]]
        report = build_explicit_split_report(
            model=result.model,
            evaluation_factory=context["validation"],
            truth_factories=truth_factories,
            batch_size=int(train_cfg["validation_batch_size"]),
            filtered=bool(train_cfg["filtered_evaluation"]),
            include_per_relation=True,
        )
        primary = dict(report["all_relations_both_sides"]["metrics"])
        directional = _directional_metrics(
            model=result.model,
            mapped_triples=context["validation"].mapped_triples,
            truth_triples=[context["training"].mapped_triples, context["validation"].mapped_triples],
            batch_size=int(train_cfg["validation_batch_size"]),
            filtered=bool(train_cfg["filtered_evaluation"]),
        )
        expected_ranks = int(cfg["data"]["expected"]["validation_both_side_ranks"])
        if int(report["all_relations_both_sides"]["rank_count"]) != expected_ranks:
            raise RuntimeError("Validation rank-count audit failed")

        if hasattr(result.model, "get_global_fusion_weights"):
            global_alpha, global_beta = result.model.get_global_fusion_weights()
        else:
            global_alpha, global_beta = result.model.get_fusion_weights()
        relation_rows = _relation_rows(result.model, context["relation_id_to_label"])
        if relation_rows:
            max_shift = max(abs(float(item["angle_shift_radians"])) for item in relation_rows)
            if max_shift > float(cfg["model"]["frozen_delta_angle_radians"]) + 1.0e-6:
                raise RuntimeError("bounded relation shift exceeded Delta=0.10")
        row.update(
            status="PASS",
            actual_optimizer_steps=actual_steps,
            validation_metrics=primary,
            directional_validation_metrics=directional,
            runtime_loss_audit=loss_audit,
            sampling_exposure_audit=sampling_audit,
            global_fusion_alpha=float(global_alpha),
            global_fusion_beta=float(global_beta),
            relation_fusion_state=relation_rows,
            num_model_parameters=sum(parameter.numel() for parameter in result.model.parameters()),
            num_trainable_model_parameters=sum(
                parameter.numel() for parameter in result.model.parameters() if parameter.requires_grad
            ),
            train_seconds=float(result.train_seconds),
            wall_seconds=float(wall_seconds),
            training_factory_num_triples=int(spec["training"].num_triples),
            training_internal_num_relations=int(spec["training"].num_relations),
        )
        write_json(run_dir / "validation_report.json", report)
    except Exception as exc:  # noqa: BLE001
        import traceback

        row.update(
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            sampling_exposure_audit=AuditedBernoulliNegativeSampler.audit_snapshot(
                num_negatives_per_positive=int(train_cfg["num_negatives"])
            ),
        )
    finally:
        write_json(run_dir / "run_result.json", row)
        if result is not None:
            del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def result_path(output_dir: Path, *, variant: str, seed: int) -> Path:
    return output_dir / "runs" / f"seed_{int(seed)}" / variant / "run_result.json"


def collect_passed_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            row = read_json(result_path(output_dir, variant=variant, seed=seed))
            if row is not None:
                rows.append(row)
    return rows


def refresh_progress_summary(output_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    summary = build_progress_summary(
        collect_passed_rows(output_dir),
        max_steps=int(cfg["training"]["max_steps"]),
        relative_target=float(cfg["success"]["relative_target_vs_D0"]),
    )
    write_json(output_dir / "v266_equal_budget_summary.json", summary)
    return summary
