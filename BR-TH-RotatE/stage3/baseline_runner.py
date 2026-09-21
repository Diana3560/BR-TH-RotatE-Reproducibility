from __future__ import annotations

import gc
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from stage3.common import (
    aggregate,
    load_base_config,
    load_stage3_config,
    make_output_dir,
    read_json,
    require_pykeen_version,
    resolve_device,
    sha256_file,
    stage3_identity,
    write_csv,
    write_json,
    write_latest_pointer,
)
from stage3.models import build_compounde_model_class, build_rate_model_class, compounde_transform_tail
from throtate_repro.evaluation import metric_summary
from throtate_repro.exact_step_v266 import build_exact_step_training_loop_class
from throtate_repro.gamma3_canonical_v251 import build_explicit_split_report, runtime_loss_audit
from throtate_repro.model import build_pykeen_model_class
from throtate_repro.multidataset_data import audit_dataset
from throtate_repro.multidataset_experiment import _build_context, _model_spec
from throtate_repro.v266_runtime import AuditedBernoulliNegativeSampler


BASELINE_MODELS = ("RatE", "CompoundE", "TH-RotatE-Cap")
METRICS = ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10")


def _compounde_broadcast_smoke_check() -> dict[str, Any]:
    """Verify score_t-style relation/candidate broadcasting before expensive training."""
    batch_size, num_candidates, dim = 10, 37, 8
    tail = torch.randn(1, num_candidates, dim)
    scale = torch.ones(batch_size, 1, dim)
    translation = torch.zeros(batch_size, 1, dim)
    theta = torch.zeros(batch_size, 1, dim // 2)
    out = compounde_transform_tail(
        tail=tail,
        scale=scale,
        translation=translation,
        theta=theta,
    )
    expected_shape = (batch_size, num_candidates, dim)
    if tuple(out.shape) != expected_shape:
        raise RuntimeError(
            f"CompoundE broadcast smoke check failed: expected {expected_shape}, got {tuple(out.shape)}"
        )
    if not torch.isfinite(out).all():
        raise RuntimeError("CompoundE broadcast smoke check produced non-finite values")
    return {
        "status": "PASS",
        "scenario": "PyKEEN score_t-style batched queries x all candidate tails",
        "query_batch_size": batch_size,
        "candidate_count": num_candidates,
        "embedding_dim": dim,
        "output_shape": list(out.shape),
    }


def _count_params(model) -> dict[str, int]:
    """Count tensor elements and real scalar degrees of freedom.

    PyTorch ``numel()`` counts a complex value as one element. The manuscript parameter
    table counts real scalar degrees of freedom, so complex tensors count twice there.
    Both counts are emitted for auditability; capacity matching uses real scalars.
    """
    tensor_elements = 0
    trainable_tensor_elements = 0
    real_scalars = 0
    trainable_real_scalars = 0
    for p in model.parameters():
        n = int(p.numel())
        multiplier = 2 if torch.is_complex(p) else 1
        tensor_elements += n
        real_scalars += n * multiplier
        if p.requires_grad:
            trainable_tensor_elements += n
            trainable_real_scalars += n * multiplier
    return {
        "tensor_elements": tensor_elements,
        "trainable_tensor_elements": trainable_tensor_elements,
        "real_scalar_parameters": real_scalars,
        "trainable_real_scalar_parameters": trainable_real_scalars,
    }


def _capacity_delta(stage3_cfg: Mapping[str, Any], dataset_key: str) -> float:
    cap = stage3_cfg["baselines"]["capacity_match"]
    key = f"{dataset_key}_target_delta_radians"
    if key not in cap:
        raise KeyError(f"No D2 target Delta configured for {dataset_key}")
    return float(cap[key])


def capacity_match(
    *,
    project_root: Path,
    stage3_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    dataset_key: str,
    context: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Choose the smallest plain TH-RotatE dimension whose capacity is >= D2.

    Selection uses architecture/parameter counts only. No Validation/Test metric is read.
    """
    cap_cfg = stage3_cfg["baselines"]["capacity_match"]
    delta = _capacity_delta(stage3_cfg, dataset_key)
    target_spec = _model_spec(base_cfg, context, "D2", delta)
    target_model = target_spec["model"](
        triples_factory=context["reciprocal_training"],
        **target_spec["model_kwargs"],
        random_seed=42,
    )
    try:
        target_counts = _count_params(target_model)
    finally:
        del target_model
        gc.collect()

    target = int(target_counts["trainable_real_scalar_parameters"])
    min_dim = int(cap_cfg["scan_min_dim"])
    max_dim = int(cap_cfg["scan_max_dim"])
    base_dim = int(base_cfg["model"]["embedding_dim"])
    TH = build_pykeen_model_class()

    def count_dim(dim: int) -> dict[str, int]:
        model = TH(
            triples_factory=context["training"],
            embedding_dim=int(dim),
            transh_norm=int(base_cfg["model"]["transh_norm"]),
            transh_power_norm=bool(base_cfg["model"]["transh_power_norm"]),
            rotate_norm=int(base_cfg["model"]["rotate_norm"]),
            random_seed=42,
        )
        try:
            return _count_params(model)
        finally:
            del model
            gc.collect()

    # Parameter count is linear in embedding dimension; use two points to estimate the
    # crossing, then explicitly verify a small neighborhood. This avoids instantiating
    # >100 large models just to count parameters.
    d0 = max(min_dim, min(base_dim, max_dim - 1))
    d1 = d0 + 1
    c0 = count_dim(d0)
    c1 = count_dim(d1)
    slope = int(c1["trainable_real_scalar_parameters"] - c0["trainable_real_scalar_parameters"])
    if slope <= 0:
        raise RuntimeError(f"Unexpected non-positive TH-RotatE capacity slope: {slope}")
    intercept = int(c0["trainable_real_scalar_parameters"] - slope * d0)
    estimated = math.ceil((target - intercept) / slope)
    estimated = min(max(estimated, min_dim), max_dim)
    dims = sorted({d for d in range(max(min_dim, estimated - 2), min(max_dim, estimated + 2) + 1)})
    rows = []
    for dim in dims:
        counts = count_dim(dim)
        rows.append({"embedding_dim": dim, **counts})
    eligible = [r for r in rows if int(r["trainable_real_scalar_parameters"]) >= target]
    if not eligible:
        # Edge case: estimate was clipped. Evaluate the upper boundary before failing.
        counts = count_dim(max_dim)
        row = {"embedding_dim": max_dim, **counts}
        if all(r["embedding_dim"] != max_dim for r in rows):
            rows.append(row)
        eligible = [r for r in rows if int(r["trainable_real_scalar_parameters"]) >= target]
    if not eligible:
        raise RuntimeError(
            f"Capacity scan [{min_dim}, {max_dim}] never reaches D2 target={target} real scalars"
        )
    selected = min(eligible, key=lambda r: int(r["embedding_dim"]))
    result = {
        "status": "PASS",
        "selection_uses_metrics": False,
        "rule": str(cap_cfg["rule"]),
        "target_model": "D2",
        "target_delta_radians": delta,
        "target_counts": target_counts,
        "target_trainable_real_scalar_parameters": target,
        "selected_embedding_dim": int(selected["embedding_dim"]),
        "selected_counts": {k: int(v) for k, v in selected.items() if k != "embedding_dim"},
        "parameter_excess_real_scalars": int(selected["trainable_real_scalar_parameters"]) - target,
        "parameter_excess_fraction": (
            int(selected["trainable_real_scalar_parameters"]) / target - 1.0
        ),
        "linear_count_slope_real_scalars_per_dim": slope,
        "verified_candidates": sorted(rows, key=lambda r: int(r["embedding_dim"])),
    }
    write_json(output_dir / "CAPACITY_MATCH.json", result)
    write_csv(output_dir / "CAPACITY_MATCH.csv", result["verified_candidates"])
    return result


def _model_spec_stage3(
    *,
    model_name: str,
    stage3_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> dict[str, Any]:
    if model_name == "RatE":
        return {
            "model": build_rate_model_class(),
            "model_kwargs": {"embedding_dim": int(stage3_cfg["baselines"]["embedding_dim"]["RatE"])},
            "implementation": "stage3.models.RatE",
        }
    if model_name == "CompoundE":
        return {
            "model": build_compounde_model_class(),
            "model_kwargs": {"embedding_dim": int(stage3_cfg["baselines"]["embedding_dim"]["CompoundE"])},
            "implementation": "stage3.models.CompoundE",
        }
    if model_name == "TH-RotatE-Cap":
        return {
            "model": build_pykeen_model_class(),
            "model_kwargs": {
                "embedding_dim": int(capacity["selected_embedding_dim"]),
                "transh_norm": int(base_cfg["model"]["transh_norm"]),
                "transh_power_norm": bool(base_cfg["model"]["transh_power_norm"]),
                "rotate_norm": int(base_cfg["model"]["rotate_norm"]),
            },
            "implementation": "throtate_repro.model.THRotatE(capacity-matched)",
        }
    raise ValueError(f"Unknown Stage3 baseline model: {model_name}")


def _train_external(
    *,
    stage3_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    context: Mapping[str, Any],
    identity: Mapping[str, Any],
    model_name: str,
    capacity: Mapping[str, Any],
    seed: int,
    gamma: float,
    evaluation_split: str,
    device_override: str | None,
    frozen_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    from pykeen.pipeline import pipeline

    if evaluation_split not in {"validation", "test"}:
        raise ValueError(evaluation_split)
    if evaluation_split == "test" and context.get("test") is None:
        raise RuntimeError("Test factory is not constructed")

    spec = _model_spec_stage3(
        model_name=model_name,
        stage3_cfg=stage3_cfg,
        base_cfg=base_cfg,
        capacity=capacity,
    )
    training_factory = context["training"]
    evaluation_factory = context[evaluation_split]
    truth_factories = [context["training"], context["validation"]]
    if evaluation_split == "test":
        truth_factories.append(context["test"])

    train_cfg = base_cfg["training"]
    budget = context["budgets"]["ordinary"]
    expected_positive = int(train_cfg["max_steps"]) * int(train_cfg["batch_size"])
    expected_negative = expected_positive * int(train_cfg["num_negatives"])
    device = resolve_device(base_cfg, device_override)

    AuditedBernoulliNegativeSampler.reset_audit()
    result = None
    started = time.perf_counter()
    try:
        result = pipeline(
            training=training_factory,
            validation=evaluation_factory,
            testing=evaluation_factory,
            model=spec["model"],
            model_kwargs=spec["model_kwargs"],
            loss=str(train_cfg["loss"]),
            loss_kwargs={
                "margin": float(gamma),
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
            evaluator_kwargs={"filtered": bool(base_cfg["protocol"]["filtered_evaluation"])},
            evaluation_kwargs={"batch_size": int(train_cfg["evaluation_batch_size"])},
            use_testing_data=False,
            filter_validation_when_testing=False,
            random_seed=int(seed),
            device=device,
        )
        wall = time.perf_counter() - started
        steps = int(getattr(result.training_loop, "v266_optimizer_steps", -1))
        if steps != int(train_cfg["max_steps"]):
            raise RuntimeError(f"Exact-step audit failed: requested={train_cfg['max_steps']} actual={steps}")
        sampling = AuditedBernoulliNegativeSampler.audit_snapshot(
            num_negatives_per_positive=int(train_cfg["num_negatives"])
        )
        if int(sampling["actual_positive_instances"]) != expected_positive:
            raise RuntimeError(f"Positive exposure audit failed: {sampling}")
        if int(sampling["requested_negatives"]) != expected_negative:
            raise RuntimeError(f"Negative exposure audit failed: {sampling}")
        if sampling["all_negatives_accepted"] is not True:
            raise RuntimeError(f"Negative sampler unexpectedly filtered samples: {sampling}")
        loss_audit = runtime_loss_audit(result.model, float(gamma))
        if not loss_audit["pass"]:
            raise RuntimeError(f"NSSA loss audit failed: {loss_audit}")

        filtered = bool(base_cfg["protocol"]["filtered_evaluation"])
        report = build_explicit_split_report(
            model=result.model,
            evaluation_factory=evaluation_factory,
            truth_factories=truth_factories,
            batch_size=int(train_cfg["evaluation_batch_size"]),
            filtered=filtered,
            include_per_relation=evaluation_split == "test",
        )
        primary = dict(report["all_relations_both_sides"]["metrics"])
        bad = {m: primary.get(m) for m in METRICS if primary.get(m) is None or not math.isfinite(float(primary[m]))}
        if bad:
            raise RuntimeError(f"Missing/non-finite metrics: {bad}")
        expected_ranks = 2 * int(evaluation_factory.num_triples)
        if int(report["all_relations_both_sides"]["rank_count"]) != expected_ranks:
            raise RuntimeError("Both-side rank count audit failed")

        counts = _count_params(result.model)
        return {
            "status": "PASS",
            "phase": f"stage3_baseline_{evaluation_split}",
            "dataset_key": context["dataset_key"],
            "stage3_fingerprint": identity["fingerprint"],
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "model": model_name,
            "implementation": spec["implementation"],
            "training_protocol": "unified_project_protocol",
            "seed": int(seed),
            "gamma": float(gamma),
            "evaluation_split": evaluation_split,
            "split_sha256": context["audit"]["sha256"][evaluation_split],
            "metrics": primary,
            "per_relation_tail_only": report.get("per_relation_tail_only", {}),
            "parameter_counts": counts,
            "actual_optimizer_steps": steps,
            "training_budget": budget,
            "sampling_exposure_audit": sampling,
            "runtime_loss_audit": loss_audit,
            "train_seconds": float(result.train_seconds),
            "wall_seconds": float(wall),
            "candidate_entities": int(context["full"].num_entities),
            "evaluation_triples": int(evaluation_factory.num_triples),
            "both_side_ranks": expected_ranks,
            "filtered_evaluation": filtered,
            "test_used_for_parameter_selection": False,
        }
    finally:
        if result is not None:
            del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cached(path: Path, expected: Mapping[str, Any], call) -> dict[str, Any]:
    old = read_json(path)
    if old and old.get("status") == "PASS":
        mismatch = {k: {"expected": v, "actual": old.get(k)} for k, v in expected.items() if old.get(k) != v}
        if mismatch:
            raise RuntimeError(f"Cached Stage3 result identity mismatch: {path}: {mismatch}")
        return old
    try:
        row = call()
        row.update(expected)
        write_json(path, row)
        return row
    except Exception as exc:
        write_json(
            path,
            {
                "status": "FAIL",
                **dict(expected),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def prepare(project_root: str | Path, dataset_key: str) -> tuple[dict, dict, dict, Path, dict, dict]:
    root = Path(project_root).resolve()
    stage3_cfg = load_stage3_config(root)
    base_cfg = load_base_config(root, stage3_cfg)
    require_pykeen_version(stage3_cfg)
    audit = audit_dataset(root, base_cfg, dataset_key)
    identity = stage3_identity(root, stage3_cfg, base_cfg, dataset_key, audit=audit)
    output_dir = make_output_dir(root, stage3_cfg, dataset_key, identity["fingerprint"], kind="baselines")
    context = _build_context(root, base_cfg, dataset_key, output_dir, include_test=False)
    capacity = read_json(output_dir / "CAPACITY_MATCH.json")
    if not capacity or capacity.get("status") != "PASS":
        capacity = capacity_match(
            project_root=root,
            stage3_cfg=stage3_cfg,
            base_cfg=base_cfg,
            dataset_key=dataset_key,
            context=context,
            output_dir=output_dir,
        )
    return stage3_cfg, base_cfg, identity, output_dir, context, capacity


def preflight(project_root: str | Path, dataset_key: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, base_cfg, identity, output_dir, context, capacity = prepare(root, dataset_key)
    summary = {
        "status": "PASS",
        "phase": "stage3_baseline_preflight_validation_only",
        "dataset_key": dataset_key,
        "stage3_fingerprint": identity["fingerprint"],
        "runtime_environment": identity["runtime_environment"],
        "data_audit": context["audit"],
        "capacity_match": capacity,
        "compounde_broadcast_smoke": _compounde_broadcast_smoke_check(),
        "formal_models": list(stage3_cfg["baselines"]["models"]),
        "formal_seeds": [int(s) for s in stage3_cfg["baselines"]["formal_seeds"]],
        "gamma_candidates": [float(x) for x in stage3_cfg["baselines"]["gamma_candidates"]],
        "test_factory_constructed": False,
        "test_metrics_used": False,
    }
    write_json(output_dir / "PRECHECK_VALIDATION_ONLY.json", summary)
    write_latest_pointer(
        root,
        stage3_cfg,
        dataset_key,
        kind="baselines",
        identity=identity,
        output_dir=output_dir,
    )
    return summary


def validation_screen(project_root: str | Path, dataset_key: str, *, model_name: str | None = None, device: str | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, base_cfg, identity, output_dir, context, capacity = prepare(root, dataset_key)
    pre = read_json(output_dir / "PRECHECK_VALIDATION_ONLY.json")
    if not pre or pre.get("status") != "PASS":
        preflight(root, dataset_key)
    models = [model_name] if model_name else list(stage3_cfg["baselines"]["models"])
    for name in models:
        if name not in BASELINE_MODELS:
            raise ValueError(f"Unsupported Stage3 baseline: {name}")
    seed = int(stage3_cfg["baselines"]["validation_seed"])
    gammas = [float(x) for x in stage3_cfg["baselines"]["gamma_candidates"]]
    selections: dict[str, Any] = {}
    for name in models:
        runs = []
        for gamma in gammas:
            label = f"{gamma:g}".replace(".", "p")
            path = output_dir / "validation_screen" / name / f"gamma_{label}_seed{seed}.json"
            expected = {
                "dataset_key": dataset_key,
                "stage3_fingerprint": identity["fingerprint"],
                "model": name,
                "seed": seed,
                "gamma": gamma,
                "evaluation_split": "validation",
            }
            row = _cached(
                path,
                expected,
                lambda name=name, gamma=gamma: _train_external(
                    stage3_cfg=stage3_cfg,
                    base_cfg=base_cfg,
                    context=context,
                    identity=identity,
                    model_name=name,
                    capacity=capacity,
                    seed=seed,
                    gamma=gamma,
                    evaluation_split="validation",
                    device_override=device,
                ),
            )
            runs.append(row)
        selected = max(
            runs,
            key=lambda r: (
                float(r["metrics"]["mrr"]),
                float(r["metrics"]["hits_at_1"]),
                -float(r["gamma"]),
            ),
        )
        selections[name] = {
            "selected_gamma": float(selected["gamma"]),
            "selected_validation_metrics": dict(selected["metrics"]),
            "candidates": [{"gamma": float(r["gamma"]), **dict(r["metrics"])} for r in runs],
        }

    # Merge with any previously completed per-model screens.
    existing = read_json(output_dir / "VALIDATION_SELECTION_SUMMARY.json") or {}
    merged = dict(existing.get("baseline_selection") or {})
    merged.update(selections)
    summary = {
        "status": "COMPLETE" if all(m in merged for m in BASELINE_MODELS) else "PARTIAL",
        "phase": "stage3_validation_only_parameter_selection",
        "dataset_key": dataset_key,
        "stage3_fingerprint": identity["fingerprint"],
        "baseline_selection": merged,
        "selection_rule": "highest filtered both-side Validation MRR; tie -> Hits@1; exact tie -> smaller gamma",
        "test_factory_constructed": False,
        "test_metrics_used": False,
    }
    write_json(output_dir / "VALIDATION_SELECTION_SUMMARY.json", summary)
    return summary


def freeze_protocol(project_root: str | Path, dataset_key: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, _base_cfg, identity, output_dir, _context, capacity = prepare(root, dataset_key)
    selection_path = output_dir / "VALIDATION_SELECTION_SUMMARY.json"
    selection = read_json(selection_path)
    if not selection or selection.get("status") != "COMPLETE":
        raise RuntimeError("Complete Validation screening for all three Stage3 baselines before freezing")
    manifest = {
        "status": "FROZEN",
        "phase": "stage3_protocol_freeze",
        "dataset_key": dataset_key,
        "stage3_fingerprint": identity["fingerprint"],
        "validation_selection_sha256": sha256_file(selection_path),
        "selected_gamma": {
            model: float(selection["baseline_selection"][model]["selected_gamma"])
            for model in BASELINE_MODELS
        },
        "capacity_match": capacity,
        "formal_models": list(BASELINE_MODELS),
        "formal_seeds": [int(s) for s in stage3_cfg["baselines"]["formal_seeds"]],
        "test_used_for_parameter_selection": False,
    }
    write_json(output_dir / "PROTOCOL_FROZEN.json", manifest)
    return manifest


def fixed_test(
    project_root: str | Path,
    dataset_key: str,
    *,
    model_name: str | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, base_cfg, identity, output_dir, _context_val, capacity = prepare(root, dataset_key)
    frozen_path = output_dir / "PROTOCOL_FROZEN.json"
    frozen = read_json(frozen_path)
    if not frozen or frozen.get("status") != "FROZEN":
        raise RuntimeError("Freeze the Stage3 protocol before any Test run")
    frozen_sha = sha256_file(frozen_path)
    # Test factory is constructed only after the frozen manifest exists.
    context = _build_context(root, base_cfg, dataset_key, output_dir, include_test=True)
    models = [model_name] if model_name else list(BASELINE_MODELS)
    seeds = [int(seed)] if seed is not None else [int(s) for s in stage3_cfg["baselines"]["formal_seeds"]]
    rows = []
    for name in models:
        if name not in BASELINE_MODELS:
            raise ValueError(name)
        gamma = float(frozen["selected_gamma"][name])
        for current_seed in seeds:
            path = output_dir / "fixed_test_runs" / f"{name}_seed{current_seed}.json"
            expected = {
                "dataset_key": dataset_key,
                "stage3_fingerprint": identity["fingerprint"],
                "frozen_manifest_sha256": frozen_sha,
                "model": name,
                "seed": current_seed,
                "gamma": gamma,
                "evaluation_split": "test",
            }
            row = _cached(
                path,
                expected,
                lambda name=name, current_seed=current_seed, gamma=gamma: _train_external(
                    stage3_cfg=stage3_cfg,
                    base_cfg=base_cfg,
                    context=context,
                    identity=identity,
                    model_name=name,
                    capacity=capacity,
                    seed=current_seed,
                    gamma=gamma,
                    evaluation_split="test",
                    device_override=device,
                    frozen_manifest_sha256=frozen_sha,
                ),
            )
            rows.append(row)
    return {"status": "COMPLETE", "runs": rows}


def _stage2_runs(root: Path, dataset_key: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Find a Stage-2 directory containing D0/D1/A0/D2 five-seed Test results."""
    base = root / "results" / "multidataset_comparison" / dataset_key
    if not base.is_dir():
        return {}
    required = [(m, s) for m in ("D0", "D1", "A0", "D2") for s in (42, 43, 44, 45, 46)]
    candidates = []
    for directory in base.iterdir():
        if not directory.is_dir():
            continue
        run_dir = directory / "fixed_test_runs"
        if not run_dir.is_dir():
            continue
        if all((run_dir / f"{m}_seed{s}.json").is_file() for m, s in required):
            candidates.append(directory)
    if not candidates:
        return {}
    # Prefer a run that also has a summary; otherwise newest by mtime.
    candidates.sort(key=lambda d: ((d / "DATASET_COMPARISON_SUMMARY.json").is_file(), d.stat().st_mtime), reverse=True)
    chosen = candidates[0]
    result = {}
    for m, s in required:
        row = read_json(chosen / "fixed_test_runs" / f"{m}_seed{s}.json")
        if row and row.get("status") == "PASS":
            result[(m, s)] = row
    return result


def summarize(project_root: str | Path, dataset_key: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, _base_cfg, identity, output_dir, _context, capacity = prepare(root, dataset_key)
    frozen_path = output_dir / "PROTOCOL_FROZEN.json"
    frozen = read_json(frozen_path)
    if not frozen or frozen.get("status") != "FROZEN":
        raise RuntimeError("Protocol is not frozen")
    frozen_sha = sha256_file(frozen_path)
    seeds = [int(s) for s in stage3_cfg["baselines"]["formal_seeds"]]
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for model in BASELINE_MODELS:
        for seed in seeds:
            path = output_dir / "fixed_test_runs" / f"{model}_seed{seed}.json"
            row = read_json(path)
            if not row or row.get("status") != "PASS":
                raise RuntimeError(f"Missing successful Test run: {path}")
            if row.get("frozen_manifest_sha256") != frozen_sha:
                raise RuntimeError(f"Frozen-manifest mismatch: {path}")
            lookup[(model, seed)] = row

    raw_rows = []
    summary_rows = []
    for model in BASELINE_MODELS:
        for seed in seeds:
            r = lookup[(model, seed)]
            raw_rows.append({
                "dataset": dataset_key,
                "model": model,
                "seed": seed,
                "gamma": r["gamma"],
                **{m: r["metrics"][m] for m in METRICS},
                "trainable_real_scalar_parameters": r["parameter_counts"]["trainable_real_scalar_parameters"],
                "train_seconds": r["train_seconds"],
            })
        row: dict[str, Any] = {
            "dataset": dataset_key,
            "model": model,
            "n_seeds": len(seeds),
            "gamma": frozen["selected_gamma"][model],
            "trainable_real_scalar_parameters": int(lookup[(model, seeds[0])]["parameter_counts"]["trainable_real_scalar_parameters"]),
        }
        for metric in METRICS:
            a = aggregate(float(lookup[(model, s)]["metrics"][metric]) for s in seeds)
            row[f"{metric}_mean"] = a["mean"]
            row[f"{metric}_std_sample"] = a["std_sample"]
        summary_rows.append(row)

    write_csv(output_dir / "STAGE3_BASELINE_TEST_RESULTS.csv", raw_rows)
    write_csv(output_dir / "STAGE3_BASELINE_SUMMARY.csv", summary_rows)

    # Paired deltas against the already-completed D2 runs, when available.
    stage2 = _stage2_runs(root, dataset_key)
    delta_rows = []
    if all(("D2", s) in stage2 for s in seeds):
        for model in BASELINE_MODELS:
            for metric in METRICS:
                diffs = [float(stage2[("D2", s)]["metrics"][metric]) - float(lookup[(model, s)]["metrics"][metric]) for s in seeds]
                # For MR, negative D2-new is favorable; keep raw delta and explicitly label direction.
                delta_rows.append({
                    "dataset": dataset_key,
                    "comparison": f"D2_minus_{model}",
                    "metric": metric,
                    "favorable_direction_for_D2": "positive" if metric != "mr" else "negative",
                    "mean_delta": aggregate(diffs)["mean"],
                    "std_sample_delta": aggregate(diffs)["std_sample"],
                    "wins_D2": sum((d > 0) if metric != "mr" else (d < 0) for d in diffs),
                    "ties": sum(d == 0 for d in diffs),
                    "losses_D2": sum((d < 0) if metric != "mr" else (d > 0) for d in diffs),
                })
        write_csv(output_dir / "STAGE3_DELTA_ANALYSIS.csv", delta_rows)

    result = {
        "status": "COMPLETE",
        "phase": "stage3_baseline_summary",
        "dataset_key": dataset_key,
        "stage3_fingerprint": identity["fingerprint"],
        "frozen_manifest_sha256": frozen_sha,
        "capacity_match": capacity,
        "models": list(BASELINE_MODELS),
        "seeds": seeds,
        "summary_rows": summary_rows,
        "stage2_d2_available_for_paired_delta": bool(stage2),
        "delta_rows": delta_rows,
        "reporting_note": (
            "RatE and CompoundE are scoring-architecture baselines trained under the unified project protocol; "
            "do not describe them as exact reproductions of the papers' original training/sampling recipes."
        ),
    }
    write_json(output_dir / "STAGE3_BASELINE_SUMMARY.json", result)
    return result


def run_all(project_root: str | Path, dataset_key: str, *, device: str | None = None) -> dict[str, Any]:
    preflight(project_root, dataset_key)
    validation_screen(project_root, dataset_key, device=device)
    freeze_protocol(project_root, dataset_key)
    fixed_test(project_root, dataset_key, device=device)
    return summarize(project_root, dataset_key)
