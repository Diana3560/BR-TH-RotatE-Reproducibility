from __future__ import annotations

"""Stage-3 candidate-space sensitivity experiment.

This module is intentionally additive: it imports the frozen Stage-2 model/training
implementation but never edits it. Each D0/D1/A0/D2 model is trained exactly once per
seed, then the *same in-memory trained model* is evaluated in three candidate spaces:

1. ``full``              -- every entity participating in the full graph;
2. ``modeled_entities``  -- only entities occurring in Train/Validation/Test of the
                            modeled reasoning layer (R14 on CRH-L4MKG);
3. ``type_constrained``  -- relation-side entity types inferred from TRAIN only.

Type constraints never use Validation/Test to define an admissible type. Validation and
Test are read only to audit that their gold entities are not excluded. Filtered ranking
uses Train+Validation+Test positives, as in the manuscript's standard Test protocol.
"""

import gc
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from stage3.candidate_eval import evaluate_candidate_space
from stage3.candidate_spaces import (
    CandidateArtifacts,
    audit_type_schema_gold_coverage,
    build_candidate_artifacts,
    filtered_candidate_counts,
    nominal_candidate_counts,
    summarize_counts,
    type_schema_rows,
)
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
from throtate_repro.exact_step_v266 import build_exact_step_training_loop_class
from throtate_repro.gamma3_canonical_v251 import build_explicit_split_report, runtime_loss_audit
from throtate_repro.multidataset_data import audit_dataset, dataset_paths, read_entity_types
from throtate_repro.multidataset_experiment import _build_context, _model_spec
from throtate_repro.v266_runtime import AuditedBernoulliNegativeSampler


CANDIDATE_MODELS = ("D0", "D1", "A0", "D2")
CANDIDATE_SPACES = ("full", "modeled_entities", "type_constrained")
METRICS = ("mrr", "mr", "amri", "hits_at_1", "hits_at_3", "hits_at_10")


def _candidate_cfg(stage3_cfg: Mapping[str, Any], dataset_key: str) -> Mapping[str, Any]:
    cfg = stage3_cfg["candidate_sensitivity"]
    if dataset_key not in cfg:
        raise KeyError(f"Candidate-sensitivity settings are missing for dataset={dataset_key}")
    return cfg[dataset_key]


def _fixed_delta(stage3_cfg: Mapping[str, Any], dataset_key: str, model_name: str) -> float | None:
    ds = _candidate_cfg(stage3_cfg, dataset_key)
    if model_name == "A0":
        return float(ds["a0_delta_radians"])
    if model_name == "D2":
        return float(ds["d2_delta_radians"])
    return None


def _train_model_once(
    *,
    stage3_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    context: Mapping[str, Any],
    model_name: str,
    seed: int,
    device_override: str | None,
):
    """Train one frozen Stage-2 variant without touching Test metrics."""
    from pykeen.pipeline import pipeline

    gamma = float(stage3_cfg["candidate_sensitivity"]["fixed_gamma"])
    delta = _fixed_delta(stage3_cfg, context["dataset_key"], model_name)
    spec = _model_spec(base_cfg, context, model_name, delta)
    reciprocal = bool(spec["reciprocal"])
    training_factory = context["reciprocal_training"] if reciprocal else context["training"]
    validation_factory = context["reciprocal_validation"] if reciprocal else context["validation"]
    budget = context["budgets"]["reciprocal" if reciprocal else "ordinary"]
    train_cfg = base_cfg["training"]
    expected_positive = int(train_cfg["max_steps"]) * int(train_cfg["batch_size"])
    expected_negative = expected_positive * int(train_cfg["num_negatives"])
    device = resolve_device(base_cfg, device_override)

    AuditedBernoulliNegativeSampler.reset_audit()
    started = time.perf_counter()
    result = pipeline(
        training=training_factory,
        validation=validation_factory,
        testing=validation_factory,
        model=spec["model"],
        model_kwargs=spec["model_kwargs"],
        loss=str(train_cfg["loss"]),
        loss_kwargs={
            "margin": gamma,
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
    loss_audit = runtime_loss_audit(result.model, gamma)
    if not loss_audit["pass"]:
        raise RuntimeError(f"NSSA loss audit failed: {loss_audit}")

    training_audit = {
        "status": "PASS",
        "model": model_name,
        "seed": int(seed),
        "gamma": gamma,
        "delta_angle_radians": delta,
        "reciprocal_training": reciprocal,
        "implementation": spec["implementation"],
        "actual_optimizer_steps": steps,
        "training_budget": budget,
        "sampling_exposure_audit": sampling,
        "runtime_loss_audit": loss_audit,
        "train_seconds": float(result.train_seconds),
        "wall_seconds": float(wall),
        "validation_used_during_training_only": True,
        "test_metrics_used_for_training_or_selection": False,
    }
    return result, training_audit


def _schema_and_count_audits(
    *,
    project_root: Path,
    stage3_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    dataset_key: str,
    artifacts: CandidateArtifacts,
) -> dict[str, Any]:
    context_cfg = _candidate_cfg(stage3_cfg, dataset_key)
    expected_full = int(context_cfg["expected_full_candidates"])
    expected_modeled = context_cfg.get("expected_modeled_entity_candidates")
    if len(artifacts.full_entities) != expected_full:
        raise RuntimeError(
            f"Full candidate count mismatch: expected={expected_full}, actual={len(artifacts.full_entities)}"
        )
    if expected_modeled is not None and len(artifacts.modeled_entities) != int(expected_modeled):
        raise RuntimeError(
            "Modeled-entity candidate count mismatch: "
            f"expected={expected_modeled}, actual={len(artifacts.modeled_entities)}"
        )

    paths = dataset_paths(project_root, base_cfg, dataset_key)
    entity_types = read_entity_types(paths["entity_types"])
    coverage = audit_type_schema_gold_coverage(artifacts, entity_types)
    if coverage["status"] != "PASS":
        raise RuntimeError(f"TRAIN-only type schema excludes Validation/Test gold entities: {coverage}")

    count_audits: dict[str, Any] = {}
    for space in CANDIDATE_SPACES:
        nominal = nominal_candidate_counts(artifacts, space=space)
        filtered = filtered_candidate_counts(artifacts, space=space)
        count_audits[space] = {
            "nominal": summarize_counts(nominal),
            "filtered": summarize_counts(filtered),
        }
    return {
        "full_candidate_entities": len(artifacts.full_entities),
        "modeled_entity_candidates": len(artifacts.modeled_entities),
        "type_schema_gold_coverage": coverage,
        "candidate_count_audits": count_audits,
        "type_schema_rows": type_schema_rows(artifacts),
    }


def prepare(project_root: str | Path, dataset_key: str):
    root = Path(project_root).resolve()
    stage3_cfg = load_stage3_config(root)
    base_cfg = load_base_config(root, stage3_cfg)
    require_pykeen_version(stage3_cfg)
    audit = audit_dataset(root, base_cfg, dataset_key)
    identity = stage3_identity(root, stage3_cfg, base_cfg, dataset_key, audit=audit)
    output_dir = make_output_dir(root, stage3_cfg, dataset_key, identity["fingerprint"], kind="candidate")
    artifacts = build_candidate_artifacts(root, base_cfg, dataset_key)
    schema_audit = _schema_and_count_audits(
        project_root=root,
        stage3_cfg=stage3_cfg,
        base_cfg=base_cfg,
        dataset_key=dataset_key,
        artifacts=artifacts,
    )
    # Deliberately keep Test PyKEEN factory unconstructed until the candidate protocol
    # is frozen. Raw Test triples were already part of the immutable data audit above.
    context_validation = _build_context(root, base_cfg, dataset_key, output_dir, include_test=False)
    return stage3_cfg, base_cfg, identity, output_dir, context_validation, artifacts, schema_audit


def preflight(project_root: str | Path, dataset_key: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, _base_cfg, identity, output_dir, context, _artifacts, schema_audit = prepare(root, dataset_key)
    write_csv(output_dir / "FROZEN_TYPE_SCHEMA_TRAIN_ONLY.csv", schema_audit["type_schema_rows"])
    schema_payload = {
        "status": "FROZEN_FROM_TRAIN_ONLY",
        "source": "training_split_only",
        "rows": schema_audit["type_schema_rows"],
        "gold_coverage_audit": schema_audit["type_schema_gold_coverage"],
    }
    write_json(output_dir / "FROZEN_TYPE_SCHEMA_TRAIN_ONLY.json", schema_payload)
    summary = {
        "status": "PASS",
        "phase": "stage3_candidate_preflight",
        "dataset_key": dataset_key,
        "stage3_fingerprint": identity["fingerprint"],
        "runtime_environment": identity["runtime_environment"],
        "data_audit": context["audit"],
        "candidate_audit": schema_audit,
        "models": list(CANDIDATE_MODELS),
        "formal_seeds": [int(x) for x in stage3_cfg["candidate_sensitivity"]["formal_seeds"]],
        "candidate_spaces": list(CANDIDATE_SPACES),
        "same_trained_model_reused_across_spaces": True,
        "test_factory_constructed": False,
        "test_metrics_used_for_parameter_selection": False,
    }
    write_json(output_dir / "CANDIDATE_PRECHECK.json", summary)
    write_latest_pointer(
        root,
        stage3_cfg,
        dataset_key,
        kind="candidate",
        identity=identity,
        output_dir=output_dir,
    )
    return summary


def freeze_protocol(project_root: str | Path, dataset_key: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, _base_cfg, identity, output_dir, _context, _artifacts, _schema_audit = prepare(root, dataset_key)
    pre = read_json(output_dir / "CANDIDATE_PRECHECK.json")
    if not pre or pre.get("status") != "PASS":
        preflight(root, dataset_key)
    schema_path = output_dir / "FROZEN_TYPE_SCHEMA_TRAIN_ONLY.json"
    if not schema_path.is_file():
        raise RuntimeError("Missing TRAIN-only type-schema freeze artifact")
    fixed = {
        model: {
            "gamma": float(stage3_cfg["candidate_sensitivity"]["fixed_gamma"]),
            "delta_angle_radians": _fixed_delta(stage3_cfg, dataset_key, model),
        }
        for model in CANDIDATE_MODELS
    }
    manifest = {
        "status": "FROZEN",
        "phase": "stage3_candidate_protocol_freeze",
        "dataset_key": dataset_key,
        "stage3_fingerprint": identity["fingerprint"],
        "models": list(CANDIDATE_MODELS),
        "formal_seeds": [int(x) for x in stage3_cfg["candidate_sensitivity"]["formal_seeds"]],
        "spaces": list(CANDIDATE_SPACES),
        "fixed_model_parameters": fixed,
        "type_schema_source": "training_split_only",
        "type_schema_sha256": sha256_file(schema_path),
        "filtered_truth": "train_validation_test",
        "same_trained_model_reused_across_spaces": True,
        "test_used_for_parameter_selection": False,
    }
    write_json(output_dir / "CANDIDATE_PROTOCOL_FROZEN.json", manifest)
    return manifest


def _cached_run(path: Path, expected: Mapping[str, Any], call) -> dict[str, Any]:
    old = read_json(path)
    if old and old.get("status") == "PASS":
        mismatch = {k: {"expected": v, "actual": old.get(k)} for k, v in expected.items() if old.get(k) != v}
        if mismatch:
            raise RuntimeError(f"Cached candidate result identity mismatch: {path}: {mismatch}")
        return old
    try:
        result = call()
        result.update(expected)
        write_json(path, result)
        return result
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


def _standard_full_space_audit(*, model, context: Mapping[str, Any], reciprocal: bool, batch_size: int) -> dict[str, Any]:
    """Cross-check the custom full-space evaluator against the original PyKEEN path."""
    evaluation_factory = context["reciprocal_test"] if reciprocal else context["test"]
    report = build_explicit_split_report(
        model=model,
        evaluation_factory=evaluation_factory,
        truth_factories=[context["training"], context["validation"], context["test"]],
        batch_size=int(batch_size),
        filtered=True,
        include_per_relation=False,
    )
    return dict(report["all_relations_both_sides"]["metrics"])


def _run_model_seed(
    *,
    root: Path,
    stage3_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    identity: Mapping[str, Any],
    output_dir: Path,
    artifacts: CandidateArtifacts,
    model_name: str,
    seed: int,
    device: str | None,
    frozen_sha: str,
) -> dict[str, Any]:
    # Test factory is now permitted because the protocol is already frozen.
    context = _build_context(root, base_cfg, identity["dataset_key"], output_dir, include_test=True)
    result = None
    training_instance_id = f"{model_name}-seed{seed}-{uuid.uuid4().hex}"
    try:
        result, training_audit = _train_model_once(
            stage3_cfg=stage3_cfg,
            base_cfg=base_cfg,
            context=context,
            model_name=model_name,
            seed=seed,
            device_override=device,
        )
        model = result.model
        model_object_id = id(model)
        spaces: dict[str, Any] = {}
        for space in CANDIDATE_SPACES:
            current = evaluate_candidate_space(
                model=model,
                context=context,
                artifacts=artifacts,
                space=space,
                batch_size=int(base_cfg["training"]["evaluation_batch_size"]),
            )
            if id(model) != model_object_id:
                raise RuntimeError("Model object changed between candidate-space evaluations")
            current["training_instance_id"] = training_instance_id
            spaces[space] = current

        standard_full_metrics = _standard_full_space_audit(
            model=model,
            context=context,
            reciprocal=bool(training_audit["reciprocal_training"]),
            batch_size=int(base_cfg["training"]["evaluation_batch_size"]),
        )
        comparison = {}
        for metric in ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10"):
            custom_value = float(spaces["full"]["metrics"][metric])
            standard_value = float(standard_full_metrics[metric])
            abs_diff = abs(custom_value - standard_value)
            tolerance = 1.0e-6 if metric != "mr" else 1.0e-5
            comparison[metric] = {
                "custom": custom_value,
                "standard_pykeen": standard_value,
                "absolute_difference": abs_diff,
                "tolerance": tolerance,
                "pass": abs_diff <= tolerance,
            }
        if not all(item["pass"] for item in comparison.values()):
            raise RuntimeError(f"Custom full-space evaluator does not reproduce standard PyKEEN metrics: {comparison}")

        return {
            "status": "PASS",
            "phase": "stage3_candidate_sensitivity_fixed_test",
            "dataset_key": identity["dataset_key"],
            "stage3_fingerprint": identity["fingerprint"],
            "frozen_manifest_sha256": frozen_sha,
            "model": model_name,
            "seed": int(seed),
            "training_instance_id": training_instance_id,
            "same_model_object_reused_across_spaces": True,
            "training_audit": training_audit,
            "full_space_standard_evaluator_crosscheck": {
                "status": "PASS",
                "metrics": comparison,
            },
            "spaces": spaces,
        }
    finally:
        if result is not None:
            del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_sensitivity(
    project_root: str | Path,
    dataset_key: str,
    *,
    model_name: str | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, base_cfg, identity, output_dir, _context_val, artifacts, _schema_audit = prepare(root, dataset_key)
    frozen_path = output_dir / "CANDIDATE_PROTOCOL_FROZEN.json"
    frozen = read_json(frozen_path)
    if not frozen or frozen.get("status") != "FROZEN":
        raise RuntimeError("Freeze the candidate-sensitivity protocol before Test evaluation")
    frozen_sha = sha256_file(frozen_path)
    models = [model_name] if model_name else list(CANDIDATE_MODELS)
    seeds = [int(seed)] if seed is not None else [int(x) for x in stage3_cfg["candidate_sensitivity"]["formal_seeds"]]
    rows = []
    for name in models:
        if name not in CANDIDATE_MODELS:
            raise ValueError(f"Unsupported candidate-sensitivity model: {name}")
        for current_seed in seeds:
            path = output_dir / "fixed_test_runs" / f"{name}_seed{current_seed}.json"
            expected = {
                "dataset_key": dataset_key,
                "stage3_fingerprint": identity["fingerprint"],
                "frozen_manifest_sha256": frozen_sha,
                "model": name,
                "seed": current_seed,
            }
            row = _cached_run(
                path,
                expected,
                lambda name=name, current_seed=current_seed: _run_model_seed(
                    root=root,
                    stage3_cfg=stage3_cfg,
                    base_cfg=base_cfg,
                    identity=identity,
                    output_dir=output_dir,
                    artifacts=artifacts,
                    model_name=name,
                    seed=current_seed,
                    device=device,
                    frozen_sha=frozen_sha,
                ),
            )
            rows.append(row)
    return {"status": "COMPLETE", "runs": rows}


def _favorable(metric: str, delta: float) -> int:
    if abs(delta) <= 1.0e-15:
        return 0
    if metric == "mr":
        return 1 if delta < 0 else -1
    return 1 if delta > 0 else -1


def summarize(project_root: str | Path, dataset_key: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage3_cfg, _base_cfg, identity, output_dir, _context, _artifacts, _schema_audit = prepare(root, dataset_key)
    frozen_path = output_dir / "CANDIDATE_PROTOCOL_FROZEN.json"
    frozen = read_json(frozen_path)
    if not frozen or frozen.get("status") != "FROZEN":
        raise RuntimeError("Candidate protocol is not frozen")
    frozen_sha = sha256_file(frozen_path)
    seeds = [int(x) for x in stage3_cfg["candidate_sensitivity"]["formal_seeds"]]

    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for model in CANDIDATE_MODELS:
        for seed in seeds:
            path = output_dir / "fixed_test_runs" / f"{model}_seed{seed}.json"
            row = read_json(path)
            if not row or row.get("status") != "PASS":
                raise RuntimeError(f"Missing successful candidate run: {path}")
            if row.get("frozen_manifest_sha256") != frozen_sha:
                raise RuntimeError(f"Frozen manifest mismatch: {path}")
            lookup[(model, seed)] = row

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model in CANDIDATE_MODELS:
        for seed in seeds:
            run = lookup[(model, seed)]
            ids = {run["spaces"][space]["training_instance_id"] for space in CANDIDATE_SPACES}
            if ids != {run["training_instance_id"]}:
                raise RuntimeError(f"Same-checkpoint audit failed for {model}/seed{seed}")
            for space in CANDIDATE_SPACES:
                item = run["spaces"][space]
                raw_rows.append(
                    {
                        "dataset": dataset_key,
                        "model": model,
                        "seed": seed,
                        "space": space,
                        **{metric: item["metrics"][metric] for metric in METRICS},
                        "nominal_candidates_mean": item["nominal_candidate_count_summary"]["mean"],
                        "nominal_candidates_median": item["nominal_candidate_count_summary"]["median"],
                        "nominal_candidates_min": item["nominal_candidate_count_summary"]["min"],
                        "nominal_candidates_max": item["nominal_candidate_count_summary"]["max"],
                        "filtered_candidates_mean": item["filtered_candidate_count_summary"]["mean"],
                        "training_instance_id": run["training_instance_id"],
                    }
                )

        for space in CANDIDATE_SPACES:
            row: dict[str, Any] = {
                "dataset": dataset_key,
                "model": model,
                "space": space,
                "n_seeds": len(seeds),
            }
            for metric in METRICS:
                agg = aggregate(
                    lookup[(model, s)]["spaces"][space]["metrics"][metric] for s in seeds
                )
                row[f"{metric}_mean"] = agg["mean"]
                row[f"{metric}_std_sample"] = agg["std_sample"]
            sample_counts = lookup[(model, seeds[0])]["spaces"][space]["nominal_candidate_count_summary"]
            row.update({f"candidate_{k}": v for k, v in sample_counts.items()})
            summary_rows.append(row)

    write_csv(output_dir / "CANDIDATE_SPACE_RESULTS.csv", raw_rows)
    write_csv(output_dir / "CANDIDATE_SPACE_SUMMARY.csv", summary_rows)

    comparisons = (("A0", "D0"), ("D2", "D1"), ("D2", "D0"))
    delta_rows: list[dict[str, Any]] = []
    for lhs, rhs in comparisons:
        for space in CANDIDATE_SPACES:
            for metric in METRICS:
                deltas = [
                    float(lookup[(lhs, s)]["spaces"][space]["metrics"][metric])
                    - float(lookup[(rhs, s)]["spaces"][space]["metrics"][metric])
                    for s in seeds
                ]
                outcomes = [_favorable(metric, d) for d in deltas]
                agg = aggregate(deltas)
                delta_rows.append(
                    {
                        "dataset": dataset_key,
                        "comparison": f"{lhs}_minus_{rhs}",
                        "space": space,
                        "metric": metric,
                        "favorable_delta": "negative" if metric == "mr" else "positive",
                        "mean_delta": agg["mean"],
                        "std_sample_delta": agg["std_sample"],
                        "wins_lhs": sum(x > 0 for x in outcomes),
                        "ties": sum(x == 0 for x in outcomes),
                        "losses_lhs": sum(x < 0 for x in outcomes),
                    }
                )
    write_csv(output_dir / "CANDIDATE_SPACE_DELTA_ANALYSIS.csv", delta_rows)

    result = {
        "status": "COMPLETE",
        "phase": "stage3_candidate_sensitivity_summary",
        "dataset_key": dataset_key,
        "stage3_fingerprint": identity["fingerprint"],
        "frozen_manifest_sha256": frozen_sha,
        "models": list(CANDIDATE_MODELS),
        "spaces": list(CANDIDATE_SPACES),
        "seeds": seeds,
        "same_trained_model_reused_across_spaces": True,
        "summary_rows": summary_rows,
        "delta_rows": delta_rows,
    }
    write_json(output_dir / "CANDIDATE_SPACE_SUMMARY.json", result)
    return result


def run_all(project_root: str | Path, dataset_key: str, *, device: str | None = None) -> dict[str, Any]:
    preflight(project_root, dataset_key)
    freeze_protocol(project_root, dataset_key)
    run_sensitivity(project_root, dataset_key, device=device)
    return summarize(project_root, dataset_key)
