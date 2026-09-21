from __future__ import annotations

"""Audited two-dataset experiment runtime.

The original OwnKG frozen workflow remains untouched. This module adds an
isolated, dataset-keyed protocol for running the same five models on OwnKG and
the supplied Paper4 reconstruction without sharing caches or result files.
"""

import csv
import gc
import json
import math
import platform
import statistics
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .bounded_adaptive_v26 import (
    build_bounded_adaptive_model_class,
    build_bounded_relation_feature_bundle,
)
from .config import load_config
from .evaluation import metric_summary
from .exact_step_v266 import build_exact_step_training_loop_class
from .gamma3_canonical_v251 import build_explicit_split_report, runtime_loss_audit
from .model import build_paper_transh_model_class, build_pykeen_model_class
from .multidataset_data import (
    audit_dataset,
    compute_training_structure_features,
    dataset_paths,
    read_entity_types,
    read_triples,
    sha256_file,
    stable_fingerprint,
    validate_config,
    write_structure_csv,
)
from .reciprocal_v263 import build_reciprocal_feature_bundle
from .v266_protocol import exact_step_budget
from .v266_runtime import AuditedBernoulliNegativeSampler


MODEL_NAMES = ("TransH", "RotatE", "D0", "D1", "A0", "D2")
METRIC_NAMES = ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10")
IMPLEMENTATION_FILES = (
    "src/throtate_repro/config.py",
    "src/throtate_repro/model.py",
    "src/throtate_repro/bounded_adaptive_v26.py",
    "src/throtate_repro/rrs_moge_v19.py",
    "src/throtate_repro/reciprocal_v263.py",
    "src/throtate_repro/exact_step_v266.py",
    "src/throtate_repro/evaluation.py",
    "src/throtate_repro/gamma3_canonical_v251.py",
    "src/throtate_repro/v266_protocol.py",
    "src/throtate_repro/v266_runtime.py",
    "src/throtate_repro/multidataset_data.py",
    "src/throtate_repro/multidataset_experiment.py",
)


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_multidataset_config(project_root: str | Path) -> dict[str, Any]:
    return load_config(Path(project_root) / "config" / "multidataset_comparison.yaml")


def _implementation_hashes(project_root: Path) -> dict[str, str]:
    return {name: sha256_file(project_root / name) for name in IMPLEMENTATION_FILES}


def experiment_identity(
    project_root: str | Path,
    cfg: Mapping[str, Any],
    dataset_key: str,
    *,
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    audit = dict(audit or audit_dataset(project_root, cfg, dataset_key))
    config_path = Path(str(cfg["_config_path"]))
    implementation = _implementation_hashes(project_root)
    try:
        pykeen_version = metadata.version("pykeen")
    except metadata.PackageNotFoundError:
        pykeen_version = "NOT_INSTALLED"
    runtime_environment = {
        "python": platform.python_version(),
        "pykeen": pykeen_version,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    payload = {
        "protocol": dict(cfg["protocol"]),
        "model": dict(cfg["model"]),
        "training": dict(cfg["training"]),
        "validation_search": dict(cfg["validation_search"]),
        "formal": dict(cfg["formal"]),
        "dataset_key": dataset_key,
        "dataset": dict(cfg["datasets"][dataset_key]),
        "actual_data_sha256": dict(audit["sha256"]),
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": implementation,
        "runtime_environment": runtime_environment,
    }
    return {
        "dataset_key": dataset_key,
        "fingerprint": stable_fingerprint(payload),
        "config_sha256": payload["config_sha256"],
        "implementation_sha256": implementation,
        "data_sha256": dict(audit["sha256"]),
        "runtime_environment": runtime_environment,
    }


def dataset_output_dir(
    project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str, fingerprint: str
) -> Path:
    return (
        Path(project_root)
        / cfg["output"]["directory"]
        / dataset_key
        / str(fingerprint)[:16]
    )


def _write_latest_pointer(
    project_root: Path,
    cfg: Mapping[str, Any],
    dataset_key: str,
    identity: Mapping[str, Any],
    output_dir: Path,
) -> None:
    base = project_root / cfg["output"]["directory"] / dataset_key
    write_json(
        base / "LATEST.json",
        {
            "dataset_key": dataset_key,
            "experiment_fingerprint": identity["fingerprint"],
            "output_directory": str(output_dir.relative_to(project_root)),
        },
    )


def _load_factory(
    path: Path,
    *,
    entity_to_id=None,
    relation_to_id=None,
    create_inverse_triples: bool = False,
):
    from pykeen.triples import TriplesFactory

    return TriplesFactory.from_path(
        path,
        create_inverse_triples=bool(create_inverse_triples),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        compact_id=False,
        load_triples_kwargs={"delimiter": "\t"},
    )


def _validate_runtime_config(cfg: Mapping[str, Any]) -> None:
    validate_config(cfg)
    training = cfg["training"]
    for name in ("max_steps", "batch_size", "num_negatives", "evaluation_batch_size"):
        if int(training[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if training.get("drop_last") is not True:
        raise ValueError("drop_last must remain true for equal exposure")
    if training.get("negative_sampler_filtered") is not False:
        raise ValueError("The formal protocol uses unfiltered Bernoulli negative sampling")
    if str(training.get("loss", "")).upper() != "NSSA":
        raise ValueError("The formal protocol uses NSSA loss")
    if str(training.get("optimizer", "")).lower() != "adam":
        raise ValueError("The formal protocol uses Adam")
    if tuple(float(x) for x in cfg["validation_search"]["baseline_gamma_candidates"]) != (
        3.0,
        6.0,
        12.0,
        24.0,
    ):
        raise ValueError("Baseline gamma candidates must remain 3/6/12/24")
    if tuple(float(x) for x in cfg["validation_search"]["d2_delta_candidates_radians"]) != (
        0.05,
        0.10,
        0.15,
    ):
        raise ValueError("D2 Delta candidates must remain 0.05/0.10/0.15")
    if int(cfg["validation_search"]["baseline_seed"]) != 42:
        raise ValueError("Baseline Validation selection seed must remain 42")
    if tuple(int(x) for x in cfg["validation_search"]["d2_delta_seeds"]) != tuple(
        int(x) for x in cfg["formal"]["seeds"]
    ):
        raise ValueError("D2 Delta selection seeds must match the five formal seeds")


def _build_context(
    project_root: Path,
    cfg: Mapping[str, Any],
    dataset_key: str,
    output_dir: Path,
    *,
    include_test: bool,
) -> dict[str, Any]:
    audit = audit_dataset(project_root, cfg, dataset_key)
    spec = cfg["datasets"][dataset_key]
    paths = dataset_paths(project_root, cfg, dataset_key)

    full = _load_factory(paths["full"])
    training = _load_factory(
        paths["train"], entity_to_id=full.entity_to_id, relation_to_id=full.relation_to_id
    )
    validation = _load_factory(
        paths["validation"], entity_to_id=full.entity_to_id, relation_to_id=full.relation_to_id
    )
    test = None
    if include_test:
        test = _load_factory(
            paths["test"], entity_to_id=full.entity_to_id, relation_to_id=full.relation_to_id
        )

    reciprocal_training = _load_factory(
        paths["train"],
        entity_to_id=full.entity_to_id,
        relation_to_id=full.relation_to_id,
        create_inverse_triples=True,
    )
    reciprocal_validation = _load_factory(
        paths["validation"],
        entity_to_id=full.entity_to_id,
        relation_to_id=full.relation_to_id,
        create_inverse_triples=False,
    )
    reciprocal_test = None
    if include_test:
        reciprocal_test = _load_factory(
            paths["test"],
            entity_to_id=full.entity_to_id,
            relation_to_id=full.relation_to_id,
            create_inverse_triples=False,
        )

    modeled_relations = tuple(str(value) for value in spec["modeled_relations"])
    entity_types = read_entity_types(paths["entity_types"])
    training_rows = read_triples(paths["train"])
    structure_rows = compute_training_structure_features(
        training_rows,
        entity_types,
        modeled_relations,
        dict(spec["relation_roles"]),
    )
    structure_path = output_dir / "artifacts" / "training_structure_features.csv"
    write_structure_csv(structure_path, structure_rows)
    base_bundle = build_bounded_relation_feature_bundle(
        structure_csv=structure_path,
        relation_to_id=full.relation_to_id,
        allowed_reasoning_relations=modeled_relations,
    )
    reciprocal_bundle = build_reciprocal_feature_bundle(
        base_bundle=base_bundle, relation_to_id=full.relation_to_id
    )

    if int(full.num_entities) != int(audit["counts"]["candidate_entities"]):
        raise ValueError("PyKEEN candidate-entity count disagrees with the data audit")
    if int(full.num_relations) != int(audit["counts"]["mapping_relations"]):
        raise ValueError("PyKEEN relation count disagrees with the data audit")
    if int(reciprocal_training.num_relations) != 2 * int(full.num_relations):
        raise ValueError("Reciprocal relation count is not twice the mapping relation count")
    if int(base_bundle.features.tensor.shape[0]) != int(full.num_relations):
        raise ValueError("Base feature rows do not match PyKEEN real relations")
    if int(base_bundle.reliability.shape[0]) != int(full.num_relations):
        raise ValueError("Base reliability rows do not match PyKEEN real relations")
    if int(reciprocal_bundle.features.shape[0]) != int(reciprocal_training.num_relations):
        raise ValueError("Reciprocal feature rows do not match PyKEEN internal relations")

    train_cfg = cfg["training"]
    ordinary_budget = exact_step_budget(
        num_real_triples=int(training.num_triples),
        batch_size=int(train_cfg["batch_size"]),
        max_steps=int(train_cfg["max_steps"]),
        reciprocal=False,
        drop_last=True,
    )
    reciprocal_budget = exact_step_budget(
        num_real_triples=int(training.num_triples),
        batch_size=int(train_cfg["batch_size"]),
        max_steps=int(train_cfg["max_steps"]),
        reciprocal=True,
        drop_last=True,
    )
    if ordinary_budget["exact_positive_exposure"] != reciprocal_budget["exact_positive_exposure"]:
        raise ValueError("Ordinary and reciprocal positive exposure differs")

    return {
        "dataset_key": dataset_key,
        "audit": audit,
        "paths": paths,
        "full": full,
        "training": training,
        "validation": validation,
        "test": test,
        "reciprocal_training": reciprocal_training,
        "reciprocal_validation": reciprocal_validation,
        "reciprocal_test": reciprocal_test,
        "base_bundle": base_bundle,
        "reciprocal_bundle": reciprocal_bundle,
        "budgets": {"ordinary": ordinary_budget, "reciprocal": reciprocal_budget},
        "modeled_relations": modeled_relations,
        "relation_id_to_label": {int(value): str(key) for key, value in full.relation_to_id.items()},
        "structure_path": structure_path,
        "test_factory_constructed": include_test,
    }


def preflight(project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str) -> dict[str, Any]:
    project_root = Path(project_root)
    _validate_runtime_config(cfg)
    required_pykeen = str(cfg["protocol"]["pykeen_version"])
    try:
        actual_pykeen = metadata.version("pykeen")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("PyKEEN is not installed. Run: python -m pip install -r requirements.txt") from error
    if actual_pykeen != required_pykeen:
        raise RuntimeError(f"PyKEEN {required_pykeen} is required, found {actual_pykeen}")

    audit = audit_dataset(project_root, cfg, dataset_key)
    identity = experiment_identity(project_root, cfg, dataset_key, audit=audit)
    output_dir = dataset_output_dir(project_root, cfg, dataset_key, identity["fingerprint"])
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _build_context(project_root, cfg, dataset_key, output_dir, include_test=False)
    summary = {
        "status": "PASS",
        "phase": "multidataset_preflight_validation_only",
        "dataset_key": dataset_key,
        "display_name": audit["display_name"],
        "identity_note": audit["identity_note"],
        "experiment_fingerprint": identity["fingerprint"],
        "config_sha256": identity["config_sha256"],
        "implementation_sha256": identity["implementation_sha256"],
        "runtime_environment": identity["runtime_environment"],
        "data_audit": audit,
        "pykeen_version": actual_pykeen,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_budgets": context["budgets"],
        "structure_features_file": str(context["structure_path"].relative_to(project_root)),
        "test_file_integrity_checked": True,
        "test_factory_constructed": False,
        "test_metrics_used": False,
    }
    write_json(output_dir / "PRECHECK_VALIDATION_ONLY.json", summary)
    _write_latest_pointer(project_root, cfg, dataset_key, identity, output_dir)
    return summary


def _current_run(
    project_root: Path, cfg: Mapping[str, Any], dataset_key: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    audit = audit_dataset(project_root, cfg, dataset_key)
    identity = experiment_identity(project_root, cfg, dataset_key, audit=audit)
    output_dir = dataset_output_dir(project_root, cfg, dataset_key, identity["fingerprint"])
    precheck = read_json(output_dir / "PRECHECK_VALIDATION_ONLY.json")
    if not precheck or precheck.get("status") != "PASS":
        raise RuntimeError(f"Run preflight first for dataset {dataset_key}")
    if precheck.get("experiment_fingerprint") != identity["fingerprint"]:
        raise RuntimeError("Preflight belongs to a different experiment fingerprint")
    return audit, identity, output_dir


def _common_model_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = cfg["model"]
    return {
        "embedding_dim": int(model_cfg["embedding_dim"]),
        "transh_norm": int(model_cfg["transh_norm"]),
        "transh_power_norm": bool(model_cfg["transh_power_norm"]),
        "rotate_norm": int(model_cfg["rotate_norm"]),
    }


def _model_spec(
    cfg: Mapping[str, Any], context: Mapping[str, Any], model_name: str, delta: float | None
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")
    common = _common_model_kwargs(cfg)
    if model_name == "TransH":
        return {
            "model": build_paper_transh_model_class(),
            "model_kwargs": {
                "embedding_dim": common["embedding_dim"],
                "transh_norm": common["transh_norm"],
                "transh_power_norm": common["transh_power_norm"],
            },
            "reciprocal": False,
            "implementation": "throtate_repro.model.PaperTransH",
        }
    if model_name == "RotatE":
        from pykeen.models import RotatE

        return {
            "model": RotatE,
            "model_kwargs": {"embedding_dim": common["embedding_dim"]},
            "reciprocal": False,
            "implementation": "pykeen.models.RotatE",
        }
    if model_name == "D0":
        return {
            "model": build_pykeen_model_class(),
            "model_kwargs": common,
            "reciprocal": False,
            "implementation": "throtate_repro.model.THRotatE",
        }
    if model_name == "D1":
        return {
            "model": build_pykeen_model_class(),
            "model_kwargs": common,
            "reciprocal": True,
            "implementation": "throtate_repro.model.THRotatE+reciprocal",
        }
    if model_name == "A0":
        if delta is None:
            raise ValueError("A0 requires an explicit Delta")
        bundle = context["base_bundle"]
        return {
            "model": build_bounded_adaptive_model_class(),
            "model_kwargs": {
                **common,
                "relation_feature_tensor": bundle.features.tensor,
                "relation_reliability_tensor": bundle.reliability,
                "max_angle_deviation": float(delta),
            },
            "reciprocal": False,
            "implementation": (
                "throtate_repro.bounded_adaptive_v26.BoundedAdaptiveTHRotatE"
                "+bounded_adapter_without_reciprocal"
            ),
        }
    if model_name == "D2":
        if delta is None:
            raise ValueError("D2 requires an explicit Delta")
        bundle = context["reciprocal_bundle"]
        return {
            "model": build_bounded_adaptive_model_class(),
            "model_kwargs": {
                **common,
                "relation_feature_tensor": bundle.features,
                "relation_reliability_tensor": bundle.reliability,
                "max_angle_deviation": float(delta),
            },
            "reciprocal": True,
            "implementation": "throtate_repro.bounded_adaptive_v26.BoundedAdaptiveTHRotatE+reciprocal",
        }
    raise AssertionError(f"Unhandled model name: {model_name}")


def _directional_metrics(
    *, model, mapped_triples, truth_triples: Sequence, batch_size: int, filtered: bool
) -> dict[str, Any]:
    from pykeen.evaluation import RankBasedEvaluator

    output: dict[str, Any] = {}
    for label, targets, target_name in (
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
        output[label] = metric_summary(result, target=target_name)
    return output


def _relation_fusion_rows(model, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not hasattr(model, "get_all_relation_fusion_state"):
        return []
    state = model.get_all_relation_fusion_state()
    weights = state["weights"]
    shifts = state["angle_shift"]
    reliability = state["reliability"]
    global_alpha, global_beta = model.get_global_fusion_weights()
    theta_global = math.atan2(float(global_beta), float(global_alpha))
    rows: list[dict[str, Any]] = []
    label_to_id = context["full"].relation_to_id
    real_relation_count = int(context["full"].num_relations)
    internal_rows = int(weights.shape[0])

    if internal_rows == real_relation_count:
        # A0: bounded adapter on ordinary TH-RotatE training. PyKEEN relation IDs are
        # the real relation IDs, so there is only one learned direction per relation.
        direction_layout = "ordinary"
        def ids_for(real_id: int):
            return (("ordinary", real_id),)
    elif internal_rows == 2 * real_relation_count:
        # D2: reciprocal training. PyKEEN stores real relation r as 2*r and its
        # artificial inverse as 2*r+1.
        direction_layout = "reciprocal"
        def ids_for(real_id: int):
            return (("forward", 2 * real_id), ("inverse", 2 * real_id + 1))
    else:
        raise RuntimeError(
            f"Unexpected relation-fusion row count: {internal_rows}; "
            f"expected {real_relation_count} or {2 * real_relation_count}"
        )

    for relation in context["modeled_relations"]:
        real_id = int(label_to_id[relation])
        for direction, internal_id in ids_for(real_id):
            alpha = float(weights[internal_id, 0])
            beta = float(weights[internal_id, 1])
            rows.append(
                {
                    "relation": relation,
                    "real_relation_id": real_id,
                    "internal_relation_id": internal_id,
                    "direction": direction,
                    "direction_layout": direction_layout,
                    "reliability_q": float(reliability[internal_id, 0]),
                    "theta_global_radians": theta_global,
                    "theta_shift_radians": float(shifts[internal_id]),
                    "theta_relation_radians": math.atan2(beta, alpha),
                    "alpha": alpha,
                    "beta": beta,
                }
            )
    return rows


def train_and_evaluate(
    *,
    cfg: Mapping[str, Any],
    context: Mapping[str, Any],
    identity: Mapping[str, Any],
    model_name: str,
    seed: int,
    gamma: float,
    delta: float | None,
    evaluation_split: str,
    frozen_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    from pykeen.pipeline import pipeline

    if evaluation_split not in {"validation", "test"}:
        raise ValueError(evaluation_split)
    if evaluation_split == "test" and context.get("test") is None:
        raise RuntimeError("Test factory has not been constructed")

    spec = _model_spec(cfg, context, model_name, delta)
    reciprocal = bool(spec["reciprocal"])
    training_factory = context["reciprocal_training"] if reciprocal else context["training"]
    evaluation_factory = (
        context[f"reciprocal_{evaluation_split}"] if reciprocal else context[evaluation_split]
    )
    truth_factories = [context["training"], context["validation"]]
    if evaluation_split == "test":
        truth_factories.append(context["test"])

    train_cfg = cfg["training"]
    budget = context["budgets"]["reciprocal" if reciprocal else "ordinary"]
    expected_positive = int(train_cfg["max_steps"]) * int(train_cfg["batch_size"])
    expected_negative = expected_positive * int(train_cfg["num_negatives"])
    device_value = str(train_cfg.get("device", "auto"))
    device = None if device_value == "auto" else device_value

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
            evaluator_kwargs={"filtered": bool(cfg["protocol"]["filtered_evaluation"])},
            evaluation_kwargs={"batch_size": int(train_cfg["evaluation_batch_size"])},
            use_testing_data=False,
            filter_validation_when_testing=False,
            random_seed=int(seed),
            device=device,
        )
        wall_seconds = time.perf_counter() - started
        actual_steps = int(getattr(result.training_loop, "v266_optimizer_steps", -1))
        if actual_steps != int(train_cfg["max_steps"]):
            raise RuntimeError(
                f"Exact-step audit failed: requested={train_cfg['max_steps']}, actual={actual_steps}"
            )
        sampling = AuditedBernoulliNegativeSampler.audit_snapshot(
            num_negatives_per_positive=int(train_cfg["num_negatives"])
        )
        if int(sampling["actual_positive_instances"]) != expected_positive:
            raise RuntimeError(f"Positive-exposure audit failed: {sampling}")
        if int(sampling["requested_negatives"]) != expected_negative:
            raise RuntimeError(f"Negative-exposure audit failed: {sampling}")
        if sampling["all_negatives_accepted"] is not True:
            raise RuntimeError(f"Unexpected negative filtering: {sampling}")
        loss_audit = runtime_loss_audit(result.model, float(gamma))
        if not loss_audit["pass"]:
            raise RuntimeError(f"NSSA margin audit failed: {loss_audit}")

        filtered = bool(cfg["protocol"]["filtered_evaluation"])
        report = build_explicit_split_report(
            model=result.model,
            evaluation_factory=evaluation_factory,
            truth_factories=truth_factories,
            batch_size=int(train_cfg["evaluation_batch_size"]),
            filtered=filtered,
            include_per_relation=evaluation_split == "test",
        )
        primary = dict(report["all_relations_both_sides"]["metrics"])
        invalid_metrics = {
            metric: primary.get(metric)
            for metric in METRIC_NAMES
            if primary.get(metric) is None or not math.isfinite(float(primary[metric]))
        }
        if invalid_metrics:
            raise RuntimeError(f"Missing or non-finite primary metrics: {invalid_metrics}")
        expected_ranks = 2 * int(evaluation_factory.num_triples)
        if int(report["all_relations_both_sides"]["rank_count"]) != expected_ranks:
            raise RuntimeError("Both-side rank-count audit failed")
        directional = _directional_metrics(
            model=result.model,
            mapped_triples=evaluation_factory.mapped_triples,
            truth_triples=[factory.mapped_triples for factory in truth_factories],
            batch_size=int(train_cfg["evaluation_batch_size"]),
            filtered=filtered,
        )
        relation_state = _relation_fusion_rows(result.model, context)
        if relation_state and delta is not None:
            max_shift = max(abs(float(row["theta_shift_radians"])) for row in relation_state)
            if max_shift > float(delta) + 1.0e-6:
                raise RuntimeError("Bounded-adapter relation shift exceeded the selected Delta")

        fusion = None
        if hasattr(result.model, "get_global_fusion_weights"):
            fusion = list(result.model.get_global_fusion_weights())
        elif hasattr(result.model, "get_fusion_weights"):
            fusion = list(result.model.get_fusion_weights())
        return {
            "status": "PASS",
            "phase": f"multidataset_{evaluation_split}",
            "dataset_key": context["dataset_key"],
            "experiment_fingerprint": identity["fingerprint"],
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "model": model_name,
            "implementation": spec["implementation"],
            "seed": int(seed),
            "gamma": float(gamma),
            "delta_angle_radians": float(delta) if delta is not None else None,
            "reciprocal_training": reciprocal,
            "evaluation_split": evaluation_split,
            "split_sha256": context["audit"]["sha256"][evaluation_split],
            "metrics": primary,
            "directional_metrics": directional,
            "per_relation_tail_only": report.get("per_relation_tail_only", {}),
            "relation_fusion_state": relation_state,
            "global_fusion_weights": fusion,
            "num_model_parameters": int(sum(parameter.numel() for parameter in result.model.parameters())),
            "num_trainable_model_parameters": int(
                sum(parameter.numel() for parameter in result.model.parameters() if parameter.requires_grad)
            ),
            "actual_optimizer_steps": actual_steps,
            "training_budget": budget,
            "sampling_exposure_audit": sampling,
            "runtime_loss_audit": loss_audit,
            "train_seconds": float(result.train_seconds),
            "wall_seconds": float(wall_seconds),
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


def _cached_run(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    call,
) -> dict[str, Any]:
    old = read_json(path)
    if old and old.get("status") == "PASS":
        mismatches = {
            key: {"expected": value, "actual": old.get(key)}
            for key, value in expected_identity.items()
            if old.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Cached result identity mismatch at {path}: {mismatches}")
        return old
    try:
        row = call()
        row.update(expected_identity)
        write_json(path, row)
        return row
    except Exception as error:
        failure = {
            "status": "FAIL",
            **dict(expected_identity),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(path, failure)
        raise


def screen_validation(
    project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str
) -> dict[str, Any]:
    project_root = Path(project_root)
    _audit, identity, output_dir = _current_run(project_root, cfg, dataset_key)
    context = _build_context(project_root, cfg, dataset_key, output_dir, include_test=False)
    search = cfg["validation_search"]
    baseline_seed = int(search["baseline_seed"])

    baseline_selection: dict[str, Any] = {}
    for model_name in ("TransH", "RotatE"):
        runs = []
        for gamma in (float(value) for value in search["baseline_gamma_candidates"]):
            gamma_label = f"{gamma:g}".replace(".", "p")
            path = output_dir / "validation_screen" / model_name / f"gamma_{gamma_label}_seed{baseline_seed}.json"
            expected = {
                "dataset_key": dataset_key,
                "experiment_fingerprint": identity["fingerprint"],
                "model": model_name,
                "seed": baseline_seed,
                "gamma": gamma,
                "delta_angle_radians": None,
                "evaluation_split": "validation",
            }
            row = _cached_run(
                path,
                expected_identity=expected,
                call=lambda model_name=model_name, gamma=gamma: train_and_evaluate(
                    cfg=cfg,
                    context=context,
                    identity=identity,
                    model_name=model_name,
                    seed=baseline_seed,
                    gamma=gamma,
                    delta=None,
                    evaluation_split="validation",
                ),
            )
            runs.append(row)
        selected = max(
            runs,
            key=lambda row: (
                float(row["metrics"]["mrr"]),
                float(row["metrics"]["hits_at_1"]),
                -float(row["gamma"]),
            ),
        )
        baseline_selection[model_name] = {
            "selected_gamma": float(selected["gamma"]),
            "selected_validation_metrics": dict(selected["metrics"]),
            "candidates": [
                {"gamma": float(row["gamma"]), **dict(row["metrics"])} for row in runs
            ],
        }

    adapter_delta_seeds = tuple(int(value) for value in search["d2_delta_seeds"])
    adapter_delta_candidates = tuple(float(value) for value in search["d2_delta_candidates_radians"])

    def _screen_adapter_model(model_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for delta in adapter_delta_candidates:
            for seed in adapter_delta_seeds:
                delta_label = f"{delta:.2f}".replace(".", "p")
                path = output_dir / "validation_screen" / model_name / f"delta_{delta_label}_seed{seed}.json"
                expected = {
                    "dataset_key": dataset_key,
                    "experiment_fingerprint": identity["fingerprint"],
                    "model": model_name,
                    "seed": seed,
                    "gamma": float(cfg["training"]["gamma"]),
                    "delta_angle_radians": delta,
                    "evaluation_split": "validation",
                }
                row = _cached_run(
                    path,
                    expected_identity=expected,
                    call=lambda model_name=model_name, delta=delta, seed=seed: train_and_evaluate(
                        cfg=cfg,
                        context=context,
                        identity=identity,
                        model_name=model_name,
                        seed=seed,
                        gamma=float(cfg["training"]["gamma"]),
                        delta=delta,
                        evaluation_split="validation",
                    ),
                )
                runs.append(row)

        aggregated: list[dict[str, Any]] = []
        for delta in adapter_delta_candidates:
            rows = [row for row in runs if float(row["delta_angle_radians"]) == delta]
            aggregated.append(
                {
                    "delta": delta,
                    "n": len(rows),
                    "seeds": [int(row["seed"]) for row in rows],
                    "metrics": {
                        metric: _aggregate([float(row["metrics"][metric]) for row in rows])
                        for metric in METRIC_NAMES
                    },
                }
            )
        selected = min(
            aggregated,
            key=lambda row: (-float(row["metrics"]["mrr"]["mean"]), float(row["delta"])),
        )
        return runs, aggregated, selected

    a0_delta_runs, a0_delta_aggregated, selected_a0_delta = _screen_adapter_model("A0")
    d2_delta_runs, d2_delta_aggregated, selected_d2_delta = _screen_adapter_model("D2")
    summary = {
        "status": "COMPLETE",
        "phase": "multidataset_validation_only_parameter_selection",
        "dataset_key": dataset_key,
        "experiment_fingerprint": identity["fingerprint"],
        "baseline_selection": baseline_selection,
        "a0_selection": {
            "selected_delta": float(selected_a0_delta["delta"]),
            "selection_rule": "highest five-seed mean filtered both-side Validation MRR; exact tie uses smaller Delta",
            "candidates": a0_delta_aggregated,
        },
        "d2_selection": {
            "selected_delta": float(selected_d2_delta["delta"]),
            "selection_rule": "highest five-seed mean filtered both-side Validation MRR; exact tie uses smaller Delta",
            "candidates": d2_delta_aggregated,
        },
        "completed_training_runs": len(a0_delta_runs) + len(d2_delta_runs) + 8,
        "test_factory_constructed": False,
        "test_metrics_used": False,
    }
    write_json(output_dir / "VALIDATION_SELECTION_SUMMARY.json", summary)
    return summary


def freeze_protocol(
    project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str
) -> dict[str, Any]:
    project_root = Path(project_root)
    audit, identity, output_dir = _current_run(project_root, cfg, dataset_key)
    selection_path = output_dir / "VALIDATION_SELECTION_SUMMARY.json"
    selection = read_json(selection_path)
    if not selection or selection.get("status") != "COMPLETE":
        raise RuntimeError(f"Run Validation selection first for {dataset_key}")
    if selection.get("experiment_fingerprint") != identity["fingerprint"]:
        raise RuntimeError("Validation selection belongs to another experiment fingerprint")

    manifest = {
        "status": "FROZEN",
        "phase": "multidataset_protocol_freeze",
        "dataset_key": dataset_key,
        "display_name": audit["display_name"],
        "identity_note": audit["identity_note"],
        "experiment_fingerprint": identity["fingerprint"],
        "config_sha256": identity["config_sha256"],
        "implementation_sha256": identity["implementation_sha256"],
        "runtime_environment": identity["runtime_environment"],
        "data_sha256": identity["data_sha256"],
        "validation_selection_sha256": sha256_file(selection_path),
        "selected_baseline_gamma": {
            model: float(selection["baseline_selection"][model]["selected_gamma"])
            for model in ("TransH", "RotatE")
        },
        "selected_a0_delta_radians": float(selection["a0_selection"]["selected_delta"]),
        "selected_d2_delta_radians": float(selection["d2_selection"]["selected_delta"]),
        "fixed_d0_d1_a0_d2_gamma": float(cfg["training"]["gamma"]),
        "fixed_d0_d1_d2_gamma": float(cfg["training"]["gamma"]),
        "formal_models": list(cfg["formal"]["models"]),
        "formal_seeds": [int(value) for value in cfg["formal"]["seeds"]],
        "test_sha256": audit["sha256"]["test"],
        "test_triples": int(audit["counts"]["test_triples"]),
        "test_used_for_parameter_selection": False,
        "model_or_hyperparameter_changes_after_test_allowed": False,
    }
    path = output_dir / "PROTOCOL_FROZEN.json"
    old = read_json(path)
    if old is not None and old != manifest:
        raise RuntimeError(f"A different protocol is already frozen at {path}")
    write_json(path, manifest)
    return manifest


def _selected_parameters(cfg: Mapping[str, Any], frozen: Mapping[str, Any], model_name: str) -> tuple[float, float | None]:
    if model_name in {"TransH", "RotatE"}:
        gamma = float(frozen["selected_baseline_gamma"][model_name])
    else:
        gamma = float(frozen.get("fixed_d0_d1_a0_d2_gamma", frozen["fixed_d0_d1_d2_gamma"]))
    if model_name == "A0":
        delta = float(frozen["selected_a0_delta_radians"])
    elif model_name == "D2":
        delta = float(frozen["selected_d2_delta_radians"])
    else:
        delta = None
    return gamma, delta


def run_fixed_test(
    project_root: str | Path,
    cfg: Mapping[str, Any],
    dataset_key: str,
    *,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    _audit, identity, output_dir = _current_run(project_root, cfg, dataset_key)
    frozen_path = output_dir / "PROTOCOL_FROZEN.json"
    frozen = read_json(frozen_path)
    if not frozen or frozen.get("status") != "FROZEN":
        raise RuntimeError(f"Protocol is not frozen for {dataset_key}")
    if frozen.get("experiment_fingerprint") != identity["fingerprint"]:
        raise RuntimeError("Frozen protocol fingerprint mismatch")
    frozen_sha = sha256_file(frozen_path)

    started_path = output_dir / "FIXED_TEST_STARTED.json"
    started = read_json(started_path)
    if started and started.get("frozen_manifest_sha256") != frozen_sha:
        raise RuntimeError("Frozen protocol changed after Test execution started")
    if not started:
        write_json(
            started_path,
            {
                "status": "STARTED",
                "dataset_key": dataset_key,
                "frozen_manifest_sha256": frozen_sha,
                "test_sha256": frozen["test_sha256"],
                "warning": "Fixed Test is now being evaluated; do not tune from these metrics.",
            },
        )

    requested_models = tuple(models or frozen["formal_models"])
    requested_seeds = tuple(int(value) for value in (seeds or frozen["formal_seeds"]))
    if not set(requested_models) <= set(frozen["formal_models"]):
        raise ValueError(f"Unsupported requested models: {requested_models}")
    if not set(requested_seeds) <= {int(value) for value in frozen["formal_seeds"]}:
        raise ValueError(f"Unsupported requested seeds: {requested_seeds}")

    context = _build_context(project_root, cfg, dataset_key, output_dir, include_test=True)
    completed = []
    for model_name in requested_models:
        gamma, delta = _selected_parameters(cfg, frozen, model_name)
        for seed in requested_seeds:
            path = output_dir / "fixed_test_runs" / f"{model_name}_seed{seed}.json"
            expected = {
                "dataset_key": dataset_key,
                "experiment_fingerprint": identity["fingerprint"],
                "frozen_manifest_sha256": frozen_sha,
                "model": model_name,
                "seed": int(seed),
                "gamma": gamma,
                "delta_angle_radians": delta,
                "evaluation_split": "test",
                "split_sha256": frozen["test_sha256"],
            }
            row = _cached_run(
                path,
                expected_identity=expected,
                call=lambda model_name=model_name, seed=seed, gamma=gamma, delta=delta: train_and_evaluate(
                    cfg=cfg,
                    context=context,
                    identity=identity,
                    model_name=model_name,
                    seed=seed,
                    gamma=gamma,
                    delta=delta,
                    evaluation_split="test",
                    frozen_manifest_sha256=frozen_sha,
                ),
            )
            completed.append(row)

    expected_count = len(frozen["formal_models"]) * len(frozen["formal_seeds"])
    available = [
        read_json(output_dir / "fixed_test_runs" / f"{model}_seed{seed}.json")
        for model in frozen["formal_models"]
        for seed in frozen["formal_seeds"]
    ]
    passed_count = sum(bool(row and row.get("status") == "PASS") for row in available)
    progress = {
        "status": "COMPLETE" if passed_count == expected_count else "PARTIAL",
        "dataset_key": dataset_key,
        "experiment_fingerprint": identity["fingerprint"],
        "runs_completed_this_call": len(completed),
        "total_passed_runs": passed_count,
        "expected_total_runs": expected_count,
        "remaining_runs": expected_count - passed_count,
    }
    write_json(output_dir / "FIXED_TEST_PROGRESS.json", progress)
    if passed_count == expected_count:
        progress["dataset_summary"] = summarize_dataset(project_root, cfg, dataset_key)
    return progress


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot aggregate an empty sequence")
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "std_sample": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "values": numbers,
    }


def _paired_comparison(
    rows: Mapping[tuple[str, int], Mapping[str, Any]], candidate: str, reference: str, seeds: Sequence[int]
) -> dict[str, Any]:
    by_seed = []
    for seed in seeds:
        candidate_metrics = rows[(candidate, int(seed))]["metrics"]
        reference_metrics = rows[(reference, int(seed))]["metrics"]
        delta = {
            metric: float(candidate_metrics[metric]) - float(reference_metrics[metric])
            for metric in METRIC_NAMES
        }
        by_seed.append({"seed": int(seed), "absolute_delta": delta})
    return {
        "candidate": candidate,
        "reference": reference,
        "mean_absolute_delta": {
            metric: statistics.fmean(row["absolute_delta"][metric] for row in by_seed)
            for metric in METRIC_NAMES
        },
        "wins": {
            metric: sum(
                (row["absolute_delta"][metric] < 0 if metric == "mr" else row["absolute_delta"][metric] > 0)
                for row in by_seed
            )
            for metric in METRIC_NAMES
        },
        "by_seed": by_seed,
    }


def summarize_dataset(
    project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str
) -> dict[str, Any]:
    project_root = Path(project_root)
    audit, identity, output_dir = _current_run(project_root, cfg, dataset_key)
    frozen_path = output_dir / "PROTOCOL_FROZEN.json"
    frozen = read_json(frozen_path)
    if not frozen or frozen.get("status") != "FROZEN":
        raise RuntimeError(f"Protocol is not frozen for {dataset_key}")
    frozen_sha = sha256_file(frozen_path)
    seeds = tuple(int(value) for value in frozen["formal_seeds"])
    models = tuple(str(value) for value in frozen["formal_models"])

    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for model_name in models:
        for seed in seeds:
            path = output_dir / "fixed_test_runs" / f"{model_name}_seed{seed}.json"
            row = read_json(path)
            if not row or row.get("status") != "PASS":
                raise RuntimeError(f"Missing completed fixed-Test run: {path}")
            if row.get("experiment_fingerprint") != identity["fingerprint"]:
                raise RuntimeError(f"Run fingerprint mismatch: {path}")
            if row.get("frozen_manifest_sha256") != frozen_sha:
                raise RuntimeError(f"Run frozen-manifest mismatch: {path}")
            if row.get("split_sha256") != frozen["test_sha256"]:
                raise RuntimeError(f"Run Test hash mismatch: {path}")
            if row.get("model") != model_name or int(row.get("seed", -1)) != seed:
                raise RuntimeError(f"Run identity mismatch: {path}")
            if int(row.get("actual_optimizer_steps", -1)) != int(cfg["training"]["max_steps"]):
                raise RuntimeError(f"Run optimizer-step mismatch: {path}")
            expected_gamma, expected_delta = _selected_parameters(cfg, frozen, model_name)
            if float(row.get("gamma", float("nan"))) != expected_gamma:
                raise RuntimeError(f"Run gamma mismatch: {path}")
            actual_delta = row.get("delta_angle_radians")
            if actual_delta != expected_delta:
                raise RuntimeError(f"Run Delta mismatch: {path}")
            lookup[(model_name, seed)] = row

    aggregated = {
        model_name: {
            "n": len(seeds),
            "gamma": float(lookup[(model_name, seeds[0])]["gamma"]),
            "delta_angle_radians": lookup[(model_name, seeds[0])]["delta_angle_radians"],
            "num_model_parameters": _aggregate(
                [float(lookup[(model_name, seed)]["num_model_parameters"]) for seed in seeds]
            ),
            "metrics": {
                metric: _aggregate(
                    [float(lookup[(model_name, seed)]["metrics"][metric]) for seed in seeds]
                )
                for metric in METRIC_NAMES
            },
        }
        for model_name in models
    }
    comparisons = {
        f"D2_vs_{reference}": _paired_comparison(lookup, "D2", reference, seeds)
        for reference in models
        if reference != "D2"
    }
    if "A0" in models and "D0" in models:
        comparisons["A0_vs_D0"] = _paired_comparison(lookup, "A0", "D0", seeds)
    if "D1" in models and "D0" in models:
        comparisons["D1_vs_D0"] = _paired_comparison(lookup, "D1", "D0", seeds)

    table_rows: list[dict[str, Any]] = []
    for model_name in models:
        row: dict[str, Any] = {
            "dataset": dataset_key,
            "dataset_display_name": audit["display_name"],
            "model": model_name,
            "n_seeds": len(seeds),
            "gamma": aggregated[model_name]["gamma"],
            "delta_angle_radians": aggregated[model_name]["delta_angle_radians"],
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = aggregated[model_name]["metrics"][metric]["mean"]
            row[f"{metric}_std_sample"] = aggregated[model_name]["metrics"][metric]["std_sample"]
        table_rows.append(row)

    csv_path = output_dir / "DATASET_COMPARISON_TABLE.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    summary = {
        "status": "COMPLETE",
        "phase": "multidataset_fixed_test_summary",
        "dataset_key": dataset_key,
        "display_name": audit["display_name"],
        "identity_note": audit["identity_note"],
        "experiment_fingerprint": identity["fingerprint"],
        "frozen_manifest_sha256": frozen_sha,
        "test_sha256": frozen["test_sha256"],
        "test_triples": int(audit["counts"]["test_triples"]),
        "both_side_ranks": 2 * int(audit["counts"]["test_triples"]),
        "candidate_entities": int(audit["counts"]["candidate_entities"]),
        "test_relations_present": list(audit["test_relations"]),
        "configured_modeled_relations": list(audit["modeled_relations"]),
        "selected_parameters": {
            "baseline_gamma": dict(frozen["selected_baseline_gamma"]),
            "d0_d1_a0_d2_gamma": float(frozen.get("fixed_d0_d1_a0_d2_gamma", frozen["fixed_d0_d1_d2_gamma"])),
            "a0_delta_radians": float(frozen["selected_a0_delta_radians"]),
            "d2_delta_radians": float(frozen["selected_d2_delta_radians"]),
        },
        "aggregated": aggregated,
        "paired_comparisons": comparisons,
        "paper_table_rows": table_rows,
        "test_used_for_parameter_selection": False,
        "reporting_guard": audit["identity_note"],
    }
    write_json(output_dir / "DATASET_COMPARISON_SUMMARY.json", summary)
    return summary


def summarize_two_datasets(project_root: str | Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    project_root = Path(project_root)
    summaries = {
        dataset_key: summarize_dataset(project_root, cfg, dataset_key)
        for dataset_key in cfg["datasets"]
    }
    table_rows = [
        row
        for dataset_key in cfg["datasets"]
        for row in summaries[dataset_key]["paper_table_rows"]
    ]
    output_root = project_root / cfg["output"]["directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "TWO_DATASET_COMPARISON_TABLE.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    d2_vs_d0 = {}
    for dataset_key, summary in summaries.items():
        d2 = summary["aggregated"]["D2"]["metrics"]
        d0 = summary["aggregated"]["D0"]["metrics"]
        d2_vs_d0[dataset_key] = {
            "mrr_absolute_gain": float(d2["mrr"]["mean"]) - float(d0["mrr"]["mean"]),
            "mrr_relative_gain": float(d2["mrr"]["mean"]) / float(d0["mrr"]["mean"]) - 1.0,
            "hits_at_1_absolute_gain": float(d2["hits_at_1"]["mean"])
            - float(d0["hits_at_1"]["mean"]),
            "hits_at_10_absolute_gain": float(d2["hits_at_10"]["mean"])
            - float(d0["hits_at_10"]["mean"]),
        }
    combined = {
        "status": "COMPLETE",
        "phase": "two_dataset_comparison",
        "datasets": {
            key: {
                "display_name": value["display_name"],
                "identity_note": value["identity_note"],
                "experiment_fingerprint": value["experiment_fingerprint"],
                "summary_file": str(
                    (
                        dataset_output_dir(project_root, cfg, key, value["experiment_fingerprint"])
                        / "DATASET_COMPARISON_SUMMARY.json"
                    ).relative_to(project_root)
                ),
            }
            for key, value in summaries.items()
        },
        "d2_vs_d0": d2_vs_d0,
        "paper_table_rows": table_rows,
        "cross_dataset_averaging_performed": False,
        "reporting_note": (
            "Report each dataset separately. Paper4 is a same-source conservative reconstruction, "
            "not the exact final CROEFKG from the cited paper."
        ),
    }
    write_json(output_root / "TWO_DATASET_COMPARISON_SUMMARY.json", combined)
    return combined


def run_all(project_root: str | Path, cfg: Mapping[str, Any], dataset_key: str) -> dict[str, Any]:
    preflight(project_root, cfg, dataset_key)
    screen_validation(project_root, cfg, dataset_key)
    freeze_protocol(project_root, cfg, dataset_key)
    return run_fixed_test(project_root, cfg, dataset_key)
