from __future__ import annotations

import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .bounded_adaptive_residual_v263 import build_residual_bounded_adaptive_model_class
from .bounded_adaptive_v26 import build_bounded_adaptive_model_class, build_bounded_relation_feature_bundle
from .evaluation import metric_summary
from .gamma3_canonical_v251 import build_explicit_split_report, load_factory, runtime_loss_audit, training_budget
from .model import build_pykeen_model_class
from .reciprocal_v263 import (
    build_exact_step_reciprocal_training_loop_class,
    build_reciprocal_feature_bundle,
    reciprocal_training_budget,
)
from .role_control_v18 import REASONING14_RELATIONS

EXPECTED_TOKEN = "OWNKGC_V263_R14_VALIDATION_ONLY_RESIDUAL_RECIPROCAL"
EXPECTED_GAMMA = 3.0
FROZEN_DELTA = 0.10
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("V0", "V1", "V2", "V3")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["protocol"]["token"] != EXPECTED_TOKEN:
        raise ValueError("v2.6.3 protocol token mismatch")
    if float(cfg["training"]["gamma"]) != EXPECTED_GAMMA:
        raise ValueError("v2.6.3 must keep gamma=3")
    if int(cfg["training"]["max_steps"]) != 2000:
        raise ValueError("v2.6.3 must keep exactly 2000 optimizer steps")
    if int(cfg["training"]["batch_size"]) != 1024:
        raise ValueError("v2.6.3 must keep batch_size=1024")
    if int(cfg["training"]["num_negatives"]) != 64:
        raise ValueError("v2.6.3 must keep 64 negatives")
    if abs(float(cfg["model"]["frozen_delta_angle_radians"]) - FROZEN_DELTA) > 1.0e-12:
        raise ValueError("Delta is frozen at 0.10 from completed v2.6 development")
    if tuple(int(x) for x in cfg["development"]["formal_seeds"]) != EXPECTED_SEEDS:
        raise ValueError("Formal seeds must remain 42/43/44/45/46")
    if tuple(cfg["development"]["variants"]) != VARIANTS:
        raise ValueError("Variants must remain V0/V1/V2/V3")
    if cfg["protocol"].get("typed_negative_sampling_removed") is not True:
        raise ValueError("Typed-negative module must stay removed from v2.6.3")
    if cfg["protocol"].get("test_loaded") is not False:
        raise ValueError("Test must remain sealed")


def validation_paths(project_root: Path, cfg: Mapping[str, Any]) -> dict[str, Path]:
    split_dir = project_root / cfg["data"]["split_dir"]
    return {
        "full17": project_root / cfg["data"]["full17_file"],
        "structure": project_root / cfg["data"]["training_structure_features"],
        "train": split_dir / "train.tsv",
        "validation": split_dir / "valid.tsv",
    }


def validate_hashes(project_root: Path, cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = validation_paths(project_root, cfg)
    expected = cfg["data"]["sha256"]
    report = {}
    for key in ("full17", "structure", "train", "validation"):
        expected_key = "training_structure_features" if key == "structure" else key
        actual = sha256_file(paths[key])
        target = str(expected[expected_key])
        report[key] = {"path": str(paths[key]), "expected_sha256": target, "actual_sha256": actual, "match": actual == target}
    return report


def _load_factory_with_inverse(path: Path, *, entity_to_id, relation_to_id, create_inverse_triples: bool):
    from pykeen.triples import TriplesFactory

    return TriplesFactory.from_path(
        path,
        create_inverse_triples=bool(create_inverse_triples),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        compact_id=False,
        load_triples_kwargs=dict(delimiter="\t"),
    )


def load_context(project_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Load Validation-only ordinary and reciprocal contexts. No Test path is constructed."""
    validate_config(cfg)
    hashes = validate_hashes(project_root, cfg)
    if not all(row["match"] for row in hashes.values()):
        raise ValueError("v2.6.3 data hash audit failed")
    paths = validation_paths(project_root, cfg)
    full = load_factory(paths["full17"])
    training = load_factory(paths["train"], entity_to_id=full.entity_to_id, relation_to_id=full.relation_to_id)
    validation = load_factory(paths["validation"], entity_to_id=full.entity_to_id, relation_to_id=full.relation_to_id)
    expected = cfg["data"]["expected"]
    checks = {
        "candidate_entities": int(full.num_entities) == int(expected["candidate_entities"]),
        "mapped_relations": int(full.num_relations) == int(expected["mapped_relations"]),
        "train_triples": int(training.num_triples) == int(expected["train_triples"]),
        "validation_triples": int(validation.num_triples) == int(expected["validation_triples"]),
        "validation_both_side_ranks": 2 * int(validation.num_triples) == int(expected["validation_both_side_ranks"]),
    }
    if not all(checks.values()):
        raise ValueError(f"v2.6.3 count audit failed: {checks}")

    base_bundle = build_bounded_relation_feature_bundle(
        structure_csv=paths["structure"],
        relation_to_id=full.relation_to_id,
        allowed_reasoning_relations=REASONING14_RELATIONS,
    )
    ordinary_budget = training_budget(int(training.num_triples), cfg)

    reciprocal_training = _load_factory_with_inverse(
        paths["train"], entity_to_id=full.entity_to_id, relation_to_id=full.relation_to_id, create_inverse_triples=True
    )
    # Evaluation triples stay original. The model itself converts head prediction to artificial inverse relations.
    reciprocal_validation = _load_factory_with_inverse(
        paths["validation"], entity_to_id=full.entity_to_id, relation_to_id=full.relation_to_id, create_inverse_triples=False
    )
    reciprocal_bundle = build_reciprocal_feature_bundle(
        base_bundle=base_bundle, relation_to_id=full.relation_to_id
    )
    reciprocal_budget = reciprocal_training_budget(
        num_real_triples=int(reciprocal_training.num_triples),
        batch_size=int(cfg["training"]["batch_size"]),
        max_steps=int(cfg["training"]["max_steps"]),
    )
    if int(reciprocal_training.num_relations) != 2 * int(full.num_relations):
        raise ValueError("PyKEEN reciprocal relation count audit failed")
    if reciprocal_bundle.features.shape[0] != int(reciprocal_training.num_relations):
        raise ValueError("Reciprocal feature row count does not match internal relation count")

    return {
        "paths": paths,
        "hashes": hashes,
        "checks": checks,
        "full": full,
        "training": training,
        "validation": validation,
        "base_bundle": base_bundle,
        "ordinary_budget": ordinary_budget,
        "reciprocal_training": reciprocal_training,
        "reciprocal_validation": reciprocal_validation,
        "reciprocal_bundle": reciprocal_bundle,
        "reciprocal_budget": reciprocal_budget,
        "relation_id_to_label": {int(v): str(k) for k, v in full.relation_to_id.items()},
    }


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
        raise ValueError(variant)
    common = _common_model_kwargs(cfg)
    delta = float(cfg["model"]["frozen_delta_angle_radians"])
    if variant == "V0":
        return {
            "label": "Corrected TH-RotatE gamma=3",
            "model": build_pykeen_model_class(),
            "model_kwargs": common,
            "training": context["training"],
            "validation": context["validation"],
            "budget": context["ordinary_budget"],
            "training_loop": "sLCWA",
            "training_loop_kwargs": {},
            "reciprocal": False,
        }
    if variant == "V1":
        bundle = context["base_bundle"]
        return {
            "label": "v2.6 bounded shared relation adapter",
            "model": build_bounded_adaptive_model_class(),
            "model_kwargs": {
                **common,
                "relation_feature_tensor": bundle.features.tensor,
                "relation_reliability_tensor": bundle.reliability,
                "max_angle_deviation": delta,
            },
            "training": context["training"],
            "validation": context["validation"],
            "budget": context["ordinary_budget"],
            "training_loop": "sLCWA",
            "training_loop_kwargs": {},
            "reciprocal": False,
        }
    if variant == "V2":
        bundle = context["base_bundle"]
        return {
            "label": "v2.6.3 bounded adapter + per-relation scalar residual",
            "model": build_residual_bounded_adaptive_model_class(),
            "model_kwargs": {
                **common,
                "relation_feature_tensor": bundle.features.tensor,
                "relation_reliability_tensor": bundle.reliability,
                "max_angle_deviation": delta,
            },
            "training": context["training"],
            "validation": context["validation"],
            "budget": context["ordinary_budget"],
            "training_loop": "sLCWA",
            "training_loop_kwargs": {},
            "reciprocal": False,
        }
    reciprocal = context["reciprocal_bundle"]
    rb = context["reciprocal_budget"]
    return {
        "label": "v2.6.3 V2 + reciprocal training",
        "model": build_residual_bounded_adaptive_model_class(),
        "model_kwargs": {
            **common,
            "relation_feature_tensor": reciprocal.features,
            "relation_reliability_tensor": reciprocal.reliability,
            "max_angle_deviation": delta,
        },
        "training": context["reciprocal_training"],
        "validation": context["reciprocal_validation"],
        "budget": rb,
        "training_loop": build_exact_step_reciprocal_training_loop_class(),
        "training_loop_kwargs": {
            "automatic_memory_optimization": False,
            "exact_max_steps": int(rb["exact_optimizer_steps"]),
            "reciprocal_steps_per_full_epoch": int(rb["steps_per_full_epoch"]),
            "reciprocal_final_epoch_batches": int(rb["final_epoch_batches"]),
            "reciprocal_num_epochs": int(rb["num_epochs"]),
        },
        "reciprocal": True,
    }


def _directional_metrics(*, model, mapped_triples, truth_triples, batch_size: int, filtered: bool) -> dict[str, Any]:
    from pykeen.evaluation import RankBasedEvaluator

    output = {}
    for name, targets, target_name in (
        ("head", ("head",), "head"),
        ("tail", ("tail",), "tail"),
        ("both", ("head", "tail"), "both"),
    ):
        evaluator = RankBasedEvaluator(filtered=filtered)
        result = evaluator.evaluate(
            model=model,
            mapped_triples=mapped_triples,
            additional_filter_triples=truth_triples if filtered else None,
            batch_size=int(batch_size),
            targets=targets,
        )
        output[name] = metric_summary(result, target=target_name)
    return output


def _ordinary_relation_rows(model, relation_id_to_label: Mapping[int, str]) -> list[dict[str, Any]]:
    if not hasattr(model, "get_all_relation_fusion_state"):
        return []
    state = model.get_all_relation_fusion_state()
    weights = state["weights"]
    shift = state["angle_shift"]
    reliability = state["reliability"]
    residual = state.get("relation_residual")
    rows = []
    for rid, label in sorted(relation_id_to_label.items()):
        if label not in REASONING14_RELATIONS:
            continue
        row = {
            "relation": label,
            "relation_id": int(rid),
            "direction": "forward",
            "alpha": float(weights[rid, 0]),
            "beta": float(weights[rid, 1]),
            "angle_shift_radians": float(shift[rid]),
            "reliability_q": float(reliability[rid, 0]),
        }
        if residual is not None:
            row["relation_residual_u"] = float(residual[rid, 0])
        rows.append(row)
    return rows


def _reciprocal_relation_rows(model, relation_id_to_label: Mapping[int, str]) -> list[dict[str, Any]]:
    state = model.get_all_relation_fusion_state()
    weights = state["weights"]
    shift = state["angle_shift"]
    reliability = state["reliability"]
    residual = state["relation_residual"]
    rows = []
    for rid, label in sorted(relation_id_to_label.items()):
        if label not in REASONING14_RELATIONS:
            continue
        for direction, internal_id in (("forward", 2 * rid), ("inverse", 2 * rid + 1)):
            rows.append({
                "relation": label,
                "real_relation_id": int(rid),
                "internal_relation_id": int(internal_id),
                "direction": direction,
                "alpha": float(weights[internal_id, 0]),
                "beta": float(weights[internal_id, 1]),
                "angle_shift_radians": float(shift[internal_id]),
                "reliability_q": float(reliability[internal_id, 0]),
                "relation_residual_u": float(residual[internal_id, 0]),
            })
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
    from pykeen.pipeline import pipeline

    spec = variant_spec(cfg, context, variant)
    train_cfg = cfg["training"]
    budget = spec["budget"]
    training = spec["training"]
    validation = spec["validation"]
    device_cfg = str(train_cfg.get("device", "auto"))
    device = None if device_cfg == "auto" else device_cfg
    row: dict[str, Any] = {
        "variant": variant,
        "label": spec["label"],
        "seed": int(seed),
        "status": "RUNNING",
        "gamma": EXPECTED_GAMMA,
        "delta_angle_radians": FROZEN_DELTA if variant != "V0" else 0.0,
        "reciprocal_training": bool(spec["reciprocal"]),
        "r14_test_loaded": False,
        "test_used_for_selection": False,
        "candidate_entity_space": int(context["full"].num_entities),
        "requested_optimizer_steps": int(train_cfg["max_steps"]),
    }
    result = None
    try:
        start = time.perf_counter()
        result = pipeline(
            training=training,
            validation=validation,
            testing=validation,
            model=spec["model"],
            model_kwargs=spec["model_kwargs"],
            loss=str(train_cfg["loss"]),
            loss_kwargs={
                "margin": float(train_cfg["gamma"]),
                "adversarial_temperature": float(train_cfg["adversarial_temperature"]),
            },
            optimizer=str(train_cfg["optimizer"]),
            optimizer_kwargs={"lr": float(train_cfg["learning_rate"])},
            training_loop=spec["training_loop"],
            training_loop_kwargs=spec["training_loop_kwargs"],
            negative_sampler="bernoulli",
            negative_sampler_kwargs={
                "num_negs_per_pos": int(train_cfg["num_negatives"]),
                "filtered": bool(train_cfg["negative_sampler_filtered"]),
            },
            training_kwargs={
                "num_epochs": int(budget["num_epochs"]),
                "batch_size": int(train_cfg["batch_size"]),
                "sub_batch_size": int(train_cfg["batch_size"]),
                "use_tqdm_batch": True,
            },
            evaluator="RankBasedEvaluator",
            evaluator_kwargs={"filtered": bool(train_cfg["filtered_evaluation"])},
            evaluation_kwargs={"batch_size": int(train_cfg["validation_batch_size"])},
            use_testing_data=False,
            filter_validation_when_testing=False,
            random_seed=int(seed),
            device=device,
        )
        wall = time.perf_counter() - start
        loss_audit = runtime_loss_audit(result.model, EXPECTED_GAMMA)
        if not loss_audit["pass"]:
            raise RuntimeError(f"Runtime gamma audit failed: {loss_audit}")
        if spec["reciprocal"]:
            actual_steps = int(getattr(result.training_loop, "v263_optimizer_steps", -1))
            if actual_steps != int(train_cfg["max_steps"]):
                raise RuntimeError(f"Reciprocal exact-step audit failed: {actual_steps}")
        else:
            actual_steps = int(context["ordinary_budget"]["exact_optimizer_steps"])

        # Standard Validation remains the original 1,009 R14 triples.
        truth_factories = [context["training"], context["validation"]]
        report = build_explicit_split_report(
            model=result.model,
            evaluation_factory=context["validation"],
            truth_factories=truth_factories,
            batch_size=int(train_cfg["validation_batch_size"]),
            filtered=bool(train_cfg["filtered_evaluation"]),
            include_per_relation=True,
        )
        primary = report["all_relations_both_sides"]["metrics"]
        directional = _directional_metrics(
            model=result.model,
            mapped_triples=context["validation"].mapped_triples,
            truth_triples=[context["training"].mapped_triples, context["validation"].mapped_triples],
            batch_size=int(train_cfg["validation_batch_size"]),
            filtered=bool(train_cfg["filtered_evaluation"]),
        )
        if int(report["all_relations_both_sides"]["rank_count"]) != int(cfg["data"]["expected"]["validation_both_side_ranks"]):
            raise RuntimeError("Validation rank-count audit failed")

        if hasattr(result.model, "get_global_fusion_weights"):
            global_alpha, global_beta = result.model.get_global_fusion_weights()
        else:
            global_alpha, global_beta = result.model.get_fusion_weights()

        if variant == "V3":
            relation_rows = _reciprocal_relation_rows(result.model, context["relation_id_to_label"])
        else:
            relation_rows = _ordinary_relation_rows(result.model, context["relation_id_to_label"])
        if relation_rows:
            max_shift = max(abs(float(x["angle_shift_radians"])) for x in relation_rows)
            if max_shift > FROZEN_DELTA + 1.0e-6:
                raise RuntimeError("Observed relation shift exceeded frozen Delta=0.10")

        row.update({
            "status": "PASS",
            "actual_optimizer_steps": actual_steps,
            "runtime_loss_audit": loss_audit,
            "validation_metrics": dict(primary),
            "directional_validation_metrics": directional,
            "global_fusion_alpha": float(global_alpha),
            "global_fusion_beta": float(global_beta),
            "relation_fusion_state": relation_rows,
            "num_model_parameters": sum(p.numel() for p in result.model.parameters()),
            "num_trainable_model_parameters": sum(p.numel() for p in result.model.parameters() if p.requires_grad),
            "train_seconds": float(result.train_seconds),
            "wall_seconds": float(wall),
            "training_num_real_triples": int(training.num_triples),
            "training_internal_num_relations": int(training.num_relations),
        })
        write_json(run_dir / "validation_report.json", report)
    except Exception as exc:  # noqa: BLE001
        import traceback
        row.update({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run_result.json", row)
        if result is not None:
            del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for variant in VARIANTS:
        group = [r for r in rows if r.get("variant") == variant and r.get("status") == "PASS"]
        metrics = {}
        for metric in ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10"):
            values = [float(r["validation_metrics"][metric]) for r in group]
            metrics[metric] = {
                "mean": statistics.mean(values) if values else None,
                "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                "values": values,
            }
        directional = {}
        for direction in ("head", "tail"):
            vals = [float(r["directional_validation_metrics"][direction]["mrr"]) for r in group]
            directional[f"{direction}_mrr"] = {
                "mean": statistics.mean(vals) if vals else None,
                "std_sample": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None,
                "values": vals,
            }
        output[variant] = {
            "n": len(group),
            "seeds": [int(r["seed"]) for r in group],
            "metrics": metrics,
            "directional": directional,
        }
    return output


def success_checks(aggregated: Mapping[str, Any]) -> dict[str, Any]:
    means = {v: aggregated[v]["metrics"]["mrr"]["mean"] for v in VARIANTS}
    if any(x is None for x in means.values()):
        return {}
    v0, v1, v2, v3 = (float(means[v]) for v in VARIANTS)
    return {
        "V1_minus_V0": v1 - v0,
        "V2_minus_V1": v2 - v1,
        "V3_minus_V2": v3 - v2,
        "V2_relation_residual_has_independent_gain": v2 > v1,
        "V3_reciprocal_has_incremental_gain": v3 > v2,
        "best_variant": max(VARIANTS, key=lambda v: float(means[v])),
        "no_test_used": True,
    }
