from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from throtate_repro.evaluation import metric_summary as extract_metric_summary
from throtate_repro.model import build_pykeen_model_class
from throtate_repro.role_control_v18 import sha256_file


METRICS = ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10")
HITS_METRICS = ("hits_at_1", "hits_at_3", "hits_at_10")
EXPECTED_TOKEN = "OWNKGC_V251_CANONICAL_COMMON_GAMMA3_R14_CORRECTION"
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_GAMMA = 3.0
TARGET_MODEL = "TH-RotatE"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_factory(path: Path, *, entity_to_id=None, relation_to_id=None):
    from pykeen.triples import TriplesFactory

    kwargs = dict(create_inverse_triples=False, load_triples_kwargs=dict(delimiter="\t"))
    if entity_to_id is not None:
        kwargs.update(entity_to_id=entity_to_id, relation_to_id=relation_to_id, compact_id=False)
    return TriplesFactory.from_path(path, **kwargs)


def validate_protocol(cfg: Mapping[str, Any]) -> None:
    protocol = cfg["protocol"]
    training = cfg["training"]
    model = cfg["model"]
    if protocol["token"] != EXPECTED_TOKEN:
        raise ValueError("v2.5.1 protocol token mismatch")
    if tuple(int(seed) for seed in protocol["model_seeds"]) != EXPECTED_SEEDS:
        raise ValueError("Model seeds must remain exactly 42/43/44")
    if float(protocol["common_gamma"]) != EXPECTED_GAMMA:
        raise ValueError("The canonical correction requires common gamma=3")
    if float(training["gamma"]) != EXPECTED_GAMMA:
        raise ValueError("Runtime training gamma must remain 3")
    if str(model["family"]) != TARGET_MODEL or str(protocol["target_model"]) != TARGET_MODEL:
        raise ValueError("The correction runner may train TH-RotatE only")
    if protocol.get("gamma_selection_performed") is not False:
        raise ValueError("No gamma selection is permitted")
    if protocol.get("gamma_search_performed") is not False:
        raise ValueError("No gamma sweep is permitted")
    if protocol.get("parameter_selection_from_validation") is not False:
        raise ValueError("Validation metrics must not select gamma")
    if protocol.get("parameter_selection_from_test") is not False:
        raise ValueError("Test metrics must not select parameters")
    if protocol.get("post_correction_tuning_permitted") is not False:
        raise ValueError("Post-correction tuning must remain forbidden")
    if abs(float(protocol["sa_rf_rotate_weight_remains_frozen"]) - 0.40) > 1.0e-12:
        raise ValueError("SA-RF lambda must remain 0.40")
    if int(training["max_steps"]) != 2000:
        raise ValueError("Training budget must remain 2000 optimizer steps")


def validation_paths(project_root: Path, cfg: Mapping[str, Any]) -> dict[str, Path]:
    split_dir = project_root / cfg["data"]["split_dir"]
    return {
        "full17": project_root / cfg["data"]["full17_file"],
        "train": split_dir / "train.tsv",
        "validation": split_dir / "valid.tsv",
    }


def test_paths(project_root: Path, cfg: Mapping[str, Any]) -> dict[str, Path]:
    paths = validation_paths(project_root, cfg)
    split_dir = project_root / cfg["data"]["split_dir"]
    paths["test"] = split_dir / "test.tsv"
    return paths


def validate_hashes(paths: Mapping[str, Path], cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected = cfg["data"]["sha256"]
    report: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        expected_key = "validation" if key == "validation" else key
        actual = sha256_file(path)
        target = str(expected[expected_key])
        report[key] = {
            "path": str(path),
            "expected_sha256": target,
            "actual_sha256": actual,
            "match": actual == target,
        }
    return report


def relation_labels(path: Path) -> set[str]:
    labels: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number} does not contain exactly three TSV columns")
            labels.add(fields[1])
    return labels


def training_budget(training_num_triples: int, cfg: Mapping[str, Any]) -> dict[str, int]:
    batch_size = int(cfg["training"]["batch_size"])
    max_steps = int(cfg["training"]["max_steps"])
    steps_per_epoch = math.ceil(training_num_triples / batch_size)
    if max_steps % steps_per_epoch:
        raise ValueError(
            f"Exact optimizer-step budget broken: {max_steps=} is not divisible by {steps_per_epoch=}"
        )
    return {
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "num_epochs": max_steps // steps_per_epoch,
        "exact_optimizer_steps": max_steps,
    }


def throtate_model_spec(cfg: Mapping[str, Any]):
    model_cfg = cfg["model"]
    return build_pykeen_model_class(), {
        "embedding_dim": int(model_cfg["embedding_dim"]),
        "transh_norm": int(model_cfg["transh_norm"]),
        "transh_power_norm": bool(model_cfg["transh_power_norm"]),
        "rotate_norm": int(model_cfg["rotate_norm"]),
    }


def loss_kwargs(cfg: Mapping[str, Any]) -> dict[str, float]:
    return {
        "margin": float(cfg["training"]["gamma"]),
        "adversarial_temperature": float(cfg["training"]["adversarial_temperature"]),
    }


def runtime_loss_audit(model, expected_gamma: float = EXPECTED_GAMMA) -> dict[str, Any]:
    loss = getattr(model, "loss", None)
    loss_name = type(loss).__name__ if loss is not None else None
    raw_margin = getattr(loss, "margin", None)
    try:
        actual_margin = float(raw_margin.detach().cpu().item())
    except AttributeError:
        actual_margin = float(raw_margin) if raw_margin is not None else None
    passed = loss_name == "NSSALoss" and actual_margin is not None
    passed = bool(passed and abs(actual_margin - expected_gamma) <= 1.0e-12)
    return {
        "loss_class": loss_name,
        "expected_margin_gamma": float(expected_gamma),
        "actual_runtime_margin_gamma": actual_margin,
        "pass": passed,
    }


def protocol_fingerprint(cfg: Mapping[str, Any]) -> str:
    payload = {
        "token": cfg["protocol"]["token"],
        "model_seeds": [int(x) for x in cfg["protocol"]["model_seeds"]],
        "common_gamma": float(cfg["protocol"]["common_gamma"]),
        "target_model": cfg["protocol"]["target_model"],
        "data_sha256": dict(cfg["data"]["sha256"]),
        "model": dict(cfg["model"]),
        "training": dict(cfg["training"]),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot summarize an empty value sequence")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def compact_report(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scopes = {
        name: {
            "num_triples": int(report[name]["num_triples"]),
            "rank_count": int(report[name]["rank_count"]),
            "metrics": dict(report[name]["metrics"]),
            "rank_count_audit": dict(report[name]["rank_count_audit"]),
        }
        for name in ("all_relations_both_sides", "all_relations_tail_only")
    }
    per_relation = {
        relation: {
            "num_triples": int(row["num_triples"]),
            "rank_count": int(row["rank_count"]),
            "metrics": dict(row["metrics"]),
        }
        for relation, row in (report.get("per_relation_tail_only") or {}).items()
    }
    return scopes, per_relation


def _hits_rank_count_audit(metrics: Mapping[str, float | None], rank_count: int) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    for key in HITS_METRICS:
        value = metrics.get(key)
        if value is None:
            details[key] = {"pass": False, "reason": "missing"}
            passed = False
            continue
        raw_count = float(value) * rank_count
        nearest = round(raw_count)
        error = abs(raw_count - nearest)
        item_pass = error <= 1.0e-5
        details[key] = {
            "metric_value": float(value),
            "implied_integer_hits": int(nearest),
            "absolute_integer_error": error,
            "pass": item_pass,
        }
        passed = passed and item_pass
    return {"expected_rank_count": rank_count, "metrics": details, "pass": passed}


def _evaluate_target(
    *,
    model,
    mapped_triples,
    truth_triples: list,
    batch_size: int,
    filtered: bool,
    targets: tuple[str, ...],
) -> tuple[Any, dict[str, float | None]]:
    from pykeen.evaluation import RankBasedEvaluator

    evaluator = RankBasedEvaluator(filtered=filtered)
    results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        additional_filter_triples=truth_triples if filtered else None,
        batch_size=batch_size,
        targets=targets,
    )
    target = "both" if len(targets) == 2 else targets[0]
    return results, extract_metric_summary(results, target=target)


def build_explicit_split_report(
    *,
    model,
    evaluation_factory,
    truth_factories: list,
    batch_size: int,
    filtered: bool,
    include_per_relation: bool = True,
) -> dict[str, Any]:
    mapped = evaluation_factory.mapped_triples
    truth_triples = [factory.mapped_triples for factory in truth_factories]
    num_triples = int(evaluation_factory.num_triples)

    both_results, both_metrics = _evaluate_target(
        model=model,
        mapped_triples=mapped,
        truth_triples=truth_triples,
        batch_size=batch_size,
        filtered=filtered,
        targets=("head", "tail"),
    )
    tail_results, tail_metrics = _evaluate_target(
        model=model,
        mapped_triples=mapped,
        truth_triples=truth_triples,
        batch_size=batch_size,
        filtered=filtered,
        targets=("tail",),
    )
    report: dict[str, Any] = {
        "metric_protocol": "filtered" if filtered else "raw",
        "all_relations_both_sides": {
            "num_triples": num_triples,
            "rank_count": 2 * num_triples,
            "metrics": both_metrics,
            "rank_count_audit": _hits_rank_count_audit(both_metrics, 2 * num_triples),
            "all_metrics": dict(both_results.to_flat_dict()),
        },
        "all_relations_tail_only": {
            "num_triples": num_triples,
            "rank_count": num_triples,
            "metrics": tail_metrics,
            "rank_count_audit": _hits_rank_count_audit(tail_metrics, num_triples),
            "all_metrics": dict(tail_results.to_flat_dict()),
        },
    }
    if include_per_relation:
        import torch

        per_relation = {}
        for label in sorted(evaluation_factory.relation_to_id):
            relation_id = int(evaluation_factory.relation_to_id[label])
            relation_triples = mapped[mapped[:, 1] == relation_id]
            if relation_triples.shape[0] == 0:
                continue
            _, metrics = _evaluate_target(
                model=model,
                mapped_triples=relation_triples,
                truth_triples=truth_triples,
                batch_size=batch_size,
                filtered=filtered,
                targets=("tail",),
            )
            n = int(relation_triples.shape[0])
            per_relation[label] = {
                "num_triples": n,
                "rank_count": n,
                "metrics": metrics,
                "rank_count_audit": _hits_rank_count_audit(metrics, n),
            }
            if not torch.isfinite(relation_triples).all():
                raise ValueError(f"Non-finite mapped triple detected for relation {label}")
        report["per_relation_tail_only"] = per_relation
    return report


def aggregate_runs(rows: list[dict[str, Any]], *, split_label: str) -> dict[str, Any]:
    passed = [row for row in rows if row.get("status") == "PASS"]
    if len(passed) != len(EXPECTED_SEEDS):
        return {}
    expected_seed_set = set(EXPECTED_SEEDS)
    if {int(row["model_seed"]) for row in passed} != expected_seed_set:
        raise ValueError("The completed runs do not contain exactly seeds 42/43/44")

    scopes: dict[str, Any] = {}
    for scope in ("all_relations_both_sides", "all_relations_tail_only"):
        scope_rows = [row[f"{split_label}_scopes"][scope] for row in passed]
        scopes[scope] = {
            "model": TARGET_MODEL,
            "gamma": EXPECTED_GAMMA,
            "num_passed_runs": len(passed),
            "model_seeds": list(EXPECTED_SEEDS),
            f"num_{split_label}_triples": int(scope_rows[0]["num_triples"]),
            "rank_count_per_run": int(scope_rows[0]["rank_count"]),
            f"{split_label}_metrics": {
                key: numeric_summary(float(item["metrics"][key]) for item in scope_rows)
                for key in METRICS
            },
        }
    return scopes


def aggregate_per_relation(rows: list[dict[str, Any]], *, split_label: str) -> dict[str, Any]:
    passed = [row for row in rows if row.get("status") == "PASS"]
    if len(passed) != len(EXPECTED_SEEDS):
        return {}
    relations = sorted(set().union(*(row.get("per_relation_tail_only", {}) for row in passed)))
    result = {}
    for relation in relations:
        values = [row["per_relation_tail_only"][relation] for row in passed]
        result[relation] = {
            "model": TARGET_MODEL,
            "gamma": EXPECTED_GAMMA,
            f"num_{split_label}_triples": int(values[0]["num_triples"]),
            f"{split_label}_metrics": {
                key: numeric_summary(float(item["metrics"][key]) for item in values)
                for key in METRICS
            },
        }
    return result


def seed_metric_rows(rows: list[dict[str, Any]], *, split_label: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metrics = row.get(f"{split_label}_metrics") or {}
        output.append({
            "model": row.get("family"),
            "gamma": row.get("gamma"),
            "seed": row.get("model_seed"),
            "status": row.get("status"),
            f"{split_label}_mrr": metrics.get("mrr"),
            f"{split_label}_mr": metrics.get("mr"),
            f"{split_label}_hits1": metrics.get("hits_at_1"),
            f"{split_label}_hits3": metrics.get("hits_at_3"),
            f"{split_label}_hits10": metrics.get("hits_at_10"),
            "runtime_loss_margin": (row.get("runtime_loss_audit") or {}).get("actual_runtime_margin_gamma"),
            "protocol_fingerprint": row.get("protocol_fingerprint"),
        })
    return output


def canonical_validation_table(
    *,
    project_root: Path,
    cfg: Mapping[str, Any],
    new_aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    v18 = json.loads((project_root / cfg["references"]["v18_validation_summary"]).read_text(encoding="utf-8"))
    v22 = json.loads((project_root / cfg["references"]["v22_sa_rf_validation_summary"]).read_text(encoding="utf-8"))
    v23 = json.loads((project_root / cfg["references"]["v23_sa_rf_freeze_summary"]).read_text(encoding="utf-8"))
    if v18.get("status") != "COMPLETE" or v22.get("status") != "COMPLETE" or v23.get("status") != "COMPLETE":
        raise ValueError("A frozen Validation reference is incomplete")
    if v23.get("final_selection", {}).get("selected_rotate_weight") != 0.4:
        raise ValueError("The frozen SA-RF lambda reference is not 0.40")

    v18_scope = v18["aggregated_validation_metrics"]["all_relations_both_sides"]
    sa_scope = v22["aggregated_validation_metrics"]["all_relations_both_sides"]["sa_rf_fixed_040"]
    new_scope = new_aggregate["all_relations_both_sides"]

    def row(model: str, gamma: float, metrics: Mapping[str, Any], source: str, note: str) -> dict[str, Any]:
        return {
            "model": model,
            "gamma": gamma,
            "seed_count": int(metrics["mrr"]["n"]),
            "validation_mrr_mean": float(metrics["mrr"]["mean"]),
            "validation_mrr_std_sample": float(metrics["mrr"]["std_sample"]),
            "validation_mr_mean": float(metrics["mr"]["mean"]),
            "validation_hits1_mean": float(metrics["hits_at_1"]["mean"]),
            "validation_hits3_mean": float(metrics["hits_at_3"]["mean"]),
            "validation_hits10_mean": float(metrics["hits_at_10"]["mean"]),
            "source": source,
            "note": note,
        }

    rows = [
        row(
            "TransH",
            EXPECTED_GAMMA,
            v18_scope["TransH"]["validation_metrics"],
            cfg["references"]["v18_validation_summary"],
            "Frozen R14 gamma=3 reference reused unchanged",
        ),
        row(
            "RotatE",
            EXPECTED_GAMMA,
            v18_scope["RotatE"]["validation_metrics"],
            cfg["references"]["v18_validation_summary"],
            "Frozen R14 gamma=3 reference reused unchanged",
        ),
        row(
            TARGET_MODEL,
            EXPECTED_GAMMA,
            new_scope["validation_metrics"],
            "new_v2.5.1_three_seed_validation_run",
            "Canonical correction; gamma fixed before metrics were observed",
        ),
        row(
            "SA-RF",
            EXPECTED_GAMMA,
            sa_scope["validation_metrics"],
            cfg["references"]["v22_sa_rf_validation_summary"],
            "Frozen lambda=0.40 R14 gamma=3 reference reused unchanged",
        ),
    ]
    return sorted(rows, key=lambda item: float(item["validation_mrr_mean"]), reverse=True)
