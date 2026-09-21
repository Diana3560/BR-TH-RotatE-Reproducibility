from __future__ import annotations

from typing import Any, Mapping


METRIC_SUFFIXES = {
    "mrr": ("inverse_harmonic_mean_rank", "mean_reciprocal_rank"),
    "mr": ("arithmetic_mean_rank", "mean_rank"),
    "hits_at_1": ("hits_at_1", "hits@1"),
    "hits_at_3": ("hits_at_3", "hits@3"),
    "hits_at_10": ("hits_at_10", "hits@10"),
}


def metric_summary(metric_results, *, target: str = "both") -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for output_name, suffixes in METRIC_SUFFIXES.items():
        value = None
        candidates = [f"{target}.realistic.{suffix}" for suffix in suffixes]
        if target == "both":
            candidates.extend(suffixes)
        for candidate in candidates:
            try:
                value = float(metric_results.get_metric(candidate))
                break
            except Exception:
                continue
        summary[output_name] = value
    return summary


def _relation_mask(mapped_triples, relation_ids: list[int]):
    import torch

    mask = torch.zeros(mapped_triples.shape[0], dtype=torch.bool, device=mapped_triples.device)
    for relation_id in relation_ids:
        mask |= mapped_triples[:, 1] == relation_id
    return mask


def _evaluate_tail(
    *,
    model,
    mapped_triples,
    all_true_triples,
    batch_size: int,
    filtered: bool,
) -> dict[str, Any]:
    from pykeen.evaluation import RankBasedEvaluator

    evaluator = RankBasedEvaluator(filtered=filtered)
    metrics = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        batch_size=batch_size,
        additional_filter_triples=all_true_triples if filtered else None,
        targets=("tail",),
    )
    return {
        "num_triples": int(mapped_triples.shape[0]),
        "metrics": metric_summary(metrics, target="tail"),
        "all_metrics": dict(metrics.to_flat_dict()),
    }


def build_evaluation_report(
    *,
    model,
    pipeline_metric_results,
    training,
    validation,
    testing,
    evaluation_config: Mapping[str, Any] | None,
    batch_size: int,
    filtered: bool,
) -> dict[str, Any]:
    """Report standard all/both metrics plus optional paper-aligned tail-only scopes."""
    evaluation_config = dict(evaluation_config or {})
    report: dict[str, Any] = {
        "metric_protocol": "filtered" if filtered else "raw",
        "all_relations_both_sides": {
            "num_triples": int(testing.num_triples),
            "metrics": metric_summary(pipeline_metric_results, target="both"),
            "all_metrics": dict(pipeline_metric_results.to_flat_dict()),
        },
    }
    all_true_triples = [
        training.mapped_triples,
        validation.mapped_triples,
        testing.mapped_triples,
    ]

    if evaluation_config.get("report_all_tail", False):
        report["all_relations_tail_only"] = _evaluate_tail(
            model=model,
            mapped_triples=testing.mapped_triples,
            all_true_triples=all_true_triples,
            batch_size=batch_size,
            filtered=filtered,
        )

    relation_labels = list(evaluation_config.get("diagnostic_relations") or [])
    missing_relations = [label for label in relation_labels if label not in testing.relation_to_id]
    present_relations = [label for label in relation_labels if label in testing.relation_to_id]
    report["diagnostic_relation_request"] = {
        "requested": relation_labels,
        "present": present_relations,
        "missing": missing_relations,
    }

    if evaluation_config.get("report_diagnostic_tail", False) and present_relations:
        relation_ids = [testing.relation_to_id[label] for label in present_relations]
        mask = _relation_mask(testing.mapped_triples, relation_ids)
        diagnostic_triples = testing.mapped_triples[mask]
        if diagnostic_triples.shape[0] == 0:
            raise ValueError("The test split contains no triples for the requested diagnostic relations")
        report["diagnostic_relations_tail_only"] = _evaluate_tail(
            model=model,
            mapped_triples=diagnostic_triples,
            all_true_triples=all_true_triples,
            batch_size=batch_size,
            filtered=filtered,
        )

    if evaluation_config.get("per_relation", False):
        per_relation: dict[str, Any] = {}
        for relation_label in sorted(testing.relation_to_id):
            relation_id = testing.relation_to_id[relation_label]
            mask = _relation_mask(testing.mapped_triples, [relation_id])
            relation_triples = testing.mapped_triples[mask]
            if relation_triples.shape[0] == 0:
                continue
            per_relation[relation_label] = _evaluate_tail(
                model=model,
                mapped_triples=relation_triples,
                all_true_triples=all_true_triples,
                batch_size=batch_size,
                filtered=filtered,
            )
        report["per_relation_tail_only"] = per_relation
    return report
