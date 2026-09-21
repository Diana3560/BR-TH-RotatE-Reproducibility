from __future__ import annotations

"""Run a validation-only pre/post-training branch-scale diagnostic for final D2.

The script deliberately does not read Test triples and does not change any model
hyperparameter. It retrains the already-frozen D2 configuration for seeds 42-46,
then reports raw TransH/RotatE distances and their weighted contributions on the
fixed Validation split. Because the original result package did not retain model
checkpoints, retraining is necessary to recover trained embeddings.
"""

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from throtate_repro.distance_scale_diagnostic import (  # noqa: E402
    aggregate_seed_summaries,
    component_summary,
    translation_normal_alignment,
)
from throtate_repro.exact_step_v266 import build_exact_step_training_loop_class  # noqa: E402
from throtate_repro.gamma3_canonical_v251 import runtime_loss_audit  # noqa: E402
from throtate_repro.multidataset_experiment import (  # noqa: E402
    _build_context,
    _model_spec,
    dataset_output_dir,
    experiment_identity,
    load_multidataset_config,
    read_json,
)
from throtate_repro.v266_runtime import AuditedBernoulliNegativeSampler  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_output(dataset_key: str) -> Path:
    latest_path = PROJECT_ROOT / "results" / "multidataset_comparison" / dataset_key / "LATEST.json"
    latest = read_json(latest_path)
    value = latest.get("output_directory")
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid LATEST pointer: {latest_path}")
    return PROJECT_ROOT / Path(value.replace("\\", "/"))


def _device_from_config(cfg) -> torch.device:
    requested = str(cfg["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _score_components(model, mapped_triples: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    if not hasattr(model, "score_components_hrt"):
        raise RuntimeError("D2 model does not expose score_components_hrt")
    chunks: dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(mapped_triples.shape[0]), int(batch_size)):
            batch = mapped_triples[start : start + int(batch_size)].to(device)
            result = model.score_components_hrt(batch)
            for key in ("transh_distance", "rotate_distance", "fusion_alpha", "fusion_beta"):
                chunks.setdefault(key, []).append(result[key].detach().cpu().reshape(-1))
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def _alignment(model) -> dict[str, Any]:
    d_r = model.relation_representations[0](indices=None)
    w_r = model.relation_representations[1](indices=None)
    return translation_normal_alignment(d_r, w_r)


def _diagnose(model, mapped_triples: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, Any]:
    comp = _score_components(model, mapped_triples, batch_size, device)
    return {
        "components": component_summary(
            transh_distance=comp["transh_distance"],
            rotate_distance=comp["rotate_distance"],
            fusion_alpha=comp["fusion_alpha"],
            fusion_beta=comp["fusion_beta"],
        ),
        "translation_normal_alignment": _alignment(model),
    }


def _instantiate_initial_model(spec, training_factory, seed: int, device: torch.device):
    from pykeen.utils import set_random_seed

    set_random_seed(int(seed))
    model = spec["model"](triples_factory=training_factory, **spec["model_kwargs"])
    return model.to(device)


def _train_final_model(*, cfg, context, spec, seed: int, gamma: float, device: torch.device):
    from pykeen.pipeline import pipeline

    reciprocal = bool(spec["reciprocal"])
    training_factory = context["reciprocal_training"] if reciprocal else context["training"]
    validation_factory = context["reciprocal_validation"] if reciprocal else context["validation"]
    train_cfg = cfg["training"]
    budget = context["budgets"]["reciprocal" if reciprocal else "ordinary"]

    expected_positive = int(train_cfg["max_steps"]) * int(train_cfg["batch_size"])
    expected_negative = expected_positive * int(train_cfg["num_negatives"])
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
        device=str(device),
    )
    wall_seconds = time.perf_counter() - started
    steps = int(getattr(result.training_loop, "v266_optimizer_steps", -1))
    if steps != int(train_cfg["max_steps"]):
        raise RuntimeError(f"Exact-step audit failed: {steps}")
    sampling = AuditedBernoulliNegativeSampler.audit_snapshot(
        num_negatives_per_positive=int(train_cfg["num_negatives"])
    )
    if int(sampling["actual_positive_instances"]) != expected_positive:
        raise RuntimeError(f"Positive exposure mismatch: {sampling}")
    if int(sampling["requested_negatives"]) != expected_negative:
        raise RuntimeError(f"Negative exposure mismatch: {sampling}")
    loss_audit = runtime_loss_audit(result.model, float(gamma))
    if not loss_audit["pass"]:
        raise RuntimeError(f"NSSA margin mismatch: {loss_audit}")
    return result, wall_seconds, sampling, loss_audit


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "DISTANCE_SCALE_DIAGNOSTIC.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_rows = []
    for row in rows:
        for stage in ("before_training", "after_training"):
            c = row[stage]["components"]
            a = row[stage]["translation_normal_alignment"]["absolute_cosine"]
            csv_rows.append({
                "seed": row["seed"],
                "stage": stage,
                "transh_mean": c["raw_transh_distance"]["mean"],
                "rotate_mean": c["raw_rotate_distance"]["mean"],
                "raw_mean_ratio_h_over_r": c["raw_mean_ratio_transh_over_rotate"],
                "weighted_transh_mean": c["weighted_transh_contribution"]["mean"],
                "weighted_rotate_mean": c["weighted_rotate_contribution"]["mean"],
                "weighted_mean_ratio_h_over_r": c["weighted_mean_ratio_transh_over_rotate"],
                "transh_fused_share": c["mean_transh_share_of_fused_distance"],
                "rotate_fused_share": c["mean_rotate_share_of_fused_distance"],
                "abs_cos_d_w_mean": a["mean"],
                "abs_cos_d_w_max": a["max"],
            })
    with (output_dir / "DISTANCE_SCALE_DIAGNOSTIC.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# TransH/RotatE distance-scale diagnostic",
        "",
        "This diagnostic uses the fixed Validation split only; Test is not constructed or read.",
        "The model/hyperparameters are already frozen. Five seeds are retrained only because the original result package did not retain checkpoints.",
        "",
        "| Stage | TransH mean distance | RotatE mean distance | Raw H/R mean ratio | Weighted H/R mean ratio | Mean TransH share | Mean |cos(d,w)| |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, label in (("before_training", "Before training"), ("after_training", "After training")):
        a = summary["aggregate_across_seeds"][stage]
        lines.append(
            "| {label} | {dh:.6f} | {dr:.6f} | {rr:.6f} | {wr:.6f} | {share:.6f} | {cos:.6f} |".format(
                label=label,
                dh=a["raw_transh_distance_mean"]["mean_across_seeds"],
                dr=a["raw_rotate_distance_mean"]["mean_across_seeds"],
                rr=a["raw_mean_ratio_transh_over_rotate"]["mean_across_seeds"],
                wr=a["weighted_mean_ratio_transh_over_rotate"]["mean_across_seeds"],
                share=a["mean_transh_share_of_fused_distance"]["mean_across_seeds"],
                cos=a["translation_normal_abs_cosine_mean"]["mean_across_seeds"],
            )
        )
    lines.extend([
        "",
        "Interpretation guard: alpha^2 + beta^2 = 1 constrains mixture coefficients but does not by itself equalize raw branch-distance scales. The table therefore reports both raw distances and weighted contributions.",
        "",
        "TransH implementation note: w_r is L2-normalized, but d_r is not explicitly projected onto the relation hyperplane; w_r^T d_r = 0 is not enforced. The |cos(d,w)| column is a diagnostic measurement, not a constraint.",
    ])
    (output_dir / "DISTANCE_SCALE_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ownkg", "paper4"), default="ownkg")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    cfg = load_multidataset_config(PROJECT_ROOT)
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    official_dir = _latest_output(args.dataset)
    frozen_path = official_dir / "PROTOCOL_FROZEN.json"
    frozen = read_json(frozen_path)
    if frozen.get("status") != "FROZEN":
        raise RuntimeError(f"Protocol is not frozen: {frozen_path}")
    formal_seeds = tuple(int(x) for x in frozen["formal_seeds"])
    if tuple(seeds) != formal_seeds:
        print("[NOTE] Running a non-full seed subset; use 42,43,44,45,46 for the paper table.")

    identity = experiment_identity(PROJECT_ROOT, cfg, args.dataset, audit=None)
    expected_fp = str(frozen["experiment_fingerprint"])
    if identity["fingerprint"] != expected_fp:
        raise RuntimeError("Current config/implementation fingerprint differs from the frozen experiment")
    output_dir = dataset_output_dir(PROJECT_ROOT, cfg, args.dataset, identity["fingerprint"])
    context = _build_context(PROJECT_ROOT, cfg, args.dataset, output_dir, include_test=False)
    if context.get("test_factory_constructed"):
        raise RuntimeError("Distance diagnostic must remain Validation-only")

    delta = float(frozen["selected_d2_delta_radians"])
    gamma = float(frozen["fixed_d0_d1_d2_gamma"])
    display_name = str(frozen.get("display_name", args.dataset))
    provenance_label = "final_unified_split_multidataset"

    spec = _model_spec(cfg, context, "D2", delta)
    if not spec["reciprocal"]:
        raise RuntimeError("D2 diagnostic expected reciprocal training")
    training_factory = context["reciprocal_training"]
    validation_factory = context["reciprocal_validation"]
    device = _device_from_config(cfg)
    batch_size = int(cfg["training"]["evaluation_batch_size"])

    diag_dir = PROJECT_ROOT / "results" / "generated" / "distance_scale_diagnostic" / args.dataset
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[distance-scale] dataset={args.dataset} seed={seed} device={device}")
        initial = _instantiate_initial_model(spec, training_factory, seed, device)
        before = _diagnose(initial, validation_factory.mapped_triples, batch_size, device)
        del initial
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trained, wall_seconds, sampling, loss_audit = _train_final_model(
            cfg=cfg, context=context, spec=spec, seed=seed, gamma=gamma, device=device
        )
        try:
            after = _diagnose(trained.model, validation_factory.mapped_triples, batch_size, device)
            row = {
                "status": "PASS",
                "dataset_key": args.dataset,
                "seed": seed,
                "model": "D2",
                "gamma": gamma,
                "delta_angle_radians": delta,
                "evaluation_split": "validation",
                "validation_sha256": context["audit"]["sha256"]["validation"],
                "test_constructed_or_read": False,
                "before_training": before,
                "after_training": after,
                "actual_optimizer_steps": int(getattr(trained.training_loop, "v266_optimizer_steps", -1)),
                "sampling_exposure_audit": sampling,
                "runtime_loss_audit": loss_audit,
                "train_seconds": float(trained.train_seconds),
                "wall_seconds": float(wall_seconds),
            }
            rows.append(row)
            (diag_dir / f"D2_seed{seed}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            del trained
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "status": "COMPLETE",
        "phase": "validation_only_transh_rotate_distance_scale_diagnostic",
        "dataset_key": args.dataset,
        "display_name": display_name,
        "provenance_label": provenance_label,
        "model": "D2",
        "formal_or_requested_seeds": list(seeds),
        "gamma": gamma,
        "delta_angle_radians": delta,
        "evaluation_split": "validation",
        "validation_sha256": context["audit"]["sha256"]["validation"],
        "test_constructed_or_read": False,
        "transh_power_norm": bool(cfg["model"]["transh_power_norm"]),
        "distance_definition": "raw unsquared L2 for both branches" if not bool(cfg["model"]["transh_power_norm"]) else "see config",
        "branch_normalization_applied": False,
        "orthogonality_constraint_wT_d_zero_enforced": False,
        "aggregate_across_seeds": aggregate_seed_summaries(rows),
        "per_seed_files": [f"D2_seed{seed}.json" for seed in seeds],
        "runtime_environment": {
            "python": platform.python_version(),
            "pykeen": metadata.version("pykeen"),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
        },
        "provenance": {
            "protocol_frozen": str(frozen_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "protocol_frozen_sha256": _sha256(frozen_path),
            "script_sha256": _sha256(Path(__file__)),
            "diagnostic_module_sha256": _sha256(PROJECT_ROOT / "src/throtate_repro/distance_scale_diagnostic.py"),
        },
        "interpretation_guard": (
            "This is a descriptive scale audit on the frozen Validation split. It is not used for model selection, "
            "does not alter the scoring function, and should not be presented as an additional Test-set significance result."
        ),
    }
    _write_outputs(diag_dir, rows, summary)
    print(json.dumps({
        "status": summary["status"],
        "dataset": summary["display_name"],
        "output_dir": str(diag_dir.relative_to(PROJECT_ROOT)),
        "aggregate_across_seeds": summary["aggregate_across_seeds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
