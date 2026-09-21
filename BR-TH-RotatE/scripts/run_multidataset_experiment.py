from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_NAMES = ("TransH", "RotatE", "D0", "D1", "A0", "D2")


def _compact(stage: str, dataset: str, result: dict) -> dict:
    output = {
        "stage": stage,
        "dataset": dataset,
        "status": result.get("status"),
    }
    for key in (
        "phase",
        "experiment_fingerprint",
        "selected_baseline_gamma",
        "selected_a0_delta_radians",
        "selected_d2_delta_radians",
        "completed_training_runs",
        "runs_completed_this_call",
        "total_passed_runs",
        "expected_total_runs",
        "remaining_runs",
    ):
        if key in result:
            output[key] = result[key]
    if "a0_selection" in result:
        output["selected_a0_delta_radians"] = result["a0_selection"]["selected_delta"]
    if "d2_selection" in result:
        output["selected_d2_delta_radians"] = result["d2_selection"]["selected_delta"]
    if "baseline_selection" in result:
        output["selected_baseline_gamma"] = {
            model: row["selected_gamma"] for model, row in result["baseline_selection"].items()
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated, resumable TH-RotatE experiments on CRH-L4MKG and CROEFKG."
    )
    parser.add_argument("--dataset", choices=("ownkg", "paper4", "both"), default="both")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "screen", "freeze", "test", "summarize", "compare", "all"),
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_NAMES,
        help="For stage=test, run only this model. Repeat for several models.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        choices=(42, 43, 44, 45, 46),
        help="For stage=test, run only this seed. Repeat for several seeds.",
    )
    args = parser.parse_args()

    try:
        from throtate_repro.multidataset_experiment import (
            freeze_protocol,
            load_multidataset_config,
            preflight,
            run_all,
            run_fixed_test,
            screen_validation,
            summarize_dataset,
            summarize_two_datasets,
        )
    except ModuleNotFoundError as error:
        if error.name in {"torch", "pykeen"}:
            parser.error(
                "Training dependencies are missing. Run: python -m pip install -r requirements.txt"
            )
        raise
    cfg = load_multidataset_config(PROJECT_ROOT)

    if args.stage == "compare":
        result = summarize_two_datasets(PROJECT_ROOT, cfg)
        print(json.dumps(_compact("compare", "both", result), ensure_ascii=False, indent=2))
        print("\n[COMPLETE] Two-dataset comparison outputs are ready.")
        return 0

    datasets = tuple(cfg["datasets"]) if args.dataset == "both" else (args.dataset,)
    results: dict[str, dict] = {}
    for dataset_key in datasets:
        print(f"\n=== dataset={dataset_key} stage={args.stage} ===", flush=True)
        if args.stage == "preflight":
            result = preflight(PROJECT_ROOT, cfg, dataset_key)
        elif args.stage == "screen":
            result = screen_validation(PROJECT_ROOT, cfg, dataset_key)
        elif args.stage == "freeze":
            result = freeze_protocol(PROJECT_ROOT, cfg, dataset_key)
        elif args.stage == "test":
            result = run_fixed_test(
                PROJECT_ROOT,
                cfg,
                dataset_key,
                models=args.model,
                seeds=args.seed,
            )
        elif args.stage == "summarize":
            result = summarize_dataset(PROJECT_ROOT, cfg, dataset_key)
        elif args.stage == "all":
            result = run_all(PROJECT_ROOT, cfg, dataset_key)
        else:  # pragma: no cover - argparse guards this branch
            raise AssertionError(args.stage)
        results[dataset_key] = result
        print(json.dumps(_compact(args.stage, dataset_key, result), ensure_ascii=False, indent=2))

    if args.stage in {"all", "summarize"} and set(datasets) == set(cfg["datasets"]):
        combined = summarize_two_datasets(PROJECT_ROOT, cfg)
        print(json.dumps(_compact("compare", "both", combined), ensure_ascii=False, indent=2))
        print("\n[COMPLETE] Combined two-dataset table is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
