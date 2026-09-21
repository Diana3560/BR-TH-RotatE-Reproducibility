from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from supplementary.protocol import digest, freeze, read_json, settings, sha, source_hashes, write_json
from supplementary.statistics import bootstrap_differences, paired, summary, table


CONTROL_NAME = "D2_PerRelation"
REFERENCE_NAME = "D2"
CONTROL_FORMULA = "theta_(r,d) = theta_global + Delta * tanh(delta_(r,d))"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct per-relation bounded-angle control for BR-TH-RotatE. "
            "Formal defaults exactly match the paper D2 protocol except that the "
            "structure-statistics router is replaced by one directly learned scalar "
            "for each modeled forward/inverse relation state."
        )
    )
    parser.add_argument("--config", default="config/supplementary_experiments.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--reference",
        choices=("auto", "existing", "rerun"),
        default="auto",
        help=(
            "auto: reuse a complete frozen paper D2 reference if present, otherwise rerun D2; "
            "existing: require an existing complete D2 reference; "
            "rerun: train D2 again in the current environment."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Run only selected seed(s). Omit for the formal paper seeds 42-51.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic end-to-end check. NEVER use smoke outputs in the paper.",
    )
    return parser.parse_args()


def _norm_relpath(value: str) -> Path:
    return Path(str(value).replace("\\", "/"))


def _hp_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("gamma", "learning_rate", "adversarial_temperature", "embedding_dim", "delta")
    for key in keys:
        if key not in a or key not in b:
            return False
        if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
            if float(a[key]) != float(b[key]):
                return False
        elif a[key] != b[key]:
            return False
    return True


def _row_data_matches(row: dict[str, Any], audit: dict[str, Any]) -> bool:
    data = row.get("identity", {}).get("data", {})
    expected = audit.get("sha256", {})
    for key in ("train", "validation", "test"):
        if str(data.get(key)) != str(expected.get(key)):
            return False
    return True


def _row_training_matches(row: dict[str, Any], training: dict[str, Any]) -> bool:
    observed = row.get("identity", {}).get("training", {})
    for key in (
        "loss",
        "adversarial_temperature",
        "optimizer",
        "learning_rate",
        "max_steps",
        "batch_size",
        "num_negatives",
        "evaluation_batch_size",
        "negative_sampler_filtered",
        "drop_last",
    ):
        if key not in observed or key not in training:
            return False
        if isinstance(observed[key], (int, float)) and isinstance(training[key], (int, float)):
            if float(observed[key]) != float(training[key]):
                return False
        elif observed[key] != training[key]:
            return False
    return True


def _find_existing_reference(
    root: Path,
    *,
    seeds: list[int],
    hp: dict[str, Any],
    audit: dict[str, Any],
    training: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]] | None:
    base = root / "results" / "supplementary_runs"
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/RUN_INDEX.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for index_path in candidates:
        try:
            index = read_json(index_path)
        except Exception:
            continue
        core = list(index.get("core", []))
        d2 = {int(r["seed"]): r for r in core if r.get("model") == REFERENCE_NAME}
        d1 = {int(r["seed"]): r for r in core if r.get("model") == "D1"}
        if not all(seed in d2 for seed in seeds):
            continue
        selected = [d2[seed] for seed in seeds]
        if not all(r.get("status") == "PASS" and r.get("split") == "test" for r in selected):
            continue
        if not all(bool(r.get("reciprocal")) for r in selected):
            continue
        if not all(_hp_equal(r.get("hp", {}), hp) for r in selected):
            continue
        if not all(_row_data_matches(r, audit) for r in selected):
            continue
        if not all(_row_training_matches(r, training) for r in selected):
            continue
        ref_root = index_path.parent
        valid_ranks = True
        for row in selected:
            rank_path = ref_root / _norm_relpath(row["ranks_file"])
            if not rank_path.exists() or sha(rank_path) != row.get("ranks_sha256"):
                valid_ranks = False
                break
        if not valid_ranks:
            continue
        d1_rows = [d1[seed] for seed in seeds if seed in d1]
        return ref_root, selected, d1_rows
    return None


def _load_ranks(base: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = base / _norm_relpath(row["ranks_file"])
    if not path.exists():
        raise FileNotFoundError(path)
    if sha(path) != row["ranks_sha256"]:
        raise RuntimeError(f"Rank checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return data["triples"].copy(), data["ranks"].copy()


def _aggregate_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    metrics = ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10")
    out: dict[str, Any] = {"model": label, "n": len(rows)}
    for metric in metrics:
        s = summary([float(r["metrics"][metric]) for r in rows])
        out[f"{metric}_mean"] = s["mean"]
        out[f"{metric}_std"] = s["std"]
    if rows:
        params = [int(r["parameter_counts"]["trainable_real_scalar_parameters"]) for r in rows]
        if len(set(params)) != 1:
            raise RuntimeError(f"Parameter count changed across seeds for {label}")
        out["trainable_real_scalar_parameters"] = params[0]
        out["train_seconds_mean"] = summary([float(r["train_seconds"]) for r in rows])["mean"]
    return out


def _summarize(
    *,
    out: Path,
    control_rows: list[dict[str, Any]],
    reference_root: Path,
    reference_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    config: dict[str, Any],
    reference_mode: str,
    smoke: bool,
) -> None:
    dest = out / "tables" / "per_relation_angle"
    dest.mkdir(parents=True, exist_ok=True)

    control_by_seed = {int(r["seed"]): r for r in control_rows}
    ref_by_seed = {int(r["seed"]): r for r in reference_rows}
    seeds = sorted(control_by_seed.keys() & ref_by_seed.keys())
    if not seeds:
        raise RuntimeError("No paired D2_PerRelation/D2 seeds available")

    per_seed = []
    for seed in seeds:
        for label, row in ((CONTROL_NAME, control_by_seed[seed]), (REFERENCE_NAME, ref_by_seed[seed])):
            per_seed.append(
                {
                    "model": label,
                    "seed": seed,
                    **row["metrics"],
                    "train_seconds": row.get("train_seconds"),
                    "trainable_real_scalar_parameters": row["parameter_counts"]["trainable_real_scalar_parameters"],
                    "implementation": row.get("implementation"),
                }
            )
    table(dest / "per_seed.csv", per_seed)

    summaries = [
        _aggregate_rows([control_by_seed[s] for s in seeds], CONTROL_NAME),
        _aggregate_rows([ref_by_seed[s] for s in seeds], REFERENCE_NAME),
    ]
    d1_by_seed = {int(r["seed"]): r for r in d1_rows}
    d1_paired = [d1_by_seed[s] for s in seeds if s in d1_by_seed]
    if len(d1_paired) == len(seeds):
        summaries.append(_aggregate_rows(d1_paired, "D1"))
    table(dest / "summary.csv", summaries)

    primary = {
        "comparison": f"{CONTROL_NAME} minus {REFERENCE_NAME}",
        "metric": "mrr",
        **paired(
            [control_by_seed[s]["metrics"]["mrr"] for s in seeds],
            [ref_by_seed[s]["metrics"]["mrr"] for s in seeds],
        ),
        "multiple_comparison_note": "Single pre-specified primary control comparison; no multiplicity correction required.",
    }
    table(dest / "paired_seed_test.csv", [primary])

    diffs = []
    for seed in seeds:
        q_control, r_control = _load_ranks(out, control_by_seed[seed])
        q_ref, r_ref = _load_ranks(reference_root, ref_by_seed[seed])
        if not np.array_equal(q_control, q_ref):
            raise RuntimeError(f"Paired query mismatch at seed={seed}")
        diffs.append(1.0 / r_control - 1.0 / r_ref)
    bootstrap_rows = [
        {
            "comparison": f"{CONTROL_NAME} minus {REFERENCE_NAME}",
            **bootstrap_differences(
                diffs,
                int(config["bootstrap_repeats"]),
                int(config["bootstrap_seed"]),
                unit,
            ),
        }
        for unit in ("query", "triple", "seed_and_triple")
    ]
    table(dest / "paired_bootstrap.csv", bootstrap_rows)

    angles = []
    for row in control_rows:
        for state in row.get("relation_fusion_state", []):
            angles.append({"model": CONTROL_NAME, "seed": row["seed"], **state})
    table(dest / "angle_per_seed.csv", angles)

    control_summary = summaries[0]
    ref_summary = summaries[1]
    param_rows = [
        {
            "model": CONTROL_NAME,
            "trainable_real_scalar_parameters": control_summary.get("trainable_real_scalar_parameters"),
            "direct_relation_angle_parameters": control_rows[0].get("control_design", {}).get("direct_angle_parameters"),
            "adapter_description": "one scalar per modeled forward/inverse relation state",
        },
        {
            "model": REFERENCE_NAME,
            "trainable_real_scalar_parameters": ref_summary.get("trainable_real_scalar_parameters"),
            "direct_relation_angle_parameters": 10,
            "adapter_description": "10-D shared linear router over Train-only relation features",
        },
    ]
    if len(summaries) == 3:
        param_rows.append(
            {
                "model": "D1",
                "trainable_real_scalar_parameters": summaries[2].get("trainable_real_scalar_parameters"),
                "direct_relation_angle_parameters": 0,
                "adapter_description": "global TH-RotatE fusion with reciprocal training",
            }
        )
    table(dest / "parameter_comparison.csv", param_rows)

    max_abs_shift = max(
        (abs(float(x["theta_shift_radians"])) for x in angles),
        default=0.0,
    )
    delta = float(control_rows[0]["hp"]["delta"])
    if max_abs_shift > delta + 1.0e-6:
        raise RuntimeError("Direct per-relation control exceeded the frozen Delta bound")

    result = {
        "status": "PASS",
        "smoke": smoke,
        "control": CONTROL_NAME,
        "reference": REFERENCE_NAME,
        "reference_mode": reference_mode,
        "paired_seeds": seeds,
        "formula": CONTROL_FORMULA,
        "frozen_delta_radians": delta,
        "max_abs_final_shift_radians": max_abs_shift,
        "summary": summaries,
        "primary_paired_mrr_test": primary,
        "bootstrap": bootstrap_rows,
        "interpretation_guard": (
            "This control tests whether Train-only structural features plus shared routing outperform "
            "a simple relation-identity lookup under the same reciprocal TH-RotatE geometry and angle bound. "
            "A non-significant difference should not be described as proof of equivalence."
        ),
    }
    write_json(dest / "results.json", result)

    delta_mrr = float(primary["mean_delta"])
    ci_low = primary["ci95_low"]
    ci_high = primary["ci95_high"]
    control_mrr = float(control_summary["mrr_mean"])
    ref_mrr = float(ref_summary["mrr_mean"])
    control_params = int(control_summary["trainable_real_scalar_parameters"])
    ref_params = int(ref_summary["trainable_real_scalar_parameters"])
    if delta_mrr > 0:
        direction = "D2_PerRelation 的平均 MRR 更高"
    elif delta_mrr < 0:
        direction = "原 D2 的平均 MRR 更高"
    else:
        direction = "两者平均 MRR 相同"

    lines = [
        "# 每关系直接学习 bounded angle 对照实验",
        "",
        "**冒烟测试结果，不得用于论文。**" if smoke else "该实验使用论文冻结配置，不根据 Test 重新调参。",
        "",
        "## 对照定义",
        "",
        f"- 控制模型：`{CONTROL_NAME}`。",
        f"- 公式：`{CONTROL_FORMULA}`。",
        "- 每个 R14 正向关系及其内部逆向状态各拥有 1 个独立标量；CRH-L4MKG 共 14 类 R14 关系，因此为 28 个直接角度偏移参数。",
        "- 不输入 TPH/HPT、关系频次、类型签名、业务角色或最佳逆关系分数；不使用 frequency shrinkage。",
        "- 保留 D2 的全局融合锚点、TransH/RotatE 两分支、互逆训练、NSSA、负采样、训练预算和评价协议。",
        f"- Delta 固定为 {delta:.2f} rad，与论文 D2 相同；偏移从 0 初始化。",
        "",
        "## 核心结果",
        "",
        f"- {CONTROL_NAME} mean MRR = {control_mrr:.6f}",
        f"- D2 mean MRR = {ref_mrr:.6f}",
        f"- paired Delta MRR = {delta_mrr:+.6f}",
        f"- 95% paired-seed CI = [{ci_low:.6f}, {ci_high:.6f}]" if ci_low is not None else "- 95% paired-seed CI = NA",
        f"- two-sided paired t p = {primary['p_t_two_sided']:.6g}" if primary["p_t_two_sided"] is not None else "- paired t p = NA",
        f"- sign-flip p = {primary['p_sign_flip_two_sided']:.6g}" if primary["p_sign_flip_two_sided"] is not None else "- sign-flip p = NA",
        f"- 方向：{direction}。",
        f"- trainable parameters: {CONTROL_NAME}={control_params:,}, D2={ref_params:,} (difference {control_params-ref_params:+,}).",
        f"- 最大保存终态 |angle shift| = {max_abs_shift:.6f} rad <= Delta={delta:.6f} rad。",
        "",
        "## 审稿解释",
        "",
        "这个对照只回答一个问题：在相同几何、互逆训练和角度上界下，使用 Train-only 结构统计的共享低容量 router，是否优于简单的 relation-ID 独立 bounded scalar。",
        "如果 D2 明显优于该控制，可直接支持结构统计驱动适配而不是简单 relation-specific gating；如果二者接近，应把贡献表述为低参数、受界的关系级适配，而不要声称结构特征本身带来主要性能；如果直接角度控制更好，则需要重新审视结构 router 的必要性。",
        "",
        "## 输出文件",
        "",
        "- `tables/per_relation_angle/per_seed.csv`：逐 seed 指标。",
        "- `tables/per_relation_angle/summary.csv`：均值和样本标准差。",
        "- `tables/per_relation_angle/paired_seed_test.csv`：预设主比较的种子配对检验。",
        "- `tables/per_relation_angle/paired_bootstrap.csv`：query/triple/seed+triple 三层重采样。",
        "- `tables/per_relation_angle/angle_per_seed.csv`：每个关系方向的最终 bounded angle 状态。",
        "- `tables/per_relation_angle/parameter_comparison.csv`：参数量对照。",
    ]
    (out / "REPORT_PER_RELATION_ANGLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / args.config)

    import torch
    import yaml
    from filelock import FileLock, Timeout
    from importlib.metadata import version

    from supplementary.engine import Engine, runtime
    from throtate_repro.multidataset_experiment import _build_context

    if version("pykeen") != "1.11.1":
        raise RuntimeError("This experiment requires PyKEEN 1.11.1")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")

    formal_seeds = [int(x) for x in config["core_seeds"]]
    seeds = [int(x) for x in (args.seed or formal_seeds)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate --seed values")
    if not set(seeds).issubset(set(formal_seeds)):
        raise ValueError(f"Seeds must be a subset of the paper core seeds: {formal_seeds}")

    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    base = yaml.safe_load((root / "config/multidataset_comparison.yaml").read_text(encoding="utf-8"))
    if args.smoke:
        config = dict(config)
        config.update(
            embedding_dim=8,
            max_steps=3,
            batch_size=16,
            num_negatives=2,
            evaluation_batch_size=4,
            evaluation_slice_size=8,
            bootstrap_repeats=100,
            torch_num_threads=min(2, int(config["torch_num_threads"])),
        )
        seeds = seeds[: min(2, len(seeds))]
    base["training"].update(
        {k: config[k] for k in ("max_steps", "batch_size", "num_negatives", "evaluation_batch_size")}
    )
    base["model"]["embedding_dim"] = int(config["embedding_dim"])
    base["training"]["device"] = device
    base["evaluation_slice_size_runtime"] = int(config["evaluation_slice_size"])

    control_hp = settings(config, "D2")
    if float(control_hp["gamma"]) != 3.0 or float(control_hp["delta"]) != 0.15:
        raise RuntimeError("Formal direct-angle control must keep paper D2 gamma=3 and Delta=0.15")
    if not args.smoke and (
        int(control_hp["embedding_dim"]) != 200
        or int(base["training"]["max_steps"]) != 3000
        or int(base["training"]["batch_size"]) != 1024
        or int(base["training"]["num_negatives"]) != 64
        or float(control_hp["learning_rate"]) != 0.001
        or float(control_hp["adversarial_temperature"]) != 1.0
    ):
        raise RuntimeError("Formal settings have drifted from the paper protocol")

    fingerprints = {
        "experiment": "direct_per_relation_bounded_angle_control_v1",
        "formula": CONTROL_FORMULA,
        "control_model": CONTROL_NAME,
        "reference_model": REFERENCE_NAME,
        "config": config,
        "selected_seeds": seeds,
        "source": source_hashes(root),
        "runtime": runtime(device),
        "smoke": bool(args.smoke),
        "original_data": base["datasets"]["ownkg"]["sha256"],
    }
    identity = digest(fingerprints)
    if args.output:
        out = (root / args.output).resolve()
    else:
        out = root / "results" / ("per_relation_angle_SMOKE" if args.smoke else "per_relation_bounded_angle") / identity[:16]
    out.mkdir(parents=True, exist_ok=True)

    try:
        lock = FileLock(str(out / "RUN.lock"), timeout=0)
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError(f"Another process is using {out}") from exc

    try:
        if args.smoke:
            from supplementary.smoke_data import prepare

            base = prepare(base, out)

        ctx = _build_context(root, base, "ownkg", out / "context", include_test=True)
        engine = Engine(root, out, identity, device)
        protocol = {
            "identity": identity,
            **fingerprints,
            "paper_settings": {
                "dataset": "CRH-L4MKG / ownkg fixed R14 split",
                "embedding_dim": int(control_hp["embedding_dim"]),
                "gamma": float(control_hp["gamma"]),
                "learning_rate": float(control_hp["learning_rate"]),
                "adversarial_temperature": float(control_hp["adversarial_temperature"]),
                "delta_radians": float(control_hp["delta"]),
                "max_steps": int(base["training"]["max_steps"]),
                "batch_size": int(base["training"]["batch_size"]),
                "num_negatives": int(base["training"]["num_negatives"]),
                "negative_sampler": "unfiltered Bernoulli",
                "loss": "NSSA",
                "optimizer": "Adam",
                "reciprocal_training": True,
                "evaluation": "full-entity filtered realistic both-side ranking",
            },
            "control_design": {
                "formula": CONTROL_FORMULA,
                "one_direct_scalar_per_modeled_forward_inverse_state": True,
                "expected_active_states": 2 * len(ctx["modeled_relations"]),
                "uses_train_structural_statistics": False,
                "uses_frequency_shrinkage": False,
                "delta_reselection": False,
                "test_used_for_parameter_selection": False,
            },
        }
        freeze(out / "PER_RELATION_PROTOCOL_FROZEN.json", protocol)
        write_json(out / "DATA_PROTOCOL_CHECK.json", ctx["audit"])

        control_rows: list[dict[str, Any]] = []
        for seed in seeds:
            row = engine.run(base, ctx, CONTROL_NAME, control_hp, seed, "test")
            control_rows.append(row)
            write_json(out / "RUN_INDEX.json", {"control": control_rows})

        reference_root: Path
        reference_rows: list[dict[str, Any]]
        d1_rows: list[dict[str, Any]] = []
        reference_mode = args.reference
        found = None
        if args.reference in ("auto", "existing") and not args.smoke:
            found = _find_existing_reference(
                root,
                seeds=seeds,
                hp=control_hp,
                audit=ctx["audit"],
                training=base["training"],
            )
        if found is not None:
            reference_root, reference_rows, d1_rows = found
            reference_mode = f"existing:{reference_root.relative_to(root)}"
        elif args.reference == "existing":
            raise RuntimeError(
                "No complete frozen D2 reference with matching data/protocol/ranks was found. "
                "Use --reference rerun or --reference auto."
            )
        else:
            reference_root = out
            reference_rows = []
            for seed in seeds:
                reference_rows.append(engine.run(base, ctx, REFERENCE_NAME, control_hp, seed, "test"))
            reference_mode = "rerun_in_current_environment"

        write_json(
            out / "REFERENCE.json",
            {
                "mode": reference_mode,
                "reference_root": str(reference_root),
                "reference_model": REFERENCE_NAME,
                "seeds": seeds,
                "hp": control_hp,
                "note": (
                    "Existing reference reuse is allowed only after exact data hashes, paper hyperparameters, "
                    "training protocol, reciprocal flag, PASS status, and saved rank checksums are verified."
                ),
            },
        )
        _summarize(
            out=out,
            control_rows=control_rows,
            reference_root=reference_root,
            reference_rows=reference_rows,
            d1_rows=d1_rows,
            config=config,
            reference_mode=reference_mode,
            smoke=args.smoke,
        )
        write_json(
            root / "results" / ("PER_RELATION_ANGLE_SMOKE_LATEST.json" if args.smoke else "PER_RELATION_ANGLE_LATEST.json"),
            {"output_directory": str(out.relative_to(root)) if out.is_relative_to(root) else str(out), "identity": identity},
        )
        print(f"\nCOMPLETE: {out / 'REPORT_PER_RELATION_ANGLE.md'}", flush=True)
        print(f"TABLES:   {out / 'tables' / 'per_relation_angle'}", flush=True)
        return 0
    finally:
        lock.release()
