from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "manuscript"
METRICS = ("mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10")


def read_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS / name
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def by_key(rows: list[dict[str, str]], key: str = "model") -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def assert_close(label: str, actual: float, expected: float, tol: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: actual={actual:.12g}, expected={expected:.12g}")
    print(f"[PASS] {label}: {actual:.6f}")


def assert_int(label: str, actual: int, expected: int) -> None:
    if int(actual) != int(expected):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")
    print(f"[PASS] {label}: {actual}")


def check_table(
    filename: str,
    expected: dict[str, tuple[int, tuple[float, float, float, float, float]]],
    *,
    count_column: str = "n_seeds",
) -> None:
    rows = by_key(read_csv(filename))
    for model, (n, values) in expected.items():
        if model not in rows:
            raise AssertionError(f"{filename}: missing model {model}")
        row = rows[model]
        if count_column:
            assert_int(f"{filename} {model} count", int(float(row[count_column])), n)
        for metric, expected_value in zip(METRICS, values):
            column = f"{metric}_mean" if f"{metric}_mean" in row else metric
            tol = 5e-4 if metric == "mr" else 5e-6
            assert_close(f"{filename} {model} {metric}", float(row[column]), expected_value, tol)


def aggregate_per_seed(filename: str) -> dict[str, dict[str, float]]:
    rows = read_csv(filename)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    out: dict[str, dict[str, float]] = {}
    for model, items in grouped.items():
        out[model] = {"n": float(len(items))}
        for metric in METRICS:
            values = [float(r[metric]) for r in items]
            out[model][f"{metric}_mean"] = statistics.fmean(values)
            out[model][f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def crosscheck_summary(per_seed_file: str, summary_file: str, model_names: tuple[str, ...]) -> None:
    aggregate = aggregate_per_seed(per_seed_file)
    summary = by_key(read_csv(summary_file))
    for model in model_names:
        assert_int(f"{per_seed_file} {model} rows", int(aggregate[model]["n"]), int(float(summary[model].get("n_seeds") or summary[model].get("mrr_n"))))
        for metric in METRICS:
            actual = aggregate[model][f"{metric}_mean"]
            expected = float(summary[model][f"{metric}_mean"])
            assert_close(f"{per_seed_file} -> {summary_file} {model} {metric}", actual, expected, 2e-6 if metric != "mr" else 2e-4)


def main() -> int:
    required = [
        "table05_crh_l4mkg_baselines.csv",
        "table05_crh_l4mkg_per_seed.csv",
        "table06_croefkg.csv",
        "table06_croefkg_per_seed.csv",
        "table07_core_ablation.csv",
        "core_per_seed.csv",
        "table08_repeated_splits.csv",
        "repeated_splits_per_split.csv",
        "table09_mpnorm.csv",
        "mpnorm_per_seed.csv",
        "table10_significance.csv",
        "table11_efficiency.csv",
        "efficiency_per_repeat.csv",
        "mechanism_shrinkage.csv",
        "shrinkage_per_seed.csv",
        "mechanism_role_control.csv",
        "role_control_per_seed.csv",
    ]
    missing = [name for name in required if not (RESULTS / name).is_file()]
    if missing:
        raise SystemExit(f"[FAIL] missing manuscript result files: {missing}")

    check_table(
        "table05_crh_l4mkg_baselines.csv",
        {
            "TransH": (5, (0.525086, 363.285, 0.434524, 0.573810, 0.715476)),
            "RotatE": (5, (0.518704, 1282.743, 0.446925, 0.552778, 0.678472)),
            "D0": (5, (0.552339, 252.648, 0.471925, 0.592857, 0.726488)),
            "RatE": (5, (0.532055, 1186.332, 0.454067, 0.566964, 0.713690)),
            "CompoundE": (5, (0.554453, 632.358, 0.474901, 0.591567, 0.729861)),
            "PairRE": (5, (0.539217, 826.235, 0.458730, 0.578968, 0.711508)),
            "TH-RotatE-Cap": (5, (0.551993, 250.975, 0.470040, 0.595040, 0.726786)),
            "D2": (5, (0.556818, 191.206, 0.474206, 0.600298, 0.734325)),
        },
    )
    crosscheck_summary(
        "table05_crh_l4mkg_per_seed.csv",
        "table05_crh_l4mkg_baselines.csv",
        ("TransH", "RotatE", "D0", "RatE", "CompoundE", "PairRE", "TH-RotatE-Cap", "D2"),
    )

    check_table(
        "table06_croefkg.csv",
        {
            "TransH": (5, (0.307947, 269.908, 0.241515, 0.328551, 0.447779)),
            "RotatE": (5, (0.311758, 600.204, 0.248580, 0.336562, 0.429497)),
            "D0": (5, (0.309285, 267.727, 0.245157, 0.326657, 0.446468)),
            "D1": (5, (0.313607, 251.950, 0.246176, 0.334086, 0.453532)),
            "D2": (5, (0.317274, 247.239, 0.250036, 0.338310, 0.456154)),
        },
    )
    crosscheck_summary("table06_croefkg_per_seed.csv", "table06_croefkg.csv", ("TransH", "RotatE", "D0", "D1", "D2"))

    check_table(
        "table07_core_ablation.csv",
        {
            "D0": (10, (0.552470, 252.649, 0.471726, 0.592560, 0.726240)),
            "D1": (10, (0.554014, 228.083, 0.469395, 0.599554, 0.732887)),
            "A0": (10, (0.556009, 209.192, 0.477778, 0.593750, 0.726042)),
            "D2": (10, (0.556860, 196.939, 0.474504, 0.600198, 0.733532)),
        },
        count_column="mrr_n",
    )
    crosscheck_summary("core_per_seed.csv", "table07_core_ablation.csv", ("D0", "D1", "A0", "D2"))

    check_table(
        "table08_repeated_splits.csv",
        {
            "D0": (5, (0.567504, 265.743, 0.492034, 0.601171, 0.729718)),
            "D1": (5, (0.566874, 242.889, 0.485785, 0.605073, 0.742050)),
            "A0": (5, (0.567517, 227.540, 0.491471, 0.601105, 0.733024)),
            "D2": (5, (0.570112, 211.004, 0.490183, 0.608611, 0.742115)),
        },
        count_column="mrr_n",
    )
    repeated = read_csv("repeated_splits_per_split.csv")
    split_ids = {row["split_seed"] for row in repeated}
    assert_int("repeated-split count", len(split_ids), 5)
    if any(int(row["n_training_seeds"]) != 3 for row in repeated):
        raise AssertionError("repeated-split rows must average exactly three training seeds")
    print("[PASS] repeated-split training seeds: 3 per split/model")

    check_table(
        "table09_mpnorm.csv",
        {
            "D0": (5, (0.552339, 252.648, 0.471925, 0.592857, 0.726488)),
            "D0_MPNorm": (5, (0.543110, 265.902, 0.457440, 0.591369, 0.717659)),
            "D1": (5, (0.554670, 227.221, 0.470238, 0.600198, 0.734325)),
            "D1_MPNorm": (5, (0.546335, 258.606, 0.460516, 0.594444, 0.724802)),
            "D2": (5, (0.556818, 191.206, 0.474206, 0.600298, 0.734325)),
            "D2_MPNorm": (5, (0.547279, 257.160, 0.459821, 0.597024, 0.725000)),
        },
        count_column="mrr_n",
    )
    crosscheck_summary("mpnorm_per_seed.csv", "table09_mpnorm.csv", ("D0", "D0_MPNorm", "D1", "D1_MPNorm", "D2", "D2_MPNorm"))

    sig = {(r["dataset"], r["comparison"]): r for r in read_csv("table10_significance.csv")}
    sig_expected = {
        ("CRH-L4MKG", "A0-D0"): (10, 0.003539, 0.001071, 0.006006, 0.0202, 0.0293, (9, 0, 1)),
        ("CRH-L4MKG", "D2-D0"): (10, 0.004390, 0.001743, 0.007038, 0.0136, 0.0293, (9, 0, 1)),
        ("CRH-L4MKG", "D2-D1"): (10, 0.002846, -0.000247, 0.005939, 0.0671, 0.0723, (8, 0, 2)),
        ("CROEFKG", "D2-D0"): (5, 0.007989, -0.004929, 0.020906, 0.1611, 0.2500, (3, 0, 2)),
        ("CROEFKG", "D2-D1"): (5, 0.003667, -0.000465, 0.007799, 0.1388, 0.1250, (4, 0, 1)),
    }
    for key, exp in sig_expected.items():
        row = sig[key]
        n, delta, low, high, holm, flip, wtl = exp
        assert_int(f"Table10 {key} n", int(row["n_pairs"]), n)
        for col, val in [("mean_delta_mrr", delta), ("ci95_low", low), ("ci95_high", high), ("holm_p", holm), ("exact_sign_flip_p", flip)]:
            assert_close(f"Table10 {key} {col}", float(row[col]), val, 5e-5)
        actual_wtl = (int(row["wins"]), int(row["ties"]), int(row["losses"]))
        if actual_wtl != wtl:
            raise AssertionError(f"Table10 {key} W/T/L: actual={actual_wtl}, expected={wtl}")
        print(f"[PASS] Table10 {key} W/T/L: {actual_wtl}")

    core = defaultdict(dict)
    for row in read_csv("core_per_seed.csv"):
        core[int(row["seed"])][row["model"]] = float(row["mrr"])
    assert_close("Core per-seed D2-D0 mean delta", statistics.fmean(v["D2"] - v["D0"] for v in core.values()), float(sig[("CRH-L4MKG", "D2-D0")]["mean_delta_mrr"]), 1e-12)
    croef = defaultdict(dict)
    for row in read_csv("table06_croefkg_per_seed.csv"):
        croef[int(row["seed"])][row["model"]] = float(row["mrr"])
    assert_close("CROEF per-seed D2-D0 mean delta", statistics.fmean(v["D2"] - v["D0"] for v in croef.values()), float(sig[("CROEFKG", "D2-D0")]["mean_delta_mrr"]), 1e-12)

    eff = by_key(read_csv("table11_efficiency.csv"))
    eff_expected = {
        "D0": (8229402, 5, 45.670, 22422, 539.2, 1119.83, 1364),
        "D1": (8243002, 5, 42.822, 23913, 567.8, 1119.99, 1384),
        "D2": (8243012, 5, 44.354, 23087, 504.5, 1120.05, 1384),
    }
    for model, (params, repeats, ms, pos, qps, alloc, reserved) in eff_expected.items():
        row = eff[model]
        assert_int(f"Table11 {model} parameters", int(row["trainable_parameters"]), params)
        assert_int(f"Table11 {model} repeats", int(row["repeat_count"]), repeats)
        assert_close(f"Table11 {model} ms/update", float(row["ms_per_update_mean"]), ms, 5e-4)
        assert_close(f"Table11 {model} positive/s", float(row["positive_instances_per_second_mean"]), pos, 0.6)
        assert_close(f"Table11 {model} eval q/s", float(row["eval_queries_per_second_mean"]), qps, 0.06)
        assert_close(f"Table11 {model} allocated MiB", float(row["peak_cuda_allocated_mib"]), alloc, 0.01)
        assert_close(f"Table11 {model} reserved MiB", float(row["peak_cuda_reserved_mib"]), reserved, 0.01)
    overhead = (float(eff["D2"]["ms_per_update_mean"]) / float(eff["D1"]["ms_per_update_mean"]) - 1.0) * 100.0
    assert_close("D2 vs D1 update-time overhead (%)", overhead, 3.58, 0.01)

    shrink = by_key(read_csv("mechanism_shrinkage.csv"))
    assert_int("noShrinkage seeds", int(shrink["D2_noShrinkage"]["mrr_n"]), 10)
    shrink_delta = float(shrink["D2"]["mrr_mean"]) - float(shrink["D2_noShrinkage"]["mrr_mean"])
    assert_close("D2 - noShrinkage MRR", shrink_delta, 0.000461, 5e-7)
    assert_close("noShrinkage MR", float(shrink["D2_noShrinkage"]["mr_mean"]), 195.309, 5e-4)

    roles = by_key(read_csv("mechanism_role_control.csv"))
    assert_int("role-control seeds", int(roles["D2_constRole"]["mrr_n"]), 5)
    role_delta = float(roles["D2_constRole"]["mrr_mean"]) - float(roles["D2_constNoRole"]["mrr_mean"])
    assert_close("constRole - constNoRole MRR", role_delta, 0.000351, 5e-7)

    print("\n[PASS] Final manuscript tables 5-11 and mechanism controls match the retained public result files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
