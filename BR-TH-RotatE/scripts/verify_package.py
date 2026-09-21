from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def main() -> int:
    required = [
        "README.md",
        ".gitignore",
        "requirements.txt",
        "requirements_stage3.txt",
        "requirements_supplementary.txt",
        "data/DATA_README.md",
        "config/multidataset_comparison.yaml",
        "config/supplementary_experiments.json",
        "src/throtate_repro/model.py",
        "src/throtate_repro/bounded_adaptive_v26.py",
        "src/throtate_repro/multidataset_experiment.py",
        "supplementary/runner.py",
        "scripts/run_multidataset_experiment.py",
        "scripts/check_multidataset_data.py",
        "scripts/check_manuscript_alignment.py",
        "scripts/verify_package.py",
        "results/manuscript/table05_crh_l4mkg_baselines.csv",
        "results/manuscript/table06_croefkg.csv",
        "results/manuscript/table07_core_ablation.csv",
        "results/manuscript/table08_repeated_splits.csv",
        "results/manuscript/table09_mpnorm.csv",
        "results/manuscript/table10_significance.csv",
        "results/manuscript/table11_efficiency.csv",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        fail(f"missing required release files: {missing}")
    print(f"[PASS] required release layout: {len(required)} files")

    data_dir = ROOT / "data"
    data_entries = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*") if p.is_file())
    if data_entries != ["DATA_README.md"]:
        fail(f"data/ must contain only DATA_README.md; found {data_entries}")
    print("[PASS] data/ contains only DATA_README.md")

    if (ROOT / "external_data").exists():
        fail("external_data/ must not be bundled in the public release")
    print("[PASS] external_data/ is not bundled")

    forbidden_suffixes = {".exe", ".msi", ".bat", ".cmd", ".ps1", ".pyc", ".pyo", ".tsv", ".parquet", ".feather", ".sqlite", ".db", ".zip", ".7z", ".rar"}
    forbidden_files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden_suffixes]
    if forbidden_files:
        fail(f"forbidden bundled files: {forbidden_files[:20]}")
    print("[PASS] no executables, caches, archives, or row-level dataset files")

    cache_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
    found_cache_dirs = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_dir() and p.name in cache_dirs]
    if found_cache_dirs:
        fail(f"cache directories remain: {found_cache_dirs[:20]}")
    print("[PASS] no cache directories")

    py_files = sorted(p for p in ROOT.rglob("*.py") if p.is_file())
    for path in py_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"[PASS] Python syntax: {len(py_files)} files")

    json_files = sorted(ROOT.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8-sig"))
    print(f"[PASS] JSON parse: {len(json_files)} files")

    yaml_files = sorted(list(ROOT.rglob("*.yaml")) + list(ROOT.rglob("*.yml")))
    for path in yaml_files:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"[PASS] YAML parse: {len(yaml_files)} files")

    cfg = yaml.safe_load((ROOT / "config/multidataset_comparison.yaml").read_text(encoding="utf-8"))
    for dataset_key, spec in cfg["datasets"].items():
        path_fields = ["full_file", "entity_types_file", "split_dir"]
        for optional in ("relation_schema_file", "dataset_stats_file"):
            if optional in spec:
                path_fields.append(optional)
        for field in path_fields:
            value = str(spec[field]).replace("\\", "/")
            if not value.startswith("external_data/"):
                fail(f"{dataset_key}.{field} must point to the ignored external_data/ workspace; got {value}")
    print("[PASS] runtime dataset paths are external and repository-local data/ is documentation-only")

    text_suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".txt"}
    text_files = [p for p in ROOT.rglob("*") if p.is_file() and (p.suffix.lower() in text_suffixes or p.name in {"README", "LICENSE"})]
    absolute_windows = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
    absolute_unix_home = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/")
    bad: list[str] = []
    for path in text_files:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if absolute_windows.search(text) or absolute_unix_home.search(text):
            bad.append(f"absolute local path: {rel}")
    if bad:
        fail("release text audit failed: " + "; ".join(bad[:20]))
    print("[PASS] no local absolute paths in release text")

    manifest = ROOT / "RELEASE_SHA256.txt"
    if manifest.is_file():
        checked = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError as exc:
                fail(f"malformed RELEASE_SHA256.txt line: {line!r}")
            path = ROOT / relative
            if not path.is_file():
                fail(f"release-manifest file missing: {relative}")
            actual = sha256(path)
            if actual != expected:
                fail(f"release-manifest hash mismatch: {relative}")
            checked += 1
        print(f"[PASS] RELEASE_SHA256.txt: {checked} files")
    else:
        print("[NOTE] RELEASE_SHA256.txt not generated yet")

    print("[PASS] public code package verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
