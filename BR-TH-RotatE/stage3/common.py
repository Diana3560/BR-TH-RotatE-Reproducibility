from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml


STAGE3_CODE_FILES = (
    "stage3/common.py",
    "stage3/models.py",
    "stage3/baseline_runner.py",
    "stage3/candidate_spaces.py",
    "stage3/candidate_eval.py",
    "stage3/candidate_runner.py",
    "stage3/run_stage3.py",
    "stage3/run_candidate_sensitivity.py",
    "stage3/config.yaml",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fingerprint(value: object) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_stage3_config(project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    path = project_root / "stage3" / "config.yaml"
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Stage3 config is not a mapping: {path}")
    cfg["_config_path"] = str(path)
    return cfg


def load_base_config(project_root: str | Path, stage3_cfg: Mapping[str, Any]):
    from throtate_repro.multidataset_experiment import load_multidataset_config

    project_root = Path(project_root).resolve()
    expected = str(stage3_cfg["stage3_protocol"]["base_config"])
    if expected != "config/multidataset_comparison.yaml":
        raise ValueError(f"Unsupported base config path: {expected}")
    return load_multidataset_config(project_root)


def runtime_environment() -> dict[str, Any]:
    try:
        pykeen_version = metadata.version("pykeen")
    except metadata.PackageNotFoundError:
        pykeen_version = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pykeen": pykeen_version,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def require_pykeen_version(stage3_cfg: Mapping[str, Any]) -> str:
    required = str(stage3_cfg["stage3_protocol"]["pykeen_version"])
    try:
        actual = metadata.version("pykeen")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "PyKEEN is not installed. Run `python -m pip install -r requirements_stage3.txt` first."
        ) from exc
    if actual != required:
        raise RuntimeError(f"PyKEEN {required} is required for this package; found {actual}")
    return actual


def implementation_hashes(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    result: dict[str, str] = {}
    for relative in STAGE3_CODE_FILES:
        path = root / relative
        if path.is_file():
            result[relative] = sha256_file(path)
    return result


def stage3_identity(
    project_root: str | Path,
    stage3_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    dataset_key: str,
    *,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    code = implementation_hashes(root)
    payload = {
        "stage3_protocol": stage3_cfg["stage3_protocol"],
        "baselines": stage3_cfg["baselines"],
        "candidate_sensitivity": stage3_cfg["candidate_sensitivity"],
        "base_model": base_cfg["model"],
        "base_training": base_cfg["training"],
        "base_protocol": base_cfg["protocol"],
        "dataset_key": dataset_key,
        "dataset_spec": base_cfg["datasets"][dataset_key],
        "data_sha256": audit["sha256"],
        "stage3_config_sha256": sha256_file(stage3_cfg["_config_path"]),
        "implementation_sha256": code,
    }
    fingerprint = stable_fingerprint(payload)
    return {
        "fingerprint": fingerprint,
        "dataset_key": dataset_key,
        "data_sha256": dict(audit["sha256"]),
        "stage3_config_sha256": payload["stage3_config_sha256"],
        "implementation_sha256": code,
        "runtime_environment": runtime_environment(),
    }


def aggregate(values: Iterable[float]) -> dict[str, float]:
    xs = [float(v) for v in values]
    if not xs:
        raise ValueError("Cannot aggregate an empty sequence")
    return {
        "mean": statistics.fmean(xs),
        "std_sample": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def resolve_device(base_cfg: Mapping[str, Any], override: str | None) -> str | None:
    value = override if override is not None else str(base_cfg["training"].get("device", "auto"))
    if value == "auto":
        return None
    return value


def make_output_dir(
    project_root: str | Path,
    stage3_cfg: Mapping[str, Any],
    dataset_key: str,
    fingerprint: str,
    *,
    kind: str,
) -> Path:
    root = Path(project_root).resolve()
    key = "baselines_directory" if kind == "baselines" else "candidate_directory"
    path = root / stage3_cfg["output"][key] / dataset_key / fingerprint[:16]
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_pointer_path(project_root: str | Path, stage3_cfg: Mapping[str, Any], dataset_key: str, *, kind: str) -> Path:
    root = Path(project_root).resolve()
    key = "baselines_directory" if kind == "baselines" else "candidate_directory"
    return root / stage3_cfg["output"][key] / dataset_key / "LATEST.json"


def write_latest_pointer(
    project_root: str | Path,
    stage3_cfg: Mapping[str, Any],
    dataset_key: str,
    *,
    kind: str,
    identity: Mapping[str, Any],
    output_dir: Path,
) -> None:
    root = Path(project_root).resolve()
    write_json(
        latest_pointer_path(root, stage3_cfg, dataset_key, kind=kind),
        {
            "dataset_key": dataset_key,
            "experiment_fingerprint": identity["fingerprint"],
            "output_directory": str(output_dir.relative_to(root)),
        },
    )
