from __future__ import annotations

"""Validation-only diagnostics for TransH/RotatE branch scales.

This module does not alter model scoring or training. It summarizes raw branch
L2 distances, weighted branch contributions, and the empirical alignment between
the TransH translation vector d_r and hyperplane normal w_r.
"""

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def _as_1d_float(value: Tensor) -> np.ndarray:
    array = value.detach().float().cpu().reshape(-1).numpy()
    if not np.all(np.isfinite(array)):
        raise ValueError("Diagnostic tensor contains non-finite values")
    return array


def distribution_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot summarize an empty diagnostic vector")
    if not np.all(np.isfinite(x)):
        raise ValueError("Diagnostic values contain NaN/Inf")
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std_population": float(np.std(x, ddof=0)),
        "min": float(np.min(x)),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(np.max(x)),
    }


def component_summary(
    *,
    transh_distance: Tensor,
    rotate_distance: Tensor,
    fusion_alpha: Tensor,
    fusion_beta: Tensor,
) -> dict[str, Any]:
    d_h = _as_1d_float(transh_distance)
    d_r = _as_1d_float(rotate_distance)
    alpha = _as_1d_float(fusion_alpha)
    beta = _as_1d_float(fusion_beta)
    if not (d_h.shape == d_r.shape == alpha.shape == beta.shape):
        raise ValueError("Component diagnostic arrays must have identical shapes")

    c_h = alpha * d_h
    c_r = beta * d_r
    eps = 1.0e-12
    total = c_h + c_r
    return {
        "raw_transh_distance": distribution_summary(d_h),
        "raw_rotate_distance": distribution_summary(d_r),
        "weighted_transh_contribution": distribution_summary(c_h),
        "weighted_rotate_contribution": distribution_summary(c_r),
        "raw_mean_ratio_transh_over_rotate": float(np.mean(d_h) / max(np.mean(d_r), eps)),
        "raw_median_ratio_transh_over_rotate": float(np.median(d_h) / max(np.median(d_r), eps)),
        "weighted_mean_ratio_transh_over_rotate": float(np.mean(c_h) / max(np.mean(c_r), eps)),
        "weighted_median_ratio_transh_over_rotate": float(np.median(c_h) / max(np.median(c_r), eps)),
        "mean_transh_share_of_fused_distance": float(np.mean(c_h / np.maximum(total, eps))),
        "mean_rotate_share_of_fused_distance": float(np.mean(c_r / np.maximum(total, eps))),
        "fusion_alpha": distribution_summary(alpha),
        "fusion_beta": distribution_summary(beta),
    }


def translation_normal_alignment(relation_translation: Tensor, relation_normal: Tensor) -> dict[str, Any]:
    """Quantify |cos(d_r, w_r)| without imposing or assuming orthogonality."""
    d = relation_translation.detach().float()
    w = relation_normal.detach().float()
    if d.shape != w.shape or d.ndim != 2:
        raise ValueError("Expected relation translation/normal matrices with identical shape [R, d]")
    numerator = torch.sum(d * w, dim=-1).abs()
    denominator = torch.linalg.vector_norm(d, dim=-1) * torch.linalg.vector_norm(w, dim=-1)
    cosine = numerator / denominator.clamp_min(1.0e-12)
    return {
        "constraint_enforced": False,
        "definition": "absolute cosine |d_r^T w_r| / (||d_r||_2 ||w_r||_2)",
        "absolute_cosine": distribution_summary(_as_1d_float(cosine)),
        "note": (
            "The reported implementation normalizes w_r but does not explicitly project d_r "
            "onto the relation hyperplane; therefore w_r^T d_r = 0 is not a hard constraint."
        ),
    }


def aggregate_seed_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No per-seed diagnostic rows")
    stages = ("before_training", "after_training")
    metrics = (
        "raw_mean_ratio_transh_over_rotate",
        "raw_median_ratio_transh_over_rotate",
        "weighted_mean_ratio_transh_over_rotate",
        "weighted_median_ratio_transh_over_rotate",
        "mean_transh_share_of_fused_distance",
        "mean_rotate_share_of_fused_distance",
    )
    output: dict[str, Any] = {}
    for stage in stages:
        stage_out: dict[str, Any] = {}
        for metric in metrics:
            values = [float(row[stage]["components"][metric]) for row in rows]
            stage_out[metric] = {
                "mean_across_seeds": float(np.mean(values)),
                "sample_std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "per_seed": values,
            }
        for branch_key in (
            "raw_transh_distance",
            "raw_rotate_distance",
            "weighted_transh_contribution",
            "weighted_rotate_contribution",
        ):
            values = [float(row[stage]["components"][branch_key]["mean"]) for row in rows]
            stage_out[f"{branch_key}_mean"] = {
                "mean_across_seeds": float(np.mean(values)),
                "sample_std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "per_seed": values,
            }
        cos_values = [
            float(row[stage]["translation_normal_alignment"]["absolute_cosine"]["mean"])
            for row in rows
        ]
        stage_out["translation_normal_abs_cosine_mean"] = {
            "mean_across_seeds": float(np.mean(cos_values)),
            "sample_std_across_seeds": float(np.std(cos_values, ddof=1)) if len(cos_values) > 1 else 0.0,
            "per_seed": cos_values,
        }
        output[stage] = stage_out
    return output


def scalar_close(a: float, b: float, *, atol: float = 1.0e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)
