"""Port of the local CL compensation operators for paired VideoQA features.

Direction is always source -> target. For a method-enhanced model, source is
that model's representation and target is the baseline's representation for
the *same* video/question. This is an experimental diagnostic, not a claim
that a map restores the generative answer.

Provenance: cl_reference/run_all_cl.py:m_rot,m_rotscale,m_affine,m_sdc and
cl_reference/geom_cl_maptype.py:ridge. Rotation/scale and affine retain the
original formulas. Ridge adds centering/intercept so it is comparable to the
centered rotation in this paired-feature diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METHODS = ("identity", "rotation", "rotation_scale", "affine", "ridge", "sdc")


def _array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    return array


@dataclass(frozen=True)
class FittedMap:
    method: str
    dimension: int
    matrix: np.ndarray | None = None
    source_mean: np.ndarray | None = None
    target_mean: np.ndarray | None = None
    source_anchors: np.ndarray | None = None
    anchor_deltas: np.ndarray | None = None

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = _array(features, "features")
        if x.shape[1] != self.dimension:
            raise ValueError(f"Expected {self.dimension} feature dimensions, got {x.shape[1]}")
        if self.method == "identity":
            return x.copy()
        if self.method == "affine":
            augmented = np.column_stack((x, np.ones(len(x))))
            return augmented @ self.matrix
        if self.method == "sdc":
            out = np.empty_like(x)
            for index, row in enumerate(x):
                d2 = np.sum((self.source_anchors - row) ** 2, axis=1)
                weights = np.exp(-d2 / (2 * np.median(d2) + 1e-9))
                weights /= weights.sum() + 1e-12
                out[index] = row + weights @ self.anchor_deltas
            return out
        return (x - self.source_mean) @ self.matrix + self.target_mean


def fit_map(
    source: np.ndarray,
    target: np.ndarray,
    method: str,
    *,
    ridge_alpha: float = 1.0,
) -> FittedMap:
    """Fit one map on paired anchors; never use held-out eval rows here."""
    a, b = _array(source, "source"), _array(target, "target")
    if a.shape != b.shape:
        raise ValueError(f"Paired source/target shapes differ: {a.shape} vs {b.shape}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if method == "identity":
        return FittedMap(method, a.shape[1])
    if method == "sdc":
        return FittedMap(method, a.shape[1], source_anchors=a.copy(),
                         anchor_deltas=(b - a).copy())
    if method == "affine":
        augmented = np.column_stack((a, np.ones(len(a))))
        weights, *_ = np.linalg.lstsq(augmented, b, rcond=1e-3)
        return FittedMap(method, a.shape[1], matrix=weights)

    ma, mb = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - ma, b - mb
    if method in ("rotation", "rotation_scale"):
        source_norm2 = float(np.sum(ac ** 2))
        if source_norm2 <= 1e-12:
            raise ValueError("Rotation needs nonconstant source anchors")
        u, singular, vt = np.linalg.svd(ac.T @ bc, full_matrices=False)
        rotation = u @ vt  # same orthogonal Procrustes solution as the CL code
        if method == "rotation_scale":
            rotation *= singular.sum() / source_norm2
        weights = rotation
    else:
        if ridge_alpha <= 0 or not np.isfinite(ridge_alpha):
            raise ValueError("ridge_alpha must be finite and positive")
        weights = np.linalg.solve(
            ac.T @ ac + ridge_alpha * np.eye(a.shape[1]), ac.T @ bc
        )
    return FittedMap(method, a.shape[1], matrix=weights,
                     source_mean=ma, target_mean=mb)
