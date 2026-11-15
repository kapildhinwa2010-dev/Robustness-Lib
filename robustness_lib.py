# Robustness-Lib - Reusable noise-stability library
# Part of the Kapildhinwa2010-dev Toolkit.

"""Robustness-Lib - noise stability checks for compression codecs.

Provides add_noise(), which perturbs a matrix with Gaussian noise, and
stability_check(), which measures how much each toolkit codec degrades
as noise rises.

Honest limitation: quantization is surprisingly noise-tolerant (it clips
to the value range) while Fourier fitting is the most fragile codec -
least-squares coefficients swing hard under noise.
"""

from __future__ import annotations

import numpy as np

__all__ = ["add_noise", "stability_check", "svd_compress", "quantize",
           "fourier_compress", "hybrid_compress", "fourier_basis"]


def fourier_basis(n_points: int, num_coeffs: int = 4) -> np.ndarray:
    """Design matrix [1, sin t, cos t, sin 2t, cos 2t, ...] over one period."""
    if n_points < 1 or num_coeffs < 1:
        raise ValueError("n_points and num_coeffs must be >= 1")
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    cols = [np.ones(n_points)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _check(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    matrix = _check(matrix)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    basis = fourier_basis(matrix.shape[1], num_coeffs)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def hybrid_compress(matrix: np.ndarray, rank: int = 4, bits: int = 8) -> np.ndarray:
    """SVD low-rank followed by quantization."""
    return quantize(svd_compress(matrix, rank=rank), bits=bits)


def add_noise(matrix: np.ndarray, std: float = 0.05, rng: np.random.Generator = None) -> np.ndarray:
    """Gaussian noise with standard deviation std added to a copy of matrix."""
    matrix = _check(matrix)
    if std < 0:
        raise ValueError("std must be >= 0")
    rng = rng or np.random.default_rng(0)
    return matrix + rng.normal(0.0, std, size=matrix.shape)


def stability_check(matrix: np.ndarray, noise_std: float = 0.05,
                    repeats: int = 3, rng: np.random.Generator = None) -> list:
    """Mean relative error of each codec when its input is noisy.

    For each of repeats trials the matrix is perturbed with Gaussian noise
    of standard deviation noise_std, each codec reconstructs the noisy
    input, and the reconstruction is compared against the CLEAN matrix.
    Returns dicts with method, mean_relative_error and stable (mean < 0.5),
    sorted most stable first.
    """
    matrix = _check(matrix)
    if noise_std < 0 or repeats < 1:
        raise ValueError("noise_std must be >= 0 and repeats >= 1")
    rng = rng or np.random.default_rng(0)
    norm = float(np.linalg.norm(matrix))
    methods = {
        "svd_rank4": lambda m: svd_compress(m, rank=4),
        "quantize8": lambda m: quantize(m, bits=8),
        "fourier4": lambda m: fourier_compress(m, num_coeffs=4),
        "hybrid": lambda m: hybrid_compress(m, rank=4, bits=8),
    }
    sums = {name: 0.0 for name in methods}
    for _ in range(repeats):
        noisy = add_noise(matrix, std=noise_std, rng=rng)
        for name, codec in methods.items():
            approx = codec(noisy)
            sums[name] += float(np.linalg.norm(matrix - approx)) / norm if norm else 0.0
    rows = [{
        "method": name,
        "mean_relative_error": total / repeats,
        "stable": (total / repeats) < 0.5,
    } for name, total in sums.items()]
    rows.sort(key=lambda row: row["mean_relative_error"])
    return rows
