from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from srt_model.config import ConfigValidationError


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


def validate_rho(rho: float) -> None:
    # Spec 122/123: rho must satisfy 0 <= rho < 1.
    if rho < 0.0 or rho >= 1.0:
        raise ConfigValidationError(f"rho must satisfy 0 <= rho < 1, got {rho}")


def simulate_uniforms_one_factor(
    n_paths: int,
    n_obligors: int,
    rho: float,
    seed: int,
) -> np.ndarray:
    """Simulate obligor uniforms U=Phi(X) under one-factor Gaussian copula.

    Spec 114/115/116/117/121/124:
    X_i = sqrt(rho)*Y + sqrt(1-rho)*eps_i, U_i = Phi(X_i), fixed-seed pseudo RNG.
    """
    if n_paths <= 0 or n_obligors <= 0:
        raise ConfigValidationError("n_paths and n_obligors must be positive integers.")
    validate_rho(float(rho))

    rng = np.random.default_rng(seed)
    y = rng.standard_normal(size=(n_paths, 1))
    eps = rng.standard_normal(size=(n_paths, n_obligors))
    x = np.sqrt(rho) * y + np.sqrt(1.0 - rho) * eps
    u = _norm_cdf(x)
    return np.clip(u, 1e-15, 1.0)
