from __future__ import annotations

from typing import Sequence

import numpy as np

from srt_model.config import ConfigValidationError
from srt_model.curves.survival_adapter import SurvivalCurveSet


def generate_default_time_years(
    u_matrix: np.ndarray,
    debtor_curve_keys: Sequence[str],
    curves: SurvivalCurveSet,
) -> np.ndarray:
    """Map copula uniforms to obligor default times in years.

    Spec 127/128/129: solve S(t)=U analytically per obligor curve.
    Spec 112: one latent variable/default time per obligor.
    """
    u = np.asarray(u_matrix, dtype=float)
    if u.ndim != 2:
        raise ConfigValidationError("u_matrix must be 2D with shape (n_paths, n_obligors).")
    n_paths, n_obligors = u.shape
    if len(debtor_curve_keys) != n_obligors:
        raise ConfigValidationError(
            f"debtor_curve_keys length {len(debtor_curve_keys)} does not match u_matrix columns {n_obligors}."
        )

    tau = np.empty_like(u, dtype=float)
    for j, curve_key in enumerate(debtor_curve_keys):
        tau[:, j] = curves.inverse_default_time_years_vec(str(curve_key), u[:, j])
    return tau

