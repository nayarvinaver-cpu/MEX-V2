from __future__ import annotations

from srt_model.config import ConfigValidationError


def solve_par_spread_closed_form(pv01: float, pv_wd_positive: float, pv_red: float) -> float:
    """Closed-form par spread solver.

    Spec 174: NPV(s)=s*PV01-PV_wd+PV_red => s*=(PV_wd-PV_red)/PV01.
    """
    if pv01 <= 0.0:
        raise ConfigValidationError("PV01 must be > 0 to solve par spread.")
    return float((pv_wd_positive - pv_red) / pv01)

