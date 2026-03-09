from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from srt_model.config import ConfigValidationError
from srt_model.ratings import collapse_to_survival_bucket


@dataclass(frozen=True)
class _RatingHazardCurve:
    tenors: np.ndarray
    lambdas: np.ndarray
    cum_hazard: np.ndarray


class SurvivalCurveSet:
    """Adapter around bootstrapped hazard/survival matrices."""

    def __init__(self, hazard_matrix: pd.DataFrame, survival_matrix: pd.DataFrame):
        if hazard_matrix.empty or survival_matrix.empty:
            raise ConfigValidationError("Survival/hazard matrices cannot be empty.")
        if not hazard_matrix.index.equals(survival_matrix.index):
            raise ConfigValidationError("Hazard and survival matrices must have the same rating index.")

        self._hazard = hazard_matrix.copy()
        self._survival = survival_matrix.copy()
        self._curves = {
            rating: self._build_hazard_curve(self._hazard.loc[rating])
            for rating in self._hazard.index.tolist()
        }

    @staticmethod
    def _build_hazard_curve(hazard_row: pd.Series) -> _RatingHazardCurve:
        row = hazard_row
        tenors = row.index.to_numpy(dtype=float)
        lambdas = row.to_numpy(dtype=float)
        order = np.argsort(tenors)
        tenors = tenors[order]
        lambdas = lambdas[order]
        dt = np.diff(np.concatenate(([0.0], tenors)))
        cum_hazard = np.cumsum(lambdas * dt)
        return _RatingHazardCurve(tenors=tenors, lambdas=lambdas, cum_hazard=cum_hazard)

    @classmethod
    def from_csv(cls, hazard_path: str, survival_path: str) -> "SurvivalCurveSet":
        hazard = pd.read_csv(hazard_path)
        survival = pd.read_csv(survival_path)
        if "Rating" not in hazard.columns or "Rating" not in survival.columns:
            raise ConfigValidationError("Survival/hazard CSV files must have a 'Rating' column.")

        hazard = hazard.set_index("Rating")
        survival = survival.set_index("Rating")
        hazard.columns = [float(c) for c in hazard.columns]
        survival.columns = [float(c) for c in survival.columns]
        hazard = hazard.sort_index().sort_index(axis=1)
        survival = survival.sort_index().sort_index(axis=1)
        return cls(hazard_matrix=hazard, survival_matrix=survival)

    def _resolve_rating(self, rating: str) -> str:
        # Spec 102 + user decision: normalize labels and collapse CC/C/D -> CCC for lookup.
        lookup = collapse_to_survival_bucket(rating)
        if lookup not in self._curves:
            raise ConfigValidationError(
                f"Rating '{rating}' normalized to '{lookup}', but no matching survival curve exists."
            )
        return lookup

    def supported_ratings(self) -> tuple[str, ...]:
        return tuple(self._curves.keys())

    def survival(self, rating: str, t_years: float) -> float:
        """Return S(t) under piecewise-constant forward hazards.

        Spec 127/128/129: U = Phi(X), solve S(t)=U with analytic piecewise hazards.
        """
        if t_years < 0:
            raise ConfigValidationError("t_years must be non-negative.")
        if t_years == 0:
            return 1.0

        key = self._resolve_rating(rating)
        curve = self._curves[key]
        if t_years > curve.tenors[-1]:
            raise ConfigValidationError(
                f"Requested survival at t={t_years} exceeds curve max tenor {curve.tenors[-1]}."
            )

        idx = int(np.searchsorted(curve.tenors, t_years, side="left"))
        t_left = 0.0 if idx == 0 else float(curve.tenors[idx - 1])
        h_left = 0.0 if idx == 0 else float(curve.cum_hazard[idx - 1])
        h = h_left + float(curve.lambdas[idx]) * (t_years - t_left)
        return float(np.exp(-h))

    def inverse_default_time_years(self, rating: str, u: float) -> float:
        """Analytic inversion for S(t)=u under piecewise-constant hazards.

        Spec 127/128/129: inversion is analytic by hazard interval (no numerical root finder).
        Returns np.inf when the implied default time is beyond last tenor.
        """
        if not (0.0 < u <= 1.0):
            raise ConfigValidationError(f"u must be in (0,1], got {u}.")
        if u == 1.0:
            return 0.0

        key = self._resolve_rating(rating)
        curve = self._curves[key]
        target_hazard = -float(np.log(u))
        if target_hazard > float(curve.cum_hazard[-1]):
            return float(np.inf)

        idx = int(np.searchsorted(curve.cum_hazard, target_hazard, side="left"))
        h_left = 0.0 if idx == 0 else float(curve.cum_hazard[idx - 1])
        t_left = 0.0 if idx == 0 else float(curve.tenors[idx - 1])
        lam = float(curve.lambdas[idx])
        if lam <= 0.0:
            return float(curve.tenors[idx])
        tau = t_left + (target_hazard - h_left) / lam
        return float(tau)

    def inverse_default_time_years_vec(self, rating: str, u: np.ndarray) -> np.ndarray:
        values = np.asarray(u, dtype=float)
        return np.array([self.inverse_default_time_years(rating, float(x)) for x in values], dtype=float)


def load_survival_curve_set_for_currency(currency: str, cfg: Any) -> SurvivalCurveSet:
    ccy = str(currency).strip().upper()
    if ccy == "EUR":
        return SurvivalCurveSet.from_csv(cfg.HAZARD_RATES_EUR_PATH, cfg.SURVIVAL_PROBS_EUR_PATH)
    if ccy == "USD":
        return SurvivalCurveSet.from_csv(cfg.HAZARD_RATES_USD_PATH, cfg.SURVIVAL_PROBS_USD_PATH)
    raise ConfigValidationError(f"Unsupported currency for survival curves: {currency}")
