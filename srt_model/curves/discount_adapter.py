from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from srt_model.config import ConfigValidationError


@dataclass(frozen=True)
class DiscountCurveAdapter:
    """Thin callable adapter that exposes DF(date) by currency."""

    currency: str
    df_fn: Callable[[date | str], float]

    def df(self, q_date: date | str) -> float:
        # Spec 167: use one DF_ccy(date) for all legs.
        return float(self.df_fn(q_date))


def load_discount_curve_adapter(currency: str, cfg: Any) -> DiscountCurveAdapter:
    ccy = str(currency).strip().upper()
    if ccy == "EUR":
        import discount_factors_eur as eur_curve

        curve = eur_curve.load_discount_curve_from_excel(
            file_path=cfg.DISCOUNT_CURVE_EUR_FILE,
            sheet_name="Discount Function",
            allow_extrapolation=False,
        )
        return DiscountCurveAdapter(currency="EUR", df_fn=curve.get_discount_factor)

    if ccy == "USD":
        import discount_factors_usd as usd_curve

        curve = usd_curve.load_discount_curve_from_excel(
            file_path=cfg.DISCOUNT_CURVE_USD_FILE,
            sheet_name="Discount Function",
        )
        return DiscountCurveAdapter(currency="USD", df_fn=curve.discount_factor)

    raise ConfigValidationError(f"Unsupported currency for discount curve: {currency}")
