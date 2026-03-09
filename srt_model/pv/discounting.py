from __future__ import annotations

from datetime import date
from typing import Callable, Iterable


def pv_cashflows(cashflows: Iterable[tuple[date, float]], df_fn: Callable[[date], float]) -> float:
    """Discount and sum dated cashflows."""
    total = 0.0
    for cf_date, amount in cashflows:
        total += float(amount) * float(df_fn(cf_date))
    return float(total)

