from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np

from srt_model.config import ConfigValidationError
from srt_model.io.portfolio import LoanRecord


def quarterly_prepayment_probability(cpr_annual: float) -> float:
    """Convert annual CPR to quarterly prepayment probability.

    Spec 165: p_q = 1 - (1 - CPR_annual)^(1/4).
    """
    if cpr_annual < 0.0 or cpr_annual > 1.0:
        raise ConfigValidationError(f"CPR_annual must be in [0,1], got {cpr_annual}")
    return float(1.0 - (1.0 - cpr_annual) ** 0.25)


def simulate_prepayment_dates(
    loans: Iterable[LoanRecord],
    quarter_dates: list[date],
    enable_prepayment: bool,
    cpr_annual: float,
    rng: np.random.Generator,
) -> dict[str, date | None]:
    """Simulate first prepayment date per loan from quarter event dates.

    Spec 165: each quarter, draw U for each live loan; if U < p_q loan prepays from that quarter onward.
    """
    out: dict[str, date | None] = {}
    loans_list = list(loans)
    if not enable_prepayment or cpr_annual <= 0.0:
        for loan in loans_list:
            out[loan.loan_id] = None
        return out

    p_q = quarterly_prepayment_probability(cpr_annual)
    for loan in loans_list:
        prepay_date: date | None = None
        for q_date in quarter_dates:
            if q_date >= loan.maturity_date:
                break
            u = float(rng.random())
            if u < p_q:
                prepay_date = q_date
                break
        out[loan.loan_id] = prepay_date
    return out

