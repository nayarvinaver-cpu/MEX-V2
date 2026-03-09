from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Iterable

from srt_model.config import ConfigValidationError
from srt_model.io.portfolio import LoanRecord


@lru_cache(maxsize=32)
def normalize_amortisation_type(amortisation_type: str) -> str:
    text = amortisation_type.strip().lower()
    # Spec 77: mapping table for amortization types.
    if text in {"bullet", "n/a", "until further notice"}:
        return "bullet"
    if text in {"amortisation", "annuity"}:
        return "linear"
    raise ConfigValidationError(f"Unsupported Amortisation_Type '{amortisation_type}'.")


def projected_balance(loan: LoanRecord, t: date, as_of_date: date) -> float:
    """Projected scheduled balance B(t) before default effects.

    Spec 79/80: bullet amortization rule and start-of-day maturity drop.
    Spec 78/79: linear day-ratio amortization for Amortisation/Annuity.
    """
    if t <= as_of_date:
        return float(loan.outstanding_principal)

    if t >= loan.maturity_date:
        return 0.0

    mode = normalize_amortisation_type(loan.amortisation_type)
    if mode == "bullet":
        return float(loan.outstanding_principal)

    total_days = (loan.maturity_date - as_of_date).days
    if total_days <= 0:
        return 0.0
    elapsed_days = max(0, (t - as_of_date).days)
    frac = elapsed_days / total_days
    bal = float(loan.outstanding_principal) * max(0.0, 1.0 - frac)
    return bal


def projected_balance_with_prepayment(
    loan: LoanRecord,
    t: date,
    as_of_date: date,
    prepayment_date: date | None,
) -> float:
    """Projected balance including optional stochastic prepayment event."""
    if prepayment_date is not None and t >= prepayment_date:
        return 0.0
    return projected_balance(loan=loan, t=t, as_of_date=as_of_date)


def ead_at_default(
    loan: LoanRecord,
    tau_date: date,
    as_of_date: date,
    prepayment_date: date | None = None,
) -> float:
    """Exposure-at-default frozen at default date.

    Spec 57/58: EAD is evaluated at default date and then frozen through notice lag.
    """
    return projected_balance_with_prepayment(
        loan=loan,
        t=tau_date,
        as_of_date=as_of_date,
        prepayment_date=prepayment_date,
    )


def pool_scheduled_balance(loans: Iterable[LoanRecord], t: date, as_of_date: date) -> float:
    """Scheduled/performing pool balance aggregate (without mechanical default reductions)."""
    return float(sum(projected_balance(loan, t, as_of_date) for loan in loans))
