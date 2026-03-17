from __future__ import annotations

from datetime import date
from typing import Callable, Iterable

from srt_model.config import (
    TRANCHE_AMORTIZATION_MODE_PRO_RATA,
    TRANCHE_AMORTIZATION_MODE_SEQUENTIAL,
    normalize_tranche_amortization_mode,
)
from srt_model.grid.dates import yearfrac


def scheduled_tranche_notional(
    alpha: float,
    pool_balance_sched: float,
    tranche_amortization_mode: str = TRANCHE_AMORTIZATION_MODE_PRO_RATA,
    junior_cap_amount: float | None = None,
) -> float:
    """Scheduled tranche notional before default losses.

    Spec 159: scheduled junior notional is derived from the scheduled total stack.
    User rule: support both pro-rata and sequential scheduled amortization.
    """
    total_sched = max(0.0, float(pool_balance_sched))
    mode = normalize_tranche_amortization_mode(tranche_amortization_mode)
    if mode == TRANCHE_AMORTIZATION_MODE_SEQUENTIAL:
        if junior_cap_amount is None:
            raise ValueError("junior_cap_amount is required for sequential amortization.")
        return float(min(max(0.0, float(junior_cap_amount)), total_sched))
    return float(max(0.0, float(alpha)) * total_sched)


def tranche_outstanding_notional(n_sched: float, cum_loss: float) -> float:
    """Outstanding tranche notional after cumulative losses.

    Spec 158: N_tr(t) = max(N_sched(t) - CumLoss(t), 0).
    """
    return float(max(n_sched - cum_loss, 0.0))


def incremental_tranche_loss(delta_loss: float, n_tr_before: float) -> float:
    """Loss allocated to junior tranche at event time.

    Spec 158: DeltaTrLoss(t) = min(DeltaLoss(t), N_tr(t-)).
    """
    return float(min(delta_loss, n_tr_before))


def write_down_cashflow(delta_tranche_loss: float) -> float:
    """Funded SPV write-down representation."""
    # Spec 163: CF_wd(t_N) = -DeltaTrLoss(t_N).
    return float(-delta_tranche_loss)


def redemption_cashflow(n_tr_before_lfm: float) -> float:
    """Principal redemption at legal final maturity."""
    # Spec 164: CF_red(T_LFM) = +N_tr(T_LFM-).
    return float(max(n_tr_before_lfm, 0.0))


def premium_accrual_piecewise(
    period_start: date,
    period_end: date,
    notice_dates_in_period: Iterable[date],
    n_tr_at_start_of_date: Callable[[date], float],
    spread: float,
    premium_day_count: str,
) -> float:
    """Premium accrual with intra-period segmentation at Notice Dates.

    Spec 161/162 and 49/50: split [T_k, T_k+1) at notice dates; use notional at start of each segment.
    """
    cuts = sorted({d for d in notice_dates_in_period if period_start < d < period_end})
    points = [period_start, *cuts, period_end]
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        yf = yearfrac(a, b, premium_day_count)
        base = n_tr_at_start_of_date(a)
        total += spread * yf * base
    return float(total)
