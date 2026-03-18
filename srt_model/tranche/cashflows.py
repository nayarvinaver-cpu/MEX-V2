from __future__ import annotations

from datetime import date
from typing import Callable, Iterable

from srt_model.config import (
    TRANCHE_AMORTIZATION_MODE_PRO_RATA,
    TRANCHE_AMORTIZATION_MODE_SEQUENTIAL,
    normalize_tranche_amortization_mode,
)
from srt_model.grid.dates import yearfrac


def scheduled_tranche_band(
    *,
    total_stack_sched: float,
    total_stack_asof: float,
    attachment_point: float,
    detachment_point: float,
    tranche_amortization_mode: str = TRANCHE_AMORTIZATION_MODE_PRO_RATA,
) -> tuple[float, float]:
    """Scheduled attachment/detachment notionals for the selected tranche."""
    total_sched = max(0.0, float(total_stack_sched))
    total_asof = max(0.0, float(total_stack_asof))
    attach_pct = max(0.0, float(attachment_point))
    detach_pct = max(attach_pct, float(detachment_point))
    mode = normalize_tranche_amortization_mode(tranche_amortization_mode)

    if mode == TRANCHE_AMORTIZATION_MODE_SEQUENTIAL:
        attach_asof = attach_pct * total_asof
        detach_asof = detach_pct * total_asof
        return float(min(attach_asof, total_sched)), float(min(detach_asof, total_sched))

    return float(attach_pct * total_sched), float(detach_pct * total_sched)


def scheduled_tranche_notional(
    *,
    total_stack_sched: float,
    total_stack_asof: float,
    attachment_point: float,
    detachment_point: float,
    tranche_amortization_mode: str = TRANCHE_AMORTIZATION_MODE_PRO_RATA,
) -> float:
    """Scheduled selected-tranche notional before default losses."""
    attach_notional, detach_notional = scheduled_tranche_band(
        total_stack_sched=total_stack_sched,
        total_stack_asof=total_stack_asof,
        attachment_point=attachment_point,
        detachment_point=detachment_point,
        tranche_amortization_mode=tranche_amortization_mode,
    )
    return float(max(detach_notional - attach_notional, 0.0))


def cumulative_tranche_loss(
    *,
    cumulative_portfolio_loss: float,
    attachment_notional: float,
    detachment_notional: float,
) -> float:
    """Selected-tranche cumulative loss given cumulative portfolio loss."""
    attach = max(0.0, float(attachment_notional))
    detach = max(attach, float(detachment_notional))
    loss = max(0.0, float(cumulative_portfolio_loss))
    return float(min(loss, detach) - min(loss, attach))


def tranche_outstanding_notional(
    *,
    attachment_notional: float,
    detachment_notional: float,
    cumulative_portfolio_loss: float,
) -> float:
    """Outstanding selected-tranche notional after cumulative portfolio losses."""
    attach = max(0.0, float(attachment_notional))
    detach = max(attach, float(detachment_notional))
    scheduled = detach - attach
    return float(
        max(
            scheduled
            - cumulative_tranche_loss(
                cumulative_portfolio_loss=cumulative_portfolio_loss,
                attachment_notional=attach,
                detachment_notional=detach,
            ),
            0.0,
        )
    )


def incremental_tranche_loss(
    *,
    delta_portfolio_loss: float,
    cumulative_portfolio_loss_before: float,
    attachment_notional: float,
    detachment_notional: float,
) -> float:
    """Incremental selected-tranche loss at an event time."""
    delta = max(0.0, float(delta_portfolio_loss))
    before = cumulative_tranche_loss(
        cumulative_portfolio_loss=cumulative_portfolio_loss_before,
        attachment_notional=attachment_notional,
        detachment_notional=detachment_notional,
    )
    after = cumulative_tranche_loss(
        cumulative_portfolio_loss=cumulative_portfolio_loss_before + delta,
        attachment_notional=attachment_notional,
        detachment_notional=detachment_notional,
    )
    return float(max(after - before, 0.0))


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
    event_dates_in_period: Iterable[date],
    n_tr_at_start_of_date: Callable[[date], float],
    spread: float,
    premium_day_count: str,
) -> float:
    """Premium accrual with intra-period segmentation at default event dates.

    Split [T_k, T_k+1) at default-driven write-down dates and use notional at
    the start of each segment.
    """
    cuts = sorted({d for d in event_dates_in_period if period_start < d < period_end})
    points = [period_start, *cuts, period_end]
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        yf = yearfrac(a, b, premium_day_count)
        base = n_tr_at_start_of_date(a)
        total += spread * yf * base
    return float(total)
