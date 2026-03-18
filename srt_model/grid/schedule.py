from __future__ import annotations

from datetime import date

from srt_model.config import ConfigValidationError
from srt_model.grid.calendar import adjust_following, adjust_modified_following
from srt_model.grid.dates import add_months


def effective_accrual_start(as_of_date: date, accrual_start_date: date) -> date:
    # Spec 17/72: valuation accrual starts at max(as-of, accrual start).
    return max(as_of_date, accrual_start_date)


def build_payment_schedule(
    first_payment_date: date,
    as_of_date: date,
    accrual_start_date: date,
    accrual_end_date: date,
    eom_on: bool,
    calendar_selection,
) -> list[date]:
    """Build quarterly payment dates for valuation with final-stub logic.

    Spec 4/5/6/8/22/23/24: rules-only schedule, +3M stepping, first date strictly after start_eff,
    and final-stub payment at accrual end when needed.
    """
    if accrual_start_date > accrual_end_date:
        raise ConfigValidationError("Accrual Start Date cannot be after Accrual End Date.")

    start_eff = effective_accrual_start(as_of_date, accrual_start_date)
    unadjusted: list[date] = []
    d = first_payment_date
    while d <= start_eff:
        d = add_months(d, 3, eom_on=eom_on)

    while d <= accrual_end_date:
        unadjusted.append(d)
        d = add_months(d, 3, eom_on=eom_on)

    if accrual_end_date not in unadjusted:
        unadjusted.append(accrual_end_date)

    adjusted = [adjust_modified_following(x, calendar_selection) for x in unadjusted]
    out: list[date] = []
    for d_adj in adjusted:
        if d_adj > as_of_date and (not out or out[-1] != d_adj):
            out.append(d_adj)
    return out


def compute_default_event_date(default_date: date, calendar_selection) -> date:
    """Compute business-day-adjusted default event date from raw simulated default date."""
    return adjust_following(default_date, calendar_selection)


def previous_payment_date_on_or_before(
    as_of_date: date,
    first_payment_date: date,
    eom_on: bool,
    calendar_selection,
) -> date:
    """Return last adjusted payment date <= as-of under quarterly roll rules."""
    cand = first_payment_date
    adj = adjust_modified_following(cand, calendar_selection)
    if adj > as_of_date:
        while adj > as_of_date:
            cand = add_months(cand, -3, eom_on=eom_on)
            adj = adjust_modified_following(cand, calendar_selection)
        return adj

    while True:
        nxt_cand = add_months(cand, 3, eom_on=eom_on)
        nxt_adj = adjust_modified_following(nxt_cand, calendar_selection)
        if nxt_adj > as_of_date:
            break
        cand = nxt_cand
        adj = nxt_adj
    return adj
