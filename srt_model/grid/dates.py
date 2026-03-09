from __future__ import annotations

import calendar
from datetime import date

from srt_model.config import ConfigValidationError


def _is_month_end(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def add_months(d: date, n_months: int, eom_on: bool) -> date:
    """Add calendar months with EOM handling.

    Spec 8/9/10: +3M stepping with explicit EOM ON/OFF semantics.
    """
    total_month = (d.year * 12 + (d.month - 1)) + n_months
    year = total_month // 12
    month = total_month % 12 + 1

    if eom_on and _is_month_end(d):
        return _month_end(year, month)

    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def yearfrac(start: date, end: date, day_count: str) -> float:
    """Year fraction on [start, end) for selected premium day count.

    Spec 25/26/27/28: day-count convention is user-selected.
    """
    if end < start:
        raise ConfigValidationError("yearfrac end date must be >= start date.")
    basis = day_count.strip().upper()
    days = (end - start).days
    if basis == "ACT/360":
        return days / 360.0
    if basis == "ACT/365F":
        return days / 365.0
    if basis == "30E/360":
        d1 = min(start.day, 30)
        d2 = min(end.day, 30)
        return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0
    if basis == "30/360 (US)":
        d1 = 30 if start.day == 31 else start.day
        d2 = end.day
        if d1 == 30 and d2 == 31:
            d2 = 30
        return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0
    if basis == "ACT/ACT (ISDA)":
        if start == end:
            return 0.0
        total = 0.0
        cur = start
        while cur < end:
            year_end = date(cur.year + 1, 1, 1)
            nxt = min(year_end, end)
            denom = 366.0 if calendar.isleap(cur.year) else 365.0
            total += (nxt - cur).days / denom
            cur = nxt
        return total
    if basis == "ACT/ACT (ICMA)":
        raise ConfigValidationError(
            "ACT/ACT (ICMA) requires coupon-period metadata and is not yet implemented."
        )
    raise ConfigValidationError(f"Unsupported day count '{day_count}'")

