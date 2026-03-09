from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

import holidays

from srt_model.config import CalendarSelection, ConfigValidationError


@lru_cache(maxsize=128)
def _build_single_calendar_cached(name: str, years_key: tuple[int, ...]):
    key = name.strip().upper()
    years = tuple(sorted(set(int(y) for y in years_key)))
    if key in {"TARGET", "TARGET2"}:
        return holidays.ECB(years=years)
    if key == "GERMANY":
        return holidays.country_holidays("DE", years=years)
    if key == "SWEDEN":
        return holidays.country_holidays("SE", years=years)
    if key == "UK":
        return holidays.country_holidays("UK", years=years)
    if key == "FRANCE":
        return holidays.country_holidays("FR", years=years)
    if key == "SPAIN":
        return holidays.country_holidays("ES", years=years)
    if key == "SWITZERLAND":
        return holidays.country_holidays("CH", years=years)
    raise ConfigValidationError(f"Unsupported calendar '{name}'")


def _build_single_calendar(name: str, years: Iterable[int]):
    return _build_single_calendar_cached(name.strip().upper(), tuple(sorted(set(int(y) for y in years))))


@lru_cache(maxsize=200_000)
def _is_business_day_cached(d: date, calendars: tuple[str, ...]) -> bool:
    if d.weekday() >= 5:
        return False
    years = (d.year,)
    for cal_name in calendars:
        cal = _build_single_calendar(cal_name, years=years)
        if d in cal:
            return False
    return True


def is_business_day(d: date, selection: CalendarSelection) -> bool:
    """Business-day predicate under selected single or joint calendar set.

    Spec 15: non-business = weekend or holiday.
    Spec 14: joint-of-N => business only if business in all selected calendars.
    """
    calendars: tuple[str, ...]
    if selection.joint_enabled:
        calendars = tuple(str(c).strip().upper() for c in selection.joint_calendars)
    else:
        calendars = (str(selection.base_calendar).strip().upper(),)
    return _is_business_day_cached(d, calendars)


def adjust_preceding(d: date, selection: CalendarSelection) -> date:
    """Adjust date using Preceding convention."""
    out = d
    while not is_business_day(out, selection):
        out = out - timedelta(days=1)
    return out


def adjust_modified_following(d: date, selection: CalendarSelection) -> date:
    """Adjust date using Modified Following convention.

    Spec 12/37/87/89/97: Modified Following is used for payment/protection/legal dates.
    """
    if is_business_day(d, selection):
        return d

    out = d
    while not is_business_day(out, selection):
        out = out + timedelta(days=1)
    if out.month == d.month:
        return out

    out = d
    while not is_business_day(out, selection):
        out = out - timedelta(days=1)
    return out
