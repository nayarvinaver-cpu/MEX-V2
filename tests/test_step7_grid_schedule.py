from __future__ import annotations

from datetime import date
import unittest

from srt_model.config import CalendarSelection
from srt_model.grid.calendar import adjust_following, adjust_modified_following, adjust_preceding, is_business_day
from srt_model.grid.dates import add_months
from srt_model.grid.schedule import build_payment_schedule, compute_default_event_date


GERMANY_CAL = CalendarSelection(base_calendar="GERMANY", joint_enabled=False, joint_calendars=())


class TestDateRolling(unittest.TestCase):
    def test_add_months_eom_on(self) -> None:
        d0 = date(2025, 1, 31)
        self.assertEqual(add_months(d0, 3, eom_on=True), date(2025, 4, 30))
        self.assertEqual(add_months(date(2025, 4, 30), 3, eom_on=True), date(2025, 7, 31))

    def test_add_months_eom_off(self) -> None:
        d0 = date(2025, 1, 31)
        d1 = add_months(d0, 3, eom_on=False)
        d2 = add_months(d1, 3, eom_on=False)
        self.assertEqual(d1, date(2025, 4, 30))
        self.assertEqual(d2, date(2025, 7, 30))


class TestCalendarAdjustments(unittest.TestCase):
    def test_preceding_and_modified_following(self) -> None:
        sat = date(2026, 1, 3)
        self.assertFalse(is_business_day(sat, GERMANY_CAL))
        self.assertEqual(adjust_preceding(sat, GERMANY_CAL), date(2026, 1, 2))
        self.assertEqual(adjust_following(sat, GERMANY_CAL), date(2026, 1, 5))
        self.assertEqual(adjust_modified_following(sat, GERMANY_CAL), date(2026, 1, 5))


class TestScheduleBuilder(unittest.TestCase):
    def test_schedule_with_final_stub(self) -> None:
        dates = build_payment_schedule(
            first_payment_date=date(2025, 3, 31),
            as_of_date=date(2025, 1, 10),
            accrual_start_date=date(2025, 1, 1),
            accrual_end_date=date(2025, 8, 15),
            eom_on=False,
            calendar_selection=GERMANY_CAL,
        )
        self.assertEqual(dates[0], date(2025, 3, 31))
        self.assertIn(date(2025, 6, 30), dates)
        self.assertEqual(dates[-1], date(2025, 8, 15))

    def test_default_event_date_adjusts_following(self) -> None:
        default_event = compute_default_event_date(
            default_date=date(2026, 1, 31),
            calendar_selection=GERMANY_CAL,
        )
        self.assertEqual(default_event, date(2026, 2, 2))


if __name__ == "__main__":
    unittest.main()
