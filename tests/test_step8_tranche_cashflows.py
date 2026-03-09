from __future__ import annotations

from datetime import date
import unittest

from srt_model.tranche.cashflows import (
    incremental_tranche_loss,
    premium_accrual_piecewise,
    redemption_cashflow,
    scheduled_tranche_notional,
    tranche_outstanding_notional,
    write_down_cashflow,
)


class TestTrancheCashflows(unittest.TestCase):
    def test_notional_and_loss_formulas(self) -> None:
        n_sched = scheduled_tranche_notional(alpha=0.06, pool_balance_sched=1_000.0)
        self.assertAlmostEqual(n_sched, 60.0)
        n_tr = tranche_outstanding_notional(n_sched=60.0, cum_loss=15.0)
        self.assertAlmostEqual(n_tr, 45.0)
        self.assertAlmostEqual(incremental_tranche_loss(delta_loss=12.0, n_tr_before=10.0), 10.0)

    def test_write_down_and_redemption_signs(self) -> None:
        self.assertAlmostEqual(write_down_cashflow(7.5), -7.5)
        self.assertAlmostEqual(redemption_cashflow(33.0), 33.0)

    def test_piecewise_premium_split_at_notice(self) -> None:
        start = date(2025, 1, 1)
        end = date(2025, 4, 1)
        notice = [date(2025, 2, 1)]

        def n_tr_at(d: date) -> float:
            return 100.0 if d < date(2025, 2, 1) else 80.0

        prem = premium_accrual_piecewise(
            period_start=start,
            period_end=end,
            notice_dates_in_period=notice,
            n_tr_at_start_of_date=n_tr_at,
            spread=0.10,
            premium_day_count="ACT/360",
        )
        expected = 0.10 * ((31.0 / 360.0) * 100.0 + (59.0 / 360.0) * 80.0)
        self.assertAlmostEqual(prem, expected, places=12)


if __name__ == "__main__":
    unittest.main()

