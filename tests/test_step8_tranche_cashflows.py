from __future__ import annotations

from datetime import date
import unittest

from srt_model.tranche.cashflows import (
    cumulative_tranche_loss,
    incremental_tranche_loss,
    premium_accrual_piecewise,
    redemption_cashflow,
    scheduled_tranche_band,
    scheduled_tranche_notional,
    tranche_outstanding_notional,
    write_down_cashflow,
)


class TestTrancheCashflows(unittest.TestCase):
    def test_pro_rata_scheduled_band_scales_with_total_stack(self) -> None:
        attach, detach = scheduled_tranche_band(
            total_stack_sched=800.0,
            total_stack_asof=1_000.0,
            attachment_point=0.20,
            detachment_point=0.40,
        )
        self.assertAlmostEqual(attach, 160.0)
        self.assertAlmostEqual(detach, 320.0)
        self.assertAlmostEqual(
            scheduled_tranche_notional(
                total_stack_sched=800.0,
                total_stack_asof=1_000.0,
                attachment_point=0.20,
                detachment_point=0.40,
            ),
            160.0,
        )

    def test_mezzanine_tranche_loss_allocation(self) -> None:
        self.assertAlmostEqual(
            cumulative_tranche_loss(
                cumulative_portfolio_loss=150.0,
                attachment_notional=200.0,
                detachment_notional=400.0,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            cumulative_tranche_loss(
                cumulative_portfolio_loss=250.0,
                attachment_notional=200.0,
                detachment_notional=400.0,
            ),
            50.0,
        )
        self.assertAlmostEqual(
            tranche_outstanding_notional(
                attachment_notional=200.0,
                detachment_notional=400.0,
                cumulative_portfolio_loss=250.0,
            ),
            150.0,
        )
        self.assertAlmostEqual(
            incremental_tranche_loss(
                delta_portfolio_loss=60.0,
                cumulative_portfolio_loss_before=180.0,
                attachment_notional=200.0,
                detachment_notional=400.0,
            ),
            40.0,
        )

    def test_write_down_and_redemption_signs(self) -> None:
        self.assertAlmostEqual(write_down_cashflow(7.5), -7.5)
        self.assertAlmostEqual(redemption_cashflow(33.0), 33.0)

    def test_sequential_amortization_uses_asof_band_overlap(self) -> None:
        attach, detach = scheduled_tranche_band(
            total_stack_sched=35.0,
            total_stack_asof=100.0,
            attachment_point=0.20,
            detachment_point=0.40,
            tranche_amortization_mode="SEQUENTIAL",
        )
        self.assertAlmostEqual(attach, 20.0)
        self.assertAlmostEqual(detach, 35.0)
        self.assertAlmostEqual(
            scheduled_tranche_notional(
                total_stack_sched=35.0,
                total_stack_asof=100.0,
                attachment_point=0.20,
                detachment_point=0.40,
                tranche_amortization_mode="SEQUENTIAL",
            ),
            15.0,
        )

    def test_sequential_amortization_can_refill_back_to_asof_band(self) -> None:
        dipped = scheduled_tranche_notional(
            total_stack_sched=35.0,
            total_stack_asof=100.0,
            attachment_point=0.20,
            detachment_point=0.40,
            tranche_amortization_mode="SEQUENTIAL",
        )
        refilled = scheduled_tranche_notional(
            total_stack_sched=38.0,
            total_stack_asof=100.0,
            attachment_point=0.20,
            detachment_point=0.40,
            tranche_amortization_mode="SEQUENTIAL",
        )
        self.assertAlmostEqual(dipped, 15.0)
        self.assertAlmostEqual(refilled, 18.0)

    def test_piecewise_premium_split_at_default_event(self) -> None:
        start = date(2025, 1, 1)
        end = date(2025, 4, 1)
        events = [date(2025, 2, 1)]

        def n_tr_at(d: date) -> float:
            return 100.0 if d < date(2025, 2, 1) else 80.0

        prem = premium_accrual_piecewise(
            period_start=start,
            period_end=end,
            event_dates_in_period=events,
            n_tr_at_start_of_date=n_tr_at,
            spread=0.10,
            premium_day_count="ACT/360",
        )
        expected = 0.10 * ((31.0 / 360.0) * 100.0 + (59.0 / 360.0) * 80.0)
        self.assertAlmostEqual(prem, expected, places=12)


if __name__ == "__main__":
    unittest.main()
