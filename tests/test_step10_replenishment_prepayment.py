from __future__ import annotations

from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from srt_model.io.portfolio import LoanRecord
from srt_model.pool.prepayment import quarterly_prepayment_probability, simulate_prepayment_dates
import srt_model.pool.replenishment as replenishment_mod
from srt_model.pool.replenishment import (
    _SyntheticPoolTracker,
    _SyntheticVintage,
    _build_synthetic_pool_state,
    build_path_pool_balance_schedule,
)


def _loan(
    loan_id: str,
    debtor_id: str,
    principal: float,
    maturity: date,
    internal_rating: float = 2.2,
    amortisation_type: str = "Bullet",
    pd_1y: float = 0.02,
    debtor_group_id: str | None = None,
    moodys_industry: str = "105",
    wal_years: float = 1.0,
) -> LoanRecord:
    return LoanRecord(
        loan_id=loan_id,
        debtor_id=debtor_id,
        debtor_group_id=debtor_group_id or f"G-{debtor_id}",
        currency="EUR",
        internal_rating=f"{internal_rating:.1f}",
        internal_rating_value=float(internal_rating),
        external_rating="BBB",
        survival_lookup_rating="BBB",
        pd_1y=pd_1y,
        turnover_amount=125_000_000.0,
        country="Germany",
        moodys_industry=moodys_industry,
        wal_years=wal_years,
        maturity_date=maturity,
        amortisation_type=amortisation_type,
        outstanding_principal=principal,
        lgd_reg=0.4,
        lgd_econ=0.35,
    )


def _replenishment_cfg(**overrides) -> SimpleNamespace:
    base = dict(
        REPLENISHMENT_MODE="SCALAR_TOPUP",
        STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED=1.0,
        STOP_EVENT_CUMULATIVE_LOSS_MAX=1.0,
        REPL_WAPD_REFERENCE_POOL_MAX=0.01,
        REPL_WAPD_PREVIOUS_POOL=0.03,
        ELIGIBILITY_FINAL_MATURITY_MAX_DATE="2035-01-01",
        ELIGIBILITY_FINAL_MATURITY_MIN_DATE="2020-01-01",
        ELIGIBILITY_LOWEST_RATING_MAX_INTERNAL=10.0,
        GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX=1.0,
        GUIDELINE_DEBTOR_CONC_R40_AND_WORSE_MAX=1.0,
        GUIDELINE_MOODYS_LARGEST_GROUP_MAX=1.0,
        GUIDELINE_MOODYS_2_TO_4_MAX=1.0,
        GUIDELINE_MOODYS_OTHER_MAX=1.0,
        GUIDELINE_DEBTOR_GROUPS_LT_5M_MAX=1.0,
        GUIDELINE_WAL_REPLENISHED_POOL_MAX=10.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestPrepayment(unittest.TestCase):
    def test_quarterly_probability(self) -> None:
        self.assertAlmostEqual(quarterly_prepayment_probability(0.0), 0.0)
        self.assertAlmostEqual(quarterly_prepayment_probability(1.0), 1.0)

    def test_simulate_prepayment_dates_disabled(self) -> None:
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1))]
        out = simulate_prepayment_dates(
            loans=loans,
            quarter_dates=[date(2026, 3, 31), date(2026, 6, 30)],
            enable_prepayment=False,
            cpr_annual=0.2,
            rng=np.random.default_rng(42),
        )
        self.assertIsNone(out["L1"])


class TestReplenishment(unittest.TestCase):
    def test_incremental_synthetic_tracker_matches_full_rebuild(self) -> None:
        tracker = _SyntheticPoolTracker(_active_by_maturity=[])
        vintages: list[_SyntheticVintage] = []
        t0 = date(2025, 12, 31)
        t1 = date(2026, 3, 31)
        t2 = date(2026, 6, 30)
        t3 = date(2026, 9, 30)

        schedule = {
            t0: [_SyntheticVintage(t0, 20.0, t2, 0.01, 0.75)],
            t1: [_SyntheticVintage(t1, 15.0, date(2027, 3, 31), 0.02, 1.25)],
            t2: [],
            t3: [],
        }

        for t in [t0, t1, t2, t3]:
            tracker.advance_to(t)
            for vintage in schedule[t]:
                tracker.add_vintage(vintage)
                vintages.append(vintage)

            incremental = tracker.snapshot()
            rebuilt = _build_synthetic_pool_state(vintages, t)

            self.assertAlmostEqual(incremental.total_balance, rebuilt.total_balance, places=10)
            self.assertAlmostEqual(incremental.pd_numerator, rebuilt.pd_numerator, places=10)
            self.assertAlmostEqual(incremental.wal_numerator, rebuilt.wal_numerator, places=10)

    def test_topup_to_cap_when_active(self) -> None:
        cfg = _replenishment_cfg()
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1))]
        dates = [date(2025, 12, 31), date(2026, 3, 31)]
        out = build_path_pool_balance_schedule(
            cfg=cfg,
            loans=loans,
            event_dates=dates,
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 12, 31),
            cap_amount=120.0,
            prepayment_date_by_loan={"L1": None},
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        self.assertGreaterEqual(out.pool_balance_sched_by_date[date(2025, 12, 31)], 119.99)
        self.assertGreaterEqual(out.pool_balance_sched_by_date[date(2026, 3, 31)], 119.99)

    def test_vintage_mode_topup_to_cap_when_active(self) -> None:
        cfg = _replenishment_cfg(REPLENISHMENT_MODE="VINTAGE_LOANS")
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1))]
        dates = [date(2025, 12, 31), date(2026, 3, 31)]
        out = build_path_pool_balance_schedule(
            cfg=cfg,
            loans=loans,
            event_dates=dates,
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 12, 31),
            cap_amount=120.0,
            prepayment_date_by_loan={"L1": None},
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        self.assertGreaterEqual(out.pool_balance_sched_by_date[date(2025, 12, 31)], 119.99)
        self.assertGreaterEqual(out.pool_balance_sched_by_date[date(2026, 3, 31)], 119.99)

    def test_vintage_mode_keeps_issued_notional_after_base_prepays(self) -> None:
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1), amortisation_type="Bullet")]
        dates = [date(2025, 12, 31), date(2026, 3, 31), date(2026, 6, 30)]
        prepay = {"L1": date(2026, 3, 31)}

        scalar = build_path_pool_balance_schedule(
            cfg=_replenishment_cfg(REPLENISHMENT_MODE="SCALAR_TOPUP"),
            loans=loans,
            event_dates=dates,
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 3, 31),
            cap_amount=120.0,
            prepayment_date_by_loan=prepay,
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        vintage = build_path_pool_balance_schedule(
            cfg=_replenishment_cfg(REPLENISHMENT_MODE="VINTAGE_LOANS"),
            loans=loans,
            event_dates=dates,
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 3, 31),
            cap_amount=120.0,
            prepayment_date_by_loan=prepay,
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )

        self.assertAlmostEqual(scalar.pool_balance_sched_by_date[date(2026, 6, 30)], 0.0, places=8)
        self.assertGreater(vintage.pool_balance_sched_by_date[date(2026, 6, 30)], 100.0)

    def test_vintage_mode_hard_stop_is_pre_topup(self) -> None:
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1), pd_1y=0.04)]
        out = build_path_pool_balance_schedule(
            cfg=_replenishment_cfg(
                REPLENISHMENT_MODE="VINTAGE_LOANS",
                STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED=0.03,
            ),
            loans=loans,
            event_dates=[date(2025, 12, 31)],
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 12, 31),
            cap_amount=120.0,
            prepayment_date_by_loan={"L1": None},
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        self.assertAlmostEqual(out.pool_balance_sched_by_date[date(2025, 12, 31)], 100.0, places=8)
        self.assertEqual(out.stop_event_reason, "WAPD stop event")

    def test_vintage_mode_skips_original_state_rebuild_after_permanent_stop(self) -> None:
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1), pd_1y=0.04)]
        dates = [date(2025, 12, 31), date(2026, 3, 31), date(2026, 6, 30)]
        with patch(
            "srt_model.pool.replenishment._build_original_pool_state",
            wraps=replenishment_mod._build_original_pool_state,
        ) as build_state:
            out = build_path_pool_balance_schedule(
                cfg=_replenishment_cfg(
                    REPLENISHMENT_MODE="VINTAGE_LOANS",
                    STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED=0.03,
                ),
                loans=loans,
                event_dates=dates,
                as_of_date=date(2025, 12, 31),
                replenishment_end_date=date(2026, 12, 31),
                cap_amount=120.0,
                prepayment_date_by_loan={"L1": None},
                debtor_default_event_date={},
                losses_by_default_event={},
                n_ref_asof=100.0,
            )
        self.assertEqual(out.stop_event_reason, "WAPD stop event")
        self.assertEqual(build_state.call_count, 1)

    def test_vintage_mode_does_not_cure_existing_portfolio_guideline_breach(self) -> None:
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1), debtor_group_id="G1")]
        out = build_path_pool_balance_schedule(
            cfg=_replenishment_cfg(
                REPLENISHMENT_MODE="VINTAGE_LOANS",
                GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX=0.60,
            ),
            loans=loans,
            event_dates=[date(2025, 12, 31)],
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 12, 31),
            cap_amount=200.0,
            prepayment_date_by_loan={"L1": None},
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        self.assertAlmostEqual(out.pool_balance_sched_by_date[date(2025, 12, 31)], 100.0, places=8)
        self.assertIsNone(out.stop_event_reason)

    def test_vintage_mode_no_replenishment_when_new_pool_wapd_is_ineligible(self) -> None:
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1))]
        out = build_path_pool_balance_schedule(
            cfg=_replenishment_cfg(
                REPLENISHMENT_MODE="VINTAGE_LOANS",
                REPL_WAPD_REFERENCE_POOL_MAX=0.04,
                REPL_WAPD_PREVIOUS_POOL=0.03,
            ),
            loans=loans,
            event_dates=[date(2025, 12, 31)],
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 12, 31),
            cap_amount=120.0,
            prepayment_date_by_loan={"L1": None},
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        self.assertAlmostEqual(out.pool_balance_sched_by_date[date(2025, 12, 31)], 100.0, places=8)
        self.assertIsNone(out.stop_event_reason)

    def test_vintage_mode_direct_wal_bound_caps_issuance(self) -> None:
        loans = [_loan("L1", "D1", 100.0, date(2027, 1, 1), wal_years=1.0)]
        wal_min = (date(2028, 12, 31) - date(2025, 12, 31)).days / 365.25
        expected_pool = 100.0 + ((1.5 * 100.0) - 100.0) / (wal_min - 1.5)
        out = build_path_pool_balance_schedule(
            cfg=_replenishment_cfg(
                REPLENISHMENT_MODE="VINTAGE_LOANS",
                GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX=1.0,
                GUIDELINE_WAL_REPLENISHED_POOL_MAX=1.5,
                ELIGIBILITY_FINAL_MATURITY_MIN_DATE="2028-12-31",
                ELIGIBILITY_FINAL_MATURITY_MAX_DATE="2030-12-31",
            ),
            loans=loans,
            event_dates=[date(2025, 12, 31)],
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 12, 31),
            cap_amount=200.0,
            prepayment_date_by_loan={"L1": None},
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        self.assertAlmostEqual(out.pool_balance_sched_by_date[date(2025, 12, 31)], expected_pool, places=6)

    def test_prior_synthetic_vintages_count_in_later_pre_stop_wapd(self) -> None:
        loans = [
            _loan("GOOD", "D1", 50.0, date(2027, 1, 1), pd_1y=0.00),
            _loan("BAD", "D2", 50.0, date(2027, 1, 1), pd_1y=0.04),
        ]
        dates = [date(2025, 12, 31), date(2026, 3, 31)]
        out = build_path_pool_balance_schedule(
            cfg=_replenishment_cfg(
                REPLENISHMENT_MODE="VINTAGE_LOANS",
                STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED=0.03,
                REPL_WAPD_REFERENCE_POOL_MAX=0.01,
                REPL_WAPD_PREVIOUS_POOL=0.03,
            ),
            loans=loans,
            event_dates=dates,
            as_of_date=date(2025, 12, 31),
            replenishment_end_date=date(2026, 12, 31),
            cap_amount=150.0,
            prepayment_date_by_loan={"GOOD": date(2026, 3, 31), "BAD": None},
            debtor_default_event_date={},
            losses_by_default_event={},
            n_ref_asof=100.0,
        )
        self.assertIsNone(out.stop_event_reason)
        self.assertGreaterEqual(out.pool_balance_sched_by_date[date(2026, 3, 31)], 149.99)


if __name__ == "__main__":
    unittest.main()
