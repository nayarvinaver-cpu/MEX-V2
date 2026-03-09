from __future__ import annotations

from datetime import date
from types import SimpleNamespace
import unittest

import numpy as np

from srt_model.io.portfolio import LoanRecord
from srt_model.pool.prepayment import quarterly_prepayment_probability, simulate_prepayment_dates
from srt_model.pool.replenishment import build_path_pool_balance_schedule


def _loan(
    loan_id: str,
    debtor_id: str,
    principal: float,
    maturity: date,
    internal_rating: float = 2.2,
) -> LoanRecord:
    return LoanRecord(
        loan_id=loan_id,
        debtor_id=debtor_id,
        debtor_group_id=f"G-{debtor_id}",
        currency="EUR",
        internal_rating=f"{internal_rating:.1f}",
        internal_rating_value=float(internal_rating),
        external_rating="BBB",
        survival_lookup_rating="BBB",
        pd_1y=0.02,
        turnover_amount=125_000_000.0,
        country="Germany",
        moodys_industry="105",
        wal_years=1.0,
        maturity_date=maturity,
        amortisation_type="Bullet",
        outstanding_principal=principal,
        lgd_reg=0.4,
        lgd_econ=0.35,
    )


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
    def test_topup_to_cap_when_active(self) -> None:
        cfg = SimpleNamespace(
            STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED=1.0,
            STOP_EVENT_CUMULATIVE_LOSS_MAX=1.0,
            ELIGIBILITY_FINAL_MATURITY_MAX_DATE="2035-01-01",
            ELIGIBILITY_FINAL_MATURITY_MIN_DATE="2020-01-01",
            ELIGIBILITY_LOWEST_RATING_MAX_INTERNAL=10.0,
            ELIGIBILITY_LOWEST_DEBTOR_TURNOVER_MIN=0.0,
            GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX=1.0,
            GUIDELINE_DEBTOR_CONC_R40_AND_WORSE_MAX=1.0,
            GUIDELINE_MOODYS_LARGEST_GROUP_MAX=1.0,
            GUIDELINE_MOODYS_2_TO_4_MAX=1.0,
            GUIDELINE_MOODYS_OTHER_MAX=1.0,
            GUIDELINE_COUNTRY_GERMANY_MIN=0.0,
            GUIDELINE_COUNTRY_OTHER_MAX=1.0,
            GUIDELINE_DEBTOR_GROUPS_LT_5M_MAX=1.0,
            GUIDELINE_WAL_REPLENISHED_POOL_MAX=10.0,
        )
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
            debtor_notice_date={},
            losses_by_notice={},
            n_ref_asof=100.0,
        )
        self.assertGreaterEqual(out.pool_balance_sched_by_date[date(2025, 12, 31)], 119.99)
        self.assertGreaterEqual(out.pool_balance_sched_by_date[date(2026, 3, 31)], 119.99)


if __name__ == "__main__":
    unittest.main()

