from __future__ import annotations

from datetime import date
import multiprocessing as mp
import os
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from srt_model.config import resolve_calendar_selection
from srt_model.pipeline import build_prepared_inputs_from_cfg
from srt_model.pool.ead import ead_at_default
import srt_model.pv.pricing as pricing_mod
from srt_model.pv.pricing import price_prepared_inputs


def _cfg_for_pricing(tape_path: str, **overrides) -> SimpleNamespace:
    base = dict(
        PORTFOLIO_TAPE_PATH=tape_path,
        PORTFOLIO_SHEET_NAME="Portfolio",
        AS_OF_DATE="2025-12-31",
        FIRST_PAYMENT_DATE="2026-03-31",
        ACCRUAL_START_DATE="2025-12-31",
        ACCRUAL_END_DATE="2027-12-31",
        PREMIUM_SPREAD=0.05,
        PROTECTION_START_DATE="2025-12-31",
        PROTECTION_END_DATE="2027-12-31",
        LEGAL_FINAL_MATURITY_DATE="2028-06-30",
        REPLENISHMENT_END_DATE="2026-02-15",
        REPLENISHMENT_MODE="SCALAR_TOPUP",
        EOM_ON=False,
        PREMIUM_DAY_COUNT="ACT/360",
        ISSUER_COUNTRY="GERMANY",
        JOINT_CALENDARS_ENABLED=False,
        JOINT_CALENDAR_COUNTRIES=[],
        ATTACHMENT_POINT=0.0,
        DETACHMENT_POINT=0.061,
        DEFAULT_TIMING_MODE="CONTINUOUS",
        TRANCHE_AMORTIZATION_MODE="PRO_RATA",
        OUR_PERCENTAGE=0.30,
        INTERNAL_TO_EXTERNAL_RATING={
            "2.2": "BBB+",
            "3.8": "B+",
            "6.3": "D",
        },
        SURVIVAL_PROBS_EUR_PATH="bootstrapped_survival_probs_EUR.csv",
        SURVIVAL_PROBS_USD_PATH="bootstrapped_survival_probs_USD.csv",
        HAZARD_RATES_EUR_PATH="bootstrapped_hazard_rates_EUR.csv",
        HAZARD_RATES_USD_PATH="bootstrapped_hazard_rates_USD.csv",
        DISCOUNT_CURVE_EUR_FILE="eur_quotes.xlsx",
        DISCOUNT_CURVE_USD_FILE="usd_quotes.xlsx",
        NUM_SIMULATIONS=64,
        PRICING_NUM_WORKERS=1,
        RHO=0.1,
        RANDOM_SEED=42,
        TAU_YEAR_BASIS_DAYS=365.25,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestPricingEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        self._tmp.close()
        df = pd.DataFrame(
            [
                {
                    "Reference_Claim_ID": "L1",
                    "Debtor_ID": "D1",
                    "Debtor_Group_ID": "G1",
                    "Loan_Currency": "EUR",
                    "Internal_Rating": "2.2",
                    "PD": 0.02,
                    "Turnover": "125m - 250m",
                    "Country": "Germany",
                    "Moodys_Industry": "105",
                    "WAL": 1.25,
                    "Maturity_Date": "2028-12-31",
                    "Amortisation_Type": "Bullet",
                    "Outstanding_Principal_Amount": 100.0,
                    "LGD_reg": 0.40,
                    "LGD_econ": 0.35,
                },
                {
                    "Reference_Claim_ID": "L2",
                    "Debtor_ID": "D2",
                    "Debtor_Group_ID": "G2",
                    "Loan_Currency": "EUR",
                    "Internal_Rating": "3.8",
                    "PD": 0.03,
                    "Turnover": "50m - 125m",
                    "Country": "Germany",
                    "Moodys_Industry": "106",
                    "WAL": 1.50,
                    "Maturity_Date": "2027-06-30",
                    "Amortisation_Type": "Amortisation",
                    "Outstanding_Principal_Amount": 150.0,
                    "LGD_reg": 0.45,
                    "LGD_econ": 0.38,
                },
            ]
        )
        with pd.ExcelWriter(self._tmp.name, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Portfolio", index=False)

    def tearDown(self) -> None:
        if os.path.exists(self._tmp.name):
            os.remove(self._tmp.name)

    def test_price_prepared_inputs_smoke(self) -> None:
        prepared = build_prepared_inputs_from_cfg(_cfg_for_pricing(self._tmp.name))
        result = price_prepared_inputs(prepared)
        self.assertEqual(result.n_paths, 64)
        self.assertEqual(result.n_obligors, 2)
        self.assertGreaterEqual(result.tranche_notional_asof_full, 0.0)
        self.assertGreaterEqual(result.tranche_notional_asof_ours, 0.0)
        self.assertTrue(result.clean_price == result.clean_price)  # not NaN

    def test_attachment_and_detachment_define_selected_tranche_thickness(self) -> None:
        prepared = build_prepared_inputs_from_cfg(
            _cfg_for_pricing(self._tmp.name, ATTACHMENT_POINT=0.20, DETACHMENT_POINT=0.40)
        )
        result = price_prepared_inputs(prepared)
        self.assertAlmostEqual(result.tranche_notional_asof_full, 50.0, places=10)
        self.assertAlmostEqual(result.tranche_notional_asof_ours, 15.0, places=10)

    def test_parallel_pricing_matches_sequential(self) -> None:
        if "fork" not in mp.get_all_start_methods():
            self.skipTest("Parallel pricing requires fork support.")

        prepared_seq = build_prepared_inputs_from_cfg(_cfg_for_pricing(self._tmp.name))
        prepared_par = build_prepared_inputs_from_cfg(
            _cfg_for_pricing(self._tmp.name, PRICING_NUM_WORKERS=2)
        )

        sequential = price_prepared_inputs(prepared_seq)
        parallel = price_prepared_inputs(prepared_par)

        self.assertAlmostEqual(parallel.pv_premium, sequential.pv_premium, places=10)
        self.assertAlmostEqual(parallel.pv_write_down, sequential.pv_write_down, places=10)
        self.assertAlmostEqual(parallel.pv_redemption, sequential.pv_redemption, places=10)
        self.assertAlmostEqual(parallel.npv_mtm, sequential.npv_mtm, places=10)
        self.assertAlmostEqual(parallel.clean_price, sequential.clean_price, places=10)
        self.assertAlmostEqual(parallel.pv01, sequential.pv01, places=10)
        self.assertAlmostEqual(parallel.par_spread, sequential.par_spread, places=10)
        self.assertAlmostEqual(parallel.expected_loss, sequential.expected_loss, places=10)
        self.assertAlmostEqual(parallel.var99_loss, sequential.var99_loss, places=10)
        self.assertAlmostEqual(parallel.es99_loss, sequential.es99_loss, places=10)
        assert_frame_equal(parallel.reconciliation_table, sequential.reconciliation_table)

    def test_sequential_amortization_keeps_selected_tranche_flat_until_stack_falls_below_band(self) -> None:
        common = dict(
            PROTECTION_START_DATE="2035-01-01",
            PROTECTION_END_DATE="2035-12-31",
            NUM_SIMULATIONS=8,
        )
        prepared_pro = build_prepared_inputs_from_cfg(
            _cfg_for_pricing(self._tmp.name, TRANCHE_AMORTIZATION_MODE="PRO_RATA", **common)
        )
        prepared_seq = build_prepared_inputs_from_cfg(
            _cfg_for_pricing(self._tmp.name, TRANCHE_AMORTIZATION_MODE="SEQUENTIAL", **common)
        )

        pro_rata = price_prepared_inputs(prepared_pro)
        sequential = price_prepared_inputs(prepared_seq)

        self.assertAlmostEqual(sequential.tranche_notional_asof_full, pro_rata.tranche_notional_asof_full)
        self.assertAlmostEqual(pro_rata.pv_write_down, 0.0, places=12)
        self.assertAlmostEqual(sequential.pv_write_down, 0.0, places=12)
        self.assertGreater(sequential.pv_premium, pro_rata.pv_premium)
        self.assertGreater(sequential.pv_redemption, pro_rata.pv_redemption)

    def test_quarterly_midpoint_uses_initial_stub_midpoint_and_midpoint_ead(self) -> None:
        cfg = _cfg_for_pricing(self._tmp.name, DEFAULT_TIMING_MODE="QUARTERLY_MIDPOINT")
        prepared = build_prepared_inputs_from_cfg(cfg)
        selection = resolve_calendar_selection(cfg)
        dates = pricing_mod._ValuationDates(
            as_of=prepared.as_of_date,
            accrual_start=prepared.as_of_date,
            accrual_end=pd.Timestamp(cfg.ACCRUAL_END_DATE).date(),
            protection_start=pd.Timestamp(cfg.PROTECTION_START_DATE).date(),
            protection_end=pd.Timestamp(cfg.PROTECTION_END_DATE).date(),
            legal_final=pd.Timestamp(cfg.LEGAL_FINAL_MATURITY_DATE).date(),
            first_payment=pd.Timestamp(cfg.FIRST_PAYMENT_DATE).date(),
        )
        payment_dates = tuple(
            pd.Timestamp(d).date()
            for d in [
                "2026-03-31",
                "2026-06-30",
                "2026-09-30",
                "2026-12-31",
                "2027-03-31",
                "2027-06-30",
                "2027-09-30",
                "2027-12-31",
            ]
        )
        boundaries = pricing_mod._build_default_timing_period_boundaries(dates, payment_dates)
        effective_default_date, event_date = pricing_mod._effective_default_and_event_dates(
            raw_default_date=date(2026, 1, 10),
            default_timing_mode="QUARTERLY_MIDPOINT",
            period_boundaries=boundaries,
            calendar_selection=selection,
        )
        self.assertEqual(effective_default_date, date(2026, 2, 16))
        self.assertEqual(event_date, date(2026, 2, 16))

        loan = next(loan for loan in prepared.loans if loan.loan_id == "L2")
        losses, debtor_events = pricing_mod._path_losses_by_default_event(
            debtor_loans={"D2": [loan]},
            debtor_ids=["D2"],
            tau_years_row=np.array([10.0 / 365.25]),
            dates=dates,
            calendar_selection=selection,
            default_timing_mode="QUARTERLY_MIDPOINT",
            default_timing_period_boundaries=boundaries,
            prepayment_date_by_loan={loan.loan_id: None},
            basis_days=365.25,
        )
        midpoint_loss = ead_at_default(
            loan,
            tau_date=date(2026, 2, 16),
            as_of_date=prepared.as_of_date,
            prepayment_date=None,
        ) * max(float(loan.lgd_econ), float(loan.lgd_reg))
        continuous_loss = ead_at_default(
            loan,
            tau_date=date(2026, 1, 10),
            as_of_date=prepared.as_of_date,
            prepayment_date=None,
        ) * max(float(loan.lgd_econ), float(loan.lgd_reg))

        self.assertEqual(debtor_events, {"D2": date(2026, 2, 16)})
        self.assertAlmostEqual(losses[date(2026, 2, 16)], midpoint_loss, places=10)
        self.assertLess(midpoint_loss, continuous_loss)

    def test_quarterly_midpoint_assigns_boundary_defaults_to_prior_period(self) -> None:
        cfg = _cfg_for_pricing(self._tmp.name, DEFAULT_TIMING_MODE="QUARTERLY_MIDPOINT")
        dates = pricing_mod._ValuationDates(
            as_of=date(2025, 12, 31),
            accrual_start=date(2025, 12, 31),
            accrual_end=date(2027, 12, 31),
            protection_start=date(2025, 12, 31),
            protection_end=date(2027, 12, 31),
            legal_final=date(2028, 6, 30),
            first_payment=date(2026, 3, 31),
        )
        selection = resolve_calendar_selection(cfg)
        boundaries = pricing_mod._build_default_timing_period_boundaries(
            dates,
            (date(2026, 3, 31), date(2026, 6, 30)),
        )
        effective_default_date, event_date = pricing_mod._effective_default_and_event_dates(
            raw_default_date=date(2026, 3, 31),
            default_timing_mode="QUARTERLY_MIDPOINT",
            period_boundaries=boundaries,
            calendar_selection=selection,
        )
        self.assertEqual(effective_default_date, date(2026, 2, 16))
        self.assertEqual(event_date, date(2026, 2, 16))


if __name__ == "__main__":
    unittest.main()
