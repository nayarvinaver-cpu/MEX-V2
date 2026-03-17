from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from srt_model.pipeline import build_prepared_inputs_from_cfg
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
        CURRENT_TRANCHE_VALUE=61.0,
        CURRENT_SRT_TOTAL_VALUE=1000.0,
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
        self.assertTrue(result.dirty_price == result.dirty_price)  # not NaN

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
        self.assertAlmostEqual(parallel.accrued_premium, sequential.accrued_premium, places=10)
        self.assertAlmostEqual(parallel.clean_price, sequential.clean_price, places=10)
        self.assertAlmostEqual(parallel.dirty_price, sequential.dirty_price, places=10)
        self.assertAlmostEqual(parallel.pv01, sequential.pv01, places=10)
        self.assertAlmostEqual(parallel.par_spread, sequential.par_spread, places=10)
        self.assertAlmostEqual(parallel.expected_loss, sequential.expected_loss, places=10)
        self.assertAlmostEqual(parallel.var99_loss, sequential.var99_loss, places=10)
        self.assertAlmostEqual(parallel.es99_loss, sequential.es99_loss, places=10)
        assert_frame_equal(parallel.reconciliation_table, sequential.reconciliation_table)


if __name__ == "__main__":
    unittest.main()
