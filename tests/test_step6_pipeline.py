from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from srt_model.config import ConfigValidationError
from srt_model.pipeline import (
    build_prepared_inputs_from_cfg,
    build_prepared_inputs_from_module,
    simulate_default_time_matrix,
)


def _integration_cfg(tape_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        PORTFOLIO_TAPE_PATH=tape_path,
        PORTFOLIO_SHEET_NAME="Portfolio",
        AS_OF_DATE="2025-12-31",
        FIRST_PAYMENT_DATE="2026-03-31",
        ACCRUAL_START_DATE="2025-12-31",
        ACCRUAL_END_DATE="2033-01-31",
        PREMIUM_SPREAD=0.05,
        PROTECTION_START_DATE="2025-12-31",
        PROTECTION_END_DATE="2033-01-31",
        LEGAL_FINAL_MATURITY_DATE="2033-07-29",
        REPLENISHMENT_END_DATE="2026-02-15",
        REPLENISHMENT_MODE="SCALAR_TOPUP",
        PREMIUM_DAY_COUNT="ACT/360",
        ISSUER_COUNTRY="GERMANY",
        JOINT_CALENDARS_ENABLED=False,
        JOINT_CALENDAR_COUNTRIES=[],
        CURRENT_TRANCHE_VALUE=61.0,
        CURRENT_SRT_TOTAL_VALUE=1000.0,
        TRANCHE_AMORTIZATION_MODE="PRO_RATA",
        OUR_PERCENTAGE=0.30,
        INTERNAL_TO_EXTERNAL_RATING={
            "1.2": "AA+",
            "1.4": "AA",
            "1.6": "AA-",
            "1.8": "A",
            "2.0": "A-",
            "2.2": "BBB+",
            "2.4": "BBB",
            "2.6": "BBB",
            "2.8": "BBB-",
            "3.0": "BB+",
            "3.2": "BB",
            "3.4": "BB",
            "3.6": "BB-",
            "3.8": "B+",
            "4.0": "B+",
            "4.2": "B",
            "4.4": "B",
            "4.6": "B",
            "4.8": "B-",
            "5.0": "CCC+",
            "5.2": "CCC",
            "5.4": "CCC-",
            "5.6": "CC",
            "5.8": "C",
            "6.2": "D",
            "6.3": "D",
            "6.4": "D",
            "6.5": "D",
        },
        SURVIVAL_PROBS_EUR_PATH="bootstrapped_survival_probs_EUR.csv",
        SURVIVAL_PROBS_USD_PATH="bootstrapped_survival_probs_USD.csv",
        HAZARD_RATES_EUR_PATH="bootstrapped_hazard_rates_EUR.csv",
        HAZARD_RATES_USD_PATH="bootstrapped_hazard_rates_USD.csv",
        DISCOUNT_CURVE_EUR_FILE="eur_quotes.xlsx",
        DISCOUNT_CURVE_USD_FILE="usd_quotes.xlsx",
        NUM_SIMULATIONS=32,
        RHO=0.1,
        RANDOM_SEED=42,
    )


class TestPipeline(unittest.TestCase):
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
                    "Maturity_Date": "2029-06-30",
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

    def test_prepare_inputs_from_cfg(self) -> None:
        prepared = build_prepared_inputs_from_cfg(_integration_cfg(self._tmp.name))
        self.assertEqual(prepared.currency, "EUR")
        self.assertGreater(len(prepared.loans), 0)
        self.assertGreater(len(prepared.debtor_ids), 0)

    def test_default_time_matrix_shape(self) -> None:
        prepared = build_prepared_inputs_from_cfg(_integration_cfg(self._tmp.name))
        tau = simulate_default_time_matrix(prepared)
        self.assertEqual(tau.shape, (32, len(prepared.debtor_ids)))

    def test_module_loader_fails_until_user_fills_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            module_name = "tmp_missing_required_cfg"
            mod_path = Path(td) / f"{module_name}.py"
            mod_path.write_text(
                "\n".join(
                    [
                        'PORTFOLIO_TAPE_PATH = "dummy.xlsx"',
                        'PORTFOLIO_SHEET_NAME = "Portfolio"',
                        'AS_OF_DATE = "2025-12-31"',
                        'FIRST_PAYMENT_DATE = "2026-03-31"',
                        'ACCRUAL_START_DATE = "2025-12-31"',
                        'ACCRUAL_END_DATE = "2033-01-31"',
                        'PREMIUM_SPREAD = 0.05',
                        'PROTECTION_START_DATE = ""',  # intentionally missing
                        'PROTECTION_END_DATE = "2033-01-31"',
                        'LEGAL_FINAL_MATURITY_DATE = "2033-07-29"',
                        'REPLENISHMENT_END_DATE = "2026-02-15"',
                        'REPLENISHMENT_MODE = "SCALAR_TOPUP"',
                        'PREMIUM_DAY_COUNT = "ACT/360"',
                        'ISSUER_COUNTRY = "GERMANY"',
                        "JOINT_CALENDARS_ENABLED = False",
                        "JOINT_CALENDAR_COUNTRIES = []",
                        "CURRENT_TRANCHE_VALUE = 61.0",
                        "CURRENT_SRT_TOTAL_VALUE = 1000.0",
                        'TRANCHE_AMORTIZATION_MODE = "PRO_RATA"',
                        "OUR_PERCENTAGE = 0.3",
                    ]
                )
            )
            sys.path.insert(0, td)
            try:
                importlib.invalidate_caches()
                with self.assertRaises(ConfigValidationError):
                    build_prepared_inputs_from_module(module_name)
            finally:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                sys.path.remove(td)


if __name__ == "__main__":
    unittest.main()
