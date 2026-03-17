from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

from srt_model.pipeline import build_prepared_inputs_from_cfg
from srt_model.validation.checks import (
    bounds_checks,
    build_validation_pack,
    convergence_check,
    monotonicity_check,
)


def _cfg(tape_path: str) -> SimpleNamespace:
    return SimpleNamespace(
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
        TRANCHE_AMORTIZATION_MODE="PRO_RATA",
        OUR_PERCENTAGE=0.30,
        REPLENISHMENT_CAP_AMOUNT=200.0,
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
        INTERNAL_TO_EXTERNAL_RATING={"2.2": "BBB+", "3.8": "B+"},
        SURVIVAL_PROBS_EUR_PATH="bootstrapped_survival_probs_EUR.csv",
        SURVIVAL_PROBS_USD_PATH="bootstrapped_survival_probs_USD.csv",
        HAZARD_RATES_EUR_PATH="bootstrapped_hazard_rates_EUR.csv",
        HAZARD_RATES_USD_PATH="bootstrapped_hazard_rates_USD.csv",
        DISCOUNT_CURVE_EUR_FILE="eur_quotes.xlsx",
        DISCOUNT_CURVE_USD_FILE="usd_quotes.xlsx",
        NUM_SIMULATIONS=64,
        RHO=0.1,
        RANDOM_SEED=42,
        TAU_YEAR_BASIS_DAYS=365.25,
        ENABLE_PREPAYMENT=False,
        CPR_ANNUAL=0.0,
    )


class TestValidationPack(unittest.TestCase):
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

    def test_validation_outputs_exist(self) -> None:
        prepared = build_prepared_inputs_from_cfg(_cfg(self._tmp.name))
        pack = build_validation_pack(
            prepared,
            spread_grid=(0.03, 0.05),
            seeds=(11,),
            path_counts=(32,),
        )
        self.assertFalse(pack.bounds.empty)
        self.assertFalse(pack.monotonicity.empty)
        self.assertFalse(pack.convergence.empty)

    def test_individual_checks(self) -> None:
        prepared = build_prepared_inputs_from_cfg(_cfg(self._tmp.name))
        res = build_validation_pack(prepared, spread_grid=(0.05,), seeds=(42,), path_counts=(32,)).pricing
        b = bounds_checks(res)
        self.assertIn("check", b.columns)
        m = monotonicity_check(prepared, spread_grid=(0.03, 0.05))
        self.assertIn("npv_mtm", m.columns)
        c = convergence_check(prepared, seeds=(7, 42), path_counts=(16,))
        self.assertEqual(len(c), 2)


if __name__ == "__main__":
    unittest.main()
