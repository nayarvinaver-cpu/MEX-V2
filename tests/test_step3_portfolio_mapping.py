from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from srt_model.config import ConfigValidationError
from srt_model.io.portfolio import (
    build_debtor_curve_keys,
    build_loan_records,
    validate_debtor_curve_coverage,
)


def _base_row(**overrides):
    row = {
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
    }
    row.update(overrides)
    return row


class TestPortfolioNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = {
            "2.2": "BBB+",
            "3.8": "B+",
            "6.3": "D",
        }

    def test_build_loan_records_filters_and_maps(self) -> None:
        df = pd.DataFrame(
            [
                _base_row(Reference_Claim_ID="L1", Debtor_ID="D1", Internal_Rating="2.2"),
                _base_row(
                    Reference_Claim_ID="L2",
                    Debtor_ID="D1",
                    Internal_Rating="3.8",
                    Outstanding_Principal_Amount=200.0,
                ),
                _base_row(
                    Reference_Claim_ID="L3",
                    Debtor_ID="D2",
                    Internal_Rating="6.3",
                    Maturity_Date="2025-12-31",
                ),  # matured at as-of -> excluded
                _base_row(
                    Reference_Claim_ID="L4",
                    Debtor_ID="D3",
                    Internal_Rating="2.2",
                    Outstanding_Principal_Amount=0.0,
                ),  # zero balance -> excluded
            ]
        )
        loans = build_loan_records(
            tape_df=df,
            as_of_date=date(2025, 12, 31),
            rating_mapping=self.mapping,
            expected_currency="EUR",
        )
        self.assertEqual(len(loans), 2)
        self.assertEqual(loans[0].external_rating, "BBB+")
        self.assertEqual(loans[1].survival_lookup_rating, "B")

    def test_worst_rating_key_per_debtor(self) -> None:
        df = pd.DataFrame(
            [
                _base_row(Reference_Claim_ID="L1", Debtor_ID="D1", Internal_Rating="2.2"),
                _base_row(Reference_Claim_ID="L2", Debtor_ID="D1", Internal_Rating="3.8"),
                _base_row(Reference_Claim_ID="L3", Debtor_ID="D2", Internal_Rating="6.3"),
            ]
        )
        loans = build_loan_records(df, date(2025, 12, 31), self.mapping, expected_currency="EUR")
        keys = build_debtor_curve_keys(loans)
        self.assertEqual(keys["D1"], "B")
        self.assertEqual(keys["D2"], "CCC")  # D collapses to CCC for survival lookup

    def test_curve_coverage_validation(self) -> None:
        with self.assertRaises(ConfigValidationError):
            validate_debtor_curve_coverage({"D1": "B", "D2": "CCC"}, supported_ratings=["AA", "B"])

    def test_lgd_range_validation(self) -> None:
        df = pd.DataFrame([_base_row(LGD_econ=1.2)])
        with self.assertRaises(ConfigValidationError):
            build_loan_records(df, date(2025, 12, 31), self.mapping, expected_currency="EUR")

    def test_turnover_bucket_parsing(self) -> None:
        df = pd.DataFrame(
            [
                _base_row(Reference_Claim_ID="L1", Turnover="less than 50m"),
                _base_row(Reference_Claim_ID="L2", Debtor_ID="D2", Turnover="more than 500m"),
                _base_row(Reference_Claim_ID="L3", Debtor_ID="D3", Turnover="125m - 250m"),
            ]
        )
        loans = build_loan_records(df, date(2025, 12, 31), self.mapping, expected_currency="EUR")
        self.assertEqual(len(loans), 3)
        self.assertEqual(loans[0].turnover_amount, 0.0)
        self.assertEqual(loans[1].turnover_amount, 500_000_000.0)
        self.assertEqual(loans[2].turnover_amount, 125_000_000.0)

    def test_blank_amortisation_defaults_to_bullet(self) -> None:
        df = pd.DataFrame([_base_row(Amortisation_Type="")])
        loans = build_loan_records(df, date(2025, 12, 31), self.mapping, expected_currency="EUR")
        self.assertEqual(len(loans), 1)
        self.assertEqual(loans[0].amortisation_type, "Bullet")


if __name__ == "__main__":
    unittest.main()
