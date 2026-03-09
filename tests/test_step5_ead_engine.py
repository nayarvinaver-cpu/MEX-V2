from __future__ import annotations

from datetime import date
import unittest

from srt_model.io.portfolio import LoanRecord
from srt_model.pool.ead import normalize_amortisation_type, projected_balance


def _loan(amortisation_type: str, maturity_date: date, principal: float = 100.0) -> LoanRecord:
    return LoanRecord(
        loan_id="L1",
        debtor_id="D1",
        debtor_group_id="G1",
        currency="EUR",
        internal_rating="2.2",
        internal_rating_value=2.2,
        external_rating="BBB+",
        survival_lookup_rating="BBB",
        pd_1y=0.02,
        turnover_amount=125_000_000.0,
        country="Germany",
        moodys_industry="105",
        wal_years=1.25,
        maturity_date=maturity_date,
        amortisation_type=amortisation_type,
        outstanding_principal=principal,
        lgd_reg=0.4,
        lgd_econ=0.35,
    )


class TestEadEngine(unittest.TestCase):
    def test_amortisation_mapping(self) -> None:
        self.assertEqual(normalize_amortisation_type("Bullet"), "bullet")
        self.assertEqual(normalize_amortisation_type("n/a"), "bullet")
        self.assertEqual(normalize_amortisation_type("until further notice"), "bullet")
        self.assertEqual(normalize_amortisation_type("Amortisation"), "linear")
        self.assertEqual(normalize_amortisation_type("Annuity"), "linear")

    def test_bullet_balance(self) -> None:
        as_of = date(2025, 1, 1)
        loan = _loan("Bullet", date(2026, 1, 1), 100.0)
        self.assertAlmostEqual(projected_balance(loan, date(2025, 6, 1), as_of), 100.0)
        self.assertAlmostEqual(projected_balance(loan, date(2026, 1, 1), as_of), 0.0)

    def test_linear_balance(self) -> None:
        as_of = date(2025, 1, 1)
        maturity = date(2026, 1, 1)
        loan = _loan("Amortisation", maturity, 120.0)
        half = date(2025, 7, 2)
        bal_half = projected_balance(loan, half, as_of)
        self.assertGreater(bal_half, 0.0)
        self.assertLess(bal_half, 120.0)
        self.assertAlmostEqual(projected_balance(loan, maturity, as_of), 0.0)


if __name__ == "__main__":
    unittest.main()
