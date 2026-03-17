from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from srt_model.config import (
    ConfigValidationError,
    load_and_validate_config,
    resolve_calendar_selection,
    validate_required_config_fields,
)
from srt_model.io.tape_loader import validate_portfolio_currency
from srt_model.ratings import (
    collapse_to_survival_bucket,
    map_internal_to_external_rating,
    normalize_survival_lookup_rating,
)


def _valid_cfg_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        PORTFOLIO_TAPE_PATH="dummy.xlsx",
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
        ISSUER_COUNTRY="Germany",
        CURRENT_TRANCHE_VALUE=61.0,
        CURRENT_SRT_TOTAL_VALUE=1000.0,
        OUR_PERCENTAGE=0.30,
        JOINT_CALENDARS_ENABLED=False,
        JOINT_CALENDAR_COUNTRIES=[],
    )


class TestConfigValidation(unittest.TestCase):
    def test_missing_required_field_raises(self) -> None:
        cfg = _valid_cfg_namespace()
        cfg.FIRST_PAYMENT_DATE = ""
        with self.assertRaises(ConfigValidationError):
            validate_required_config_fields(cfg)

    def test_calendar_resolution_issuer_country(self) -> None:
        cfg = _valid_cfg_namespace()
        selection = resolve_calendar_selection(cfg)
        self.assertEqual(selection.base_calendar, "GERMANY")
        self.assertFalse(selection.joint_enabled)

    def test_calendar_resolution_joint(self) -> None:
        cfg = _valid_cfg_namespace()
        cfg.JOINT_CALENDARS_ENABLED = True
        cfg.JOINT_CALENDAR_COUNTRIES = ["Germany", "UK"]
        selection = resolve_calendar_selection(cfg)
        self.assertTrue(selection.joint_enabled)
        self.assertEqual(selection.joint_calendars, ("GERMANY", "UK"))

    def test_load_and_validate_current_user_config_fails_on_empty_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            module_name = "tmp_missing_required_cfg_step1"
            mod_path = Path(td) / f"{module_name}.py"
            mod_path.write_text(
                "\n".join(
                    [
                        'PORTFOLIO_TAPE_PATH = "dummy.xlsx"',
                        'PORTFOLIO_SHEET_NAME = "Portfolio"',
                        'AS_OF_DATE = "2025-12-31"',
                        'FIRST_PAYMENT_DATE = ""',  # intentionally missing
                        'ACCRUAL_START_DATE = "2025-12-31"',
                        'ACCRUAL_END_DATE = "2033-01-31"',
                        'PREMIUM_SPREAD = 0.05',
                        'PROTECTION_START_DATE = "2025-12-31"',
                        'PROTECTION_END_DATE = "2033-01-31"',
                        'LEGAL_FINAL_MATURITY_DATE = "2033-07-29"',
                        'REPLENISHMENT_END_DATE = "2026-02-15"',
                        'REPLENISHMENT_MODE = "SCALAR_TOPUP"',
                        'PREMIUM_DAY_COUNT = "ACT/360"',
                        'ISSUER_COUNTRY = "GERMANY"',
                        "CURRENT_TRANCHE_VALUE = 61.0",
                        "CURRENT_SRT_TOTAL_VALUE = 1000.0",
                        "OUR_PERCENTAGE = 0.3",
                        "JOINT_CALENDARS_ENABLED = False",
                        "JOINT_CALENDAR_COUNTRIES = []",
                    ]
                )
            )
            sys.path.insert(0, td)
            try:
                importlib.invalidate_caches()
                with self.assertRaises(ConfigValidationError):
                    load_and_validate_config(module_name)
            finally:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                sys.path.remove(td)


class TestRatings(unittest.TestCase):
    def test_mapping_uses_user_table(self) -> None:
        mapping = {"6.3": "D", "2.2": "BBB+"}
        self.assertEqual(map_internal_to_external_rating("6.3", mapping), "D")
        self.assertEqual(map_internal_to_external_rating(2.2, mapping), "BBB+")

    def test_survival_lookup_removes_plus_minus(self) -> None:
        self.assertEqual(normalize_survival_lookup_rating("BB+"), "BB")
        self.assertEqual(normalize_survival_lookup_rating("BBB-"), "BBB")
        self.assertEqual(normalize_survival_lookup_rating("AA"), "AA")

    def test_survival_lookup_collapses_cc_c_d_to_ccc(self) -> None:
        self.assertEqual(collapse_to_survival_bucket("CC"), "CCC")
        self.assertEqual(collapse_to_survival_bucket("C"), "CCC")
        self.assertEqual(collapse_to_survival_bucket("D"), "CCC")


class TestTapeCurrencyValidation(unittest.TestCase):
    def test_single_currency_passes(self) -> None:
        df = pd.DataFrame({"Loan_Currency": ["eur", "EUR", " EUR "]})
        self.assertEqual(validate_portfolio_currency(df), "EUR")

    def test_mixed_currency_raises(self) -> None:
        df = pd.DataFrame({"Loan_Currency": ["EUR", "USD"]})
        with self.assertRaises(ConfigValidationError):
            validate_portfolio_currency(df)


if __name__ == "__main__":
    unittest.main()
