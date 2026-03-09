from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import srt_model_config as cfg
from srt_model.curves.discount_adapter import load_discount_curve_adapter
from srt_model.curves.survival_adapter import load_survival_curve_set_for_currency


class TestSurvivalCurveAdapter(unittest.TestCase):
    def test_load_and_exact_tenor_survival_match(self) -> None:
        curves = load_survival_curve_set_for_currency("EUR", cfg)
        self.assertIn("BBB", curves.supported_ratings())

        raw = pd.read_csv(cfg.SURVIVAL_PROBS_EUR_PATH).set_index("Rating")
        expected = float(raw.loc["BBB", "1.0"])
        actual = curves.survival("BBB", 1.0)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_inverse_round_trip(self) -> None:
        curves = load_survival_curve_set_for_currency("EUR", cfg)
        u = 0.93
        tau = curves.inverse_default_time_years("BB", u)
        self.assertTrue(np.isfinite(tau))
        u_back = curves.survival("BB", tau)
        self.assertAlmostEqual(u_back, u, places=10)

    def test_ccc_collapse_for_lookup(self) -> None:
        curves = load_survival_curve_set_for_currency("EUR", cfg)
        tau_d = curves.inverse_default_time_years("D", 0.90)
        tau_ccc = curves.inverse_default_time_years("CCC", 0.90)
        self.assertAlmostEqual(tau_d, tau_ccc, places=12)


class TestDiscountCurveAdapter(unittest.TestCase):
    def test_eur_discount_adapter(self) -> None:
        adapter = load_discount_curve_adapter("EUR", cfg)
        df = adapter.df("2030-01-30")
        self.assertGreater(df, 0.0)
        self.assertLessEqual(df, 1.5)

    def test_usd_discount_adapter(self) -> None:
        adapter = load_discount_curve_adapter("USD", cfg)
        df = adapter.df("2030-01-30")
        self.assertGreater(df, 0.0)
        self.assertLessEqual(df, 1.5)


if __name__ == "__main__":
    unittest.main()

