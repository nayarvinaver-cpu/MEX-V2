from __future__ import annotations

import unittest

import numpy as np

import srt_model_config as cfg
from srt_model.config import ConfigValidationError
from srt_model.credit.copula import simulate_uniforms_one_factor, validate_rho
from srt_model.credit.default_times import generate_default_time_years
from srt_model.curves.survival_adapter import load_survival_curve_set_for_currency


class TestCopula(unittest.TestCase):
    def test_rho_validation(self) -> None:
        validate_rho(0.0)
        validate_rho(0.5)
        with self.assertRaises(ConfigValidationError):
            validate_rho(1.0)
        with self.assertRaises(ConfigValidationError):
            validate_rho(-0.1)

    def test_simulation_reproducibility(self) -> None:
        u1 = simulate_uniforms_one_factor(n_paths=100, n_obligors=5, rho=0.1, seed=42)
        u2 = simulate_uniforms_one_factor(n_paths=100, n_obligors=5, rho=0.1, seed=42)
        self.assertTrue(np.allclose(u1, u2))
        self.assertTrue(np.all((u1 > 0.0) & (u1 <= 1.0)))


class TestDefaultTimes(unittest.TestCase):
    def test_generate_default_time_years_shape(self) -> None:
        curves = load_survival_curve_set_for_currency("EUR", cfg)
        u = np.array(
            [
                [0.99, 0.95, 0.90],
                [0.70, 0.50, 0.30],
            ],
            dtype=float,
        )
        keys = ["AA", "BBB", "CCC"]
        tau = generate_default_time_years(u_matrix=u, debtor_curve_keys=keys, curves=curves)
        self.assertEqual(tau.shape, u.shape)

    def test_default_time_monotonicity_in_u(self) -> None:
        curves = load_survival_curve_set_for_currency("EUR", cfg)
        u = np.array([[0.95], [0.90], [0.80]], dtype=float)
        tau = generate_default_time_years(u_matrix=u, debtor_curve_keys=["BB"], curves=curves).flatten()
        # Lower U implies larger default time when solving S(t)=U.
        self.assertGreater(tau[1], tau[0])
        self.assertGreater(tau[2], tau[1])


if __name__ == "__main__":
    unittest.main()

