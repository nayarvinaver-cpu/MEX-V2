from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from srt_model.config import (
    ConfigValidationError,
    load_and_validate_config,
    normalize_default_timing_mode,
    resolve_calendar_selection,
    resolve_tranche_band_points,
)
from srt_model.credit.copula import simulate_uniforms_one_factor
from srt_model.credit.default_times import generate_default_time_years
from srt_model.curves.discount_adapter import DiscountCurveAdapter, load_discount_curve_adapter
from srt_model.curves.survival_adapter import SurvivalCurveSet, load_survival_curve_set_for_currency
from srt_model.io.portfolio import (
    LoanRecord,
    build_debtor_curve_keys,
    build_loan_records,
    validate_debtor_curve_coverage,
)
from srt_model.io.tape_loader import load_portfolio_tape, validate_portfolio_currency


@dataclass(frozen=True)
class PreparedInputs:
    config: Any
    as_of_date: date
    currency: str
    loans: list[LoanRecord]
    debtor_ids: list[str]
    debtor_curve_keys: list[str]
    survival_curves: SurvivalCurveSet
    discount_curve: DiscountCurveAdapter


def _parse_as_of_date(cfg: Any) -> date:
    ts = pd.to_datetime(cfg.AS_OF_DATE, errors="coerce")
    if pd.isna(ts):
        raise ConfigValidationError(f"Invalid AS_OF_DATE: {cfg.AS_OF_DATE}")
    return ts.date()


def build_prepared_inputs_from_cfg(cfg: Any) -> PreparedInputs:
    """Prepare validated model inputs from a config object.

    Spec 104/109/110: fail fast on missing/invalid required data.
    """
    resolve_calendar_selection(cfg)
    resolve_tranche_band_points(cfg)
    normalize_default_timing_mode(getattr(cfg, "DEFAULT_TIMING_MODE", None))
    as_of = _parse_as_of_date(cfg)
    tape = load_portfolio_tape(cfg.PORTFOLIO_TAPE_PATH, cfg.PORTFOLIO_SHEET_NAME)
    currency = validate_portfolio_currency(tape)
    loans = build_loan_records(
        tape_df=tape,
        as_of_date=as_of,
        rating_mapping=cfg.INTERNAL_TO_EXTERNAL_RATING,
        expected_currency=currency,
    )
    survival_curves = load_survival_curve_set_for_currency(currency, cfg)
    discount_curve = load_discount_curve_adapter(currency, cfg)
    debtor_curve_map = build_debtor_curve_keys(loans)
    validate_debtor_curve_coverage(debtor_curve_map, survival_curves.supported_ratings())

    debtor_ids = sorted(debtor_curve_map.keys())
    debtor_curve_keys = [debtor_curve_map[d] for d in debtor_ids]
    return PreparedInputs(
        config=cfg,
        as_of_date=as_of,
        currency=currency,
        loans=loans,
        debtor_ids=debtor_ids,
        debtor_curve_keys=debtor_curve_keys,
        survival_curves=survival_curves,
        discount_curve=discount_curve,
    )


def build_prepared_inputs_from_module(module_name: str = "srt_model_config") -> PreparedInputs:
    cfg = load_and_validate_config(module_name=module_name)
    return build_prepared_inputs_from_cfg(cfg)


def simulate_default_time_matrix(prepared: PreparedInputs) -> np.ndarray:
    """Run one-factor copula + inversion to produce default times (years)."""
    u = simulate_uniforms_one_factor(
        n_paths=int(prepared.config.NUM_SIMULATIONS),
        n_obligors=len(prepared.debtor_ids),
        rho=float(prepared.config.RHO),
        seed=int(prepared.config.RANDOM_SEED),
    )
    return generate_default_time_years(
        u_matrix=u,
        debtor_curve_keys=prepared.debtor_curve_keys,
        curves=prepared.survival_curves,
    )
