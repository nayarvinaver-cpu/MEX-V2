from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from srt_model.pipeline import PreparedInputs, build_prepared_inputs_from_cfg
from srt_model.pv.pricing import PricingResult, price_prepared_inputs


@dataclass(frozen=True)
class ValidationPack:
    pricing: PricingResult
    bounds: pd.DataFrame
    monotonicity: pd.DataFrame
    convergence: pd.DataFrame


def bounds_checks(pricing: PricingResult) -> pd.DataFrame:
    """Basic bounds and dead-tranche checks (Spec 175(i))."""
    rows = [
        {
            "check": "loss_le_notional",
            "passed": pricing.var99_loss <= pricing.tranche_notional_asof_ours + 1e-9,
            "value": pricing.var99_loss,
            "limit": pricing.tranche_notional_asof_ours,
        },
        {
            "check": "n_tr_non_negative",
            "passed": pricing.tranche_notional_asof_ours >= -1e-9,
            "value": pricing.tranche_notional_asof_ours,
            "limit": 0.0,
        },
        {
            "check": "clean_dirty_finite",
            "passed": (pricing.clean_price == pricing.clean_price) and (pricing.dirty_price == pricing.dirty_price),
            "value": pricing.clean_price,
            "limit": float("nan"),
        },
    ]
    return pd.DataFrame(rows)


def monotonicity_check(prepared: PreparedInputs, spread_grid: Iterable[float]) -> pd.DataFrame:
    """Check NPV monotonicity with respect to spread (Spec 175(ii))."""
    rows = []
    for spread in spread_grid:
        cfg = deepcopy(prepared.config)
        cfg.PREMIUM_SPREAD = float(spread)
        p2 = build_prepared_inputs_from_cfg(cfg)
        res = price_prepared_inputs(p2)
        rows.append({"spread": float(spread), "npv_mtm": res.npv_mtm})
    df = pd.DataFrame(rows).sort_values("spread").reset_index(drop=True)
    df["delta_npv"] = df["npv_mtm"].diff()
    df["monotone_step"] = df["delta_npv"].fillna(0.0) >= -1e-8
    return df


def convergence_check(
    prepared: PreparedInputs,
    seeds: Iterable[int],
    path_counts: Iterable[int],
) -> pd.DataFrame:
    """Convergence hooks over seeds/path counts (Spec 175(iii))."""
    rows = []
    for n_paths in path_counts:
        for seed in seeds:
            cfg = deepcopy(prepared.config)
            cfg.NUM_SIMULATIONS = int(n_paths)
            cfg.RANDOM_SEED = int(seed)
            p2 = build_prepared_inputs_from_cfg(cfg)
            res = price_prepared_inputs(p2)
            rows.append(
                {
                    "paths": int(n_paths),
                    "seed": int(seed),
                    "par_spread": res.par_spread,
                    "npv_mtm": res.npv_mtm,
                    "expected_loss": res.expected_loss,
                }
            )
    return pd.DataFrame(rows)


def build_validation_pack(
    prepared: PreparedInputs,
    spread_grid: Iterable[float] = (0.01, 0.03, 0.05, 0.07),
    seeds: Iterable[int] = (11, 42),
    path_counts: Iterable[int] = (2000, 5000),
) -> ValidationPack:
    pricing = price_prepared_inputs(prepared)
    return ValidationPack(
        pricing=pricing,
        bounds=bounds_checks(pricing),
        monotonicity=monotonicity_check(prepared, spread_grid=spread_grid),
        convergence=convergence_check(prepared, seeds=seeds, path_counts=path_counts),
    )

