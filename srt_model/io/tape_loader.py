from __future__ import annotations

import pandas as pd

from srt_model.config import ConfigValidationError


def load_portfolio_tape(path: str, sheet_name: str = "Portfolio") -> pd.DataFrame:
    """Load the authoritative portfolio tape sheet."""
    return pd.read_excel(path, sheet_name=sheet_name)


def validate_portfolio_currency(df: pd.DataFrame) -> str:
    """Validate tape currency support and return single deal currency.

    Spec 61 + user decision: no FX conversion in v1, mixed currencies hard error.
    """
    if "Loan_Currency" not in df.columns:
        raise ConfigValidationError("Missing required column 'Loan_Currency' in portfolio tape.")

    currencies = sorted({str(c).strip().upper() for c in df["Loan_Currency"].dropna().tolist() if str(c).strip()})
    if not currencies:
        raise ConfigValidationError("Loan_Currency has no valid values.")
    if len(currencies) > 1:
        # Future extension: replace this hard error with currency-split valuation + FX layer.
        raise ConfigValidationError(
            f"Mixed loan currencies are not supported in v1: {currencies}"
        )
    ccy = currencies[0]
    if ccy not in {"EUR", "USD"}:
        raise ConfigValidationError(f"Unsupported deal currency '{ccy}'. Expected EUR or USD.")
    return ccy

