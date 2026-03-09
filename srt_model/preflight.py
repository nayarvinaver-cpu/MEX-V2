from __future__ import annotations

from srt_model.config import load_and_validate_config
from srt_model.io.tape_loader import load_portfolio_tape, validate_portfolio_currency


def run_preflight_checks() -> str:
    """Run initial config+tape checks and return single deal currency.

    Spec 61 + user decision: fail fast on mixed currencies (no FX in v1).
    Spec 104/110/118: required user inputs are hard-required.
    """
    cfg = load_and_validate_config("srt_model_config")
    tape = load_portfolio_tape(cfg.PORTFOLIO_TAPE_PATH, cfg.PORTFOLIO_SHEET_NAME)
    return validate_portfolio_currency(tape)

