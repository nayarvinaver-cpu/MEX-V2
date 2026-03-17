from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when required user config inputs are missing or invalid."""


TRANCHE_AMORTIZATION_MODE_PRO_RATA = "PRO_RATA"
TRANCHE_AMORTIZATION_MODE_SEQUENTIAL = "SEQUENTIAL"


# Spec 13/14/15: allowed calendar set and joint-calendar behavior.
ALLOWED_CALENDAR_COUNTRIES = {
    "TARGET": "TARGET",
    "TARGET2": "TARGET",
    "SWEDEN": "SWEDEN",
    "UK": "UK",
    "GERMANY": "GERMANY",
    "FRANCE": "FRANCE",
    "SPAIN": "SPAIN",
    "SWITZERLAND": "SWITZERLAND",
}

# Spec 18/19/20/31/49/51/83/85/95: required core dates/inputs.
REQUIRED_CONFIG_FIELDS = (
    "PORTFOLIO_TAPE_PATH",
    "PORTFOLIO_SHEET_NAME",
    "AS_OF_DATE",
    "FIRST_PAYMENT_DATE",
    "ACCRUAL_START_DATE",
    "ACCRUAL_END_DATE",
    "PREMIUM_SPREAD",
    "PROTECTION_START_DATE",
    "PROTECTION_END_DATE",
    "LEGAL_FINAL_MATURITY_DATE",
    "REPLENISHMENT_END_DATE",
    "REPLENISHMENT_MODE",
    "PREMIUM_DAY_COUNT",
    "ISSUER_COUNTRY",
    "ATTACHMENT_POINT",
    "DETACHMENT_POINT",
    "OUR_PERCENTAGE",
)


@dataclass(frozen=True)
class CalendarSelection:
    base_calendar: str
    joint_enabled: bool
    joint_calendars: tuple[str, ...]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def validate_required_config_fields(cfg: Any) -> None:
    """Validate required user inputs before valuation starts.

    Spec 104/110/118: required fields missing -> hard error.
    """
    missing = [name for name in REQUIRED_CONFIG_FIELDS if _is_empty(getattr(cfg, name, None))]
    if missing:
        for field_name in missing:
            print(f"[WARNING] Required config field is empty: {field_name}")
        raise ConfigValidationError(
            "Missing required configuration values. Fill all required fields in srt_model_config.py."
        )


def resolve_calendar_selection(cfg: Any) -> CalendarSelection:
    """Build calendar selection from issuer country and joint-toggle inputs.

    Spec 13/14/15: country-based calendar with optional joint-of-N support.
    """
    issuer_country = str(cfg.ISSUER_COUNTRY).strip().upper()
    if issuer_country not in ALLOWED_CALENDAR_COUNTRIES:
        raise ConfigValidationError(
            f"Unsupported issuer country/calendar '{cfg.ISSUER_COUNTRY}'. "
            f"Allowed: {sorted(ALLOWED_CALENDAR_COUNTRIES)}"
        )

    base = ALLOWED_CALENDAR_COUNTRIES[issuer_country]
    joint_enabled = bool(cfg.JOINT_CALENDARS_ENABLED)
    if not joint_enabled:
        return CalendarSelection(base_calendar=base, joint_enabled=False, joint_calendars=())

    selected = tuple(str(c).strip().upper() for c in cfg.JOINT_CALENDAR_COUNTRIES if str(c).strip())
    if not selected:
        raise ConfigValidationError(
            "JOINT_CALENDARS_ENABLED=True but JOINT_CALENDAR_COUNTRIES is empty."
        )
    invalid = [c for c in selected if c not in ALLOWED_CALENDAR_COUNTRIES]
    if invalid:
        raise ConfigValidationError(
            f"Unsupported joint calendar country entries: {invalid}. "
            f"Allowed: {sorted(ALLOWED_CALENDAR_COUNTRIES)}"
        )
    return CalendarSelection(
        base_calendar=base,
        joint_enabled=True,
        joint_calendars=tuple(ALLOWED_CALENDAR_COUNTRIES[c] for c in selected),
    )


def normalize_tranche_amortization_mode(value: Any) -> str:
    """Normalize the selected-tranche scheduled amortization mode."""
    if _is_empty(value):
        return TRANCHE_AMORTIZATION_MODE_PRO_RATA

    raw = str(value).strip().upper()
    aliases = {
        "PRORATA": TRANCHE_AMORTIZATION_MODE_PRO_RATA,
        "PRO-RATA": TRANCHE_AMORTIZATION_MODE_PRO_RATA,
        "SEQ": TRANCHE_AMORTIZATION_MODE_SEQUENTIAL,
    }
    mode = aliases.get(raw, raw)
    if mode not in {
        TRANCHE_AMORTIZATION_MODE_PRO_RATA,
        TRANCHE_AMORTIZATION_MODE_SEQUENTIAL,
    }:
        raise ConfigValidationError(
            f"Unsupported TRANCHE_AMORTIZATION_MODE '{value}'. "
            f"Allowed: {TRANCHE_AMORTIZATION_MODE_PRO_RATA}, "
            f"{TRANCHE_AMORTIZATION_MODE_SEQUENTIAL}."
        )
    return mode


def resolve_tranche_band_points(cfg: Any) -> tuple[float, float]:
    """Parse and validate selected tranche attachment/detachment points."""
    try:
        attachment = float(getattr(cfg, "ATTACHMENT_POINT"))
        detachment = float(getattr(cfg, "DETACHMENT_POINT"))
    except (TypeError, ValueError, AttributeError):
        raise ConfigValidationError(
            "ATTACHMENT_POINT and DETACHMENT_POINT must be numeric decimals in [0,1]."
        ) from None

    if attachment < 0.0 or attachment > 1.0:
        raise ConfigValidationError("ATTACHMENT_POINT must be in [0,1].")
    if detachment < 0.0 or detachment > 1.0:
        raise ConfigValidationError("DETACHMENT_POINT must be in [0,1].")
    if detachment <= attachment:
        raise ConfigValidationError("DETACHMENT_POINT must be greater than ATTACHMENT_POINT.")
    return float(attachment), float(detachment)


def load_and_validate_config(module_name: str = "srt_model_config") -> Any:
    """Import and validate user configuration module."""
    cfg = importlib.import_module(module_name)
    validate_required_config_fields(cfg)
    resolve_calendar_selection(cfg)
    resolve_tranche_band_points(cfg)
    normalize_tranche_amortization_mode(getattr(cfg, "TRANCHE_AMORTIZATION_MODE", None))
    return cfg
