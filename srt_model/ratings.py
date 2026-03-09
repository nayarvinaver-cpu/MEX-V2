from __future__ import annotations

from typing import Mapping

from srt_model.config import ConfigValidationError


def normalize_internal_rating_key(value: object) -> str:
    """Normalize internal numeric ratings to one-decimal string keys."""
    text = str(value).strip()
    if text == "":
        raise ConfigValidationError("Internal_Rating is empty.")
    try:
        return f"{float(text):.1f}"
    except ValueError as exc:
        raise ConfigValidationError(f"Invalid Internal_Rating value '{value}'.") from exc


def map_internal_to_external_rating(
    internal_rating: object,
    rating_mapping: Mapping[str, str],
) -> str:
    """Map tape internal rating to external rating bucket.

    Spec 101: exact-match mapping key required.
    Spec 104: missing mapped rating in table is a hard error.
    """
    key = normalize_internal_rating_key(internal_rating)
    if key not in rating_mapping:
        raise ConfigValidationError(
            f"Internal_Rating '{key}' not found in INTERNAL_TO_EXTERNAL_RATING mapping."
        )
    return str(rating_mapping[key]).strip().upper()


def normalize_survival_lookup_rating(external_rating: str) -> str:
    """Normalize rating labels for survival-curve lookup.

    User rule: remove +/- suffixes for lookup.
    """
    rating = external_rating.strip().upper().replace(" ", "")
    if rating.endswith("+") or rating.endswith("-"):
        rating = rating[:-1]
    return rating


def collapse_to_survival_bucket(lookup_rating: str) -> str:
    """Collapse unsupported low-grade notches into available curve buckets.

    User rule: CC/C/D should use CCC survival bucket.
    """
    rating = normalize_survival_lookup_rating(lookup_rating)
    if rating in {"CC", "C", "D"}:
        return "CCC"
    return rating
