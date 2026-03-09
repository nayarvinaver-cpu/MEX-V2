from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable, Mapping

import pandas as pd

from srt_model.config import ConfigValidationError
from srt_model.ratings import collapse_to_survival_bucket, map_internal_to_external_rating


RATING_WORST_TO_BEST = [
    "D",
    "C",
    "CC",
    "CCC-",
    "CCC",
    "CCC+",
    "B-",
    "B",
    "B+",
    "BB-",
    "BB",
    "BB+",
    "BBB-",
    "BBB",
    "BBB+",
    "A-",
    "A",
    "A+",
    "AA-",
    "AA",
    "AA+",
    "AAA",
]
_RATING_RANK = {rating: idx for idx, rating in enumerate(RATING_WORST_TO_BEST)}


@dataclass(frozen=True)
class LoanRecord:
    loan_id: str
    debtor_id: str
    debtor_group_id: str
    currency: str
    internal_rating: str
    internal_rating_value: float
    external_rating: str
    survival_lookup_rating: str
    pd_1y: float
    turnover_amount: float
    country: str
    moodys_industry: str
    wal_years: float
    maturity_date: date
    amortisation_type: str
    outstanding_principal: float
    lgd_reg: float
    lgd_econ: float


REQUIRED_TAPE_COLUMNS = {
    "Reference_Claim_ID",
    "Debtor_ID",
    "Debtor_Group_ID",
    "Loan_Currency",
    "Internal_Rating",
    "PD",
    "Turnover",
    "Country",
    "Moodys_Industry",
    "WAL",
    "Maturity_Date",
    "Amortisation_Type",
    "Outstanding_Principal_Amount",
    "LGD_reg",
    "LGD_econ",
}


def _as_date(value: object, field_name: str) -> date:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ConfigValidationError(f"Invalid or missing {field_name}: {value}")
    return ts.date()


def _as_float(value: object, field_name: str) -> float:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        raise ConfigValidationError(f"Invalid or missing numeric field {field_name}: {value}")
    return float(val)


def _as_str(value: object, field_name: str) -> str:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        raise ConfigValidationError(f"Invalid or missing text field {field_name}.")
    return text


def _as_amortisation_type(value: object) -> str:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        # User rule: blank amortisation type is treated as bullet for all loan types.
        return "Bullet"
    return text


def _check_probability(value: float, field_name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ConfigValidationError(f"{field_name} must be in [0,1], got {value}.")


def _parse_turnover_to_amount(value: object) -> float:
    """Parse tape turnover labels into a conservative numeric lower-bound amount."""
    text = str(value).strip().lower().replace(",", "")
    if text == "" or text == "nan":
        raise ConfigValidationError("Invalid or missing Turnover value.")

    def _mult(unit: str, default: float = 1.0) -> float:
        if unit == "b":
            return 1_000_000_000.0
        if unit == "m":
            return 1_000_000.0
        return default

    # Tape buckets are expected as:
    # - "less than 50m" (or occasionally "less than 50")
    # - "125m - 250m"
    # - "more than 500m"
    m_less = re.match(r"^less than\s+([0-9]+(?:\.[0-9]+)?)\s*([mb]?)$", text)
    if m_less:
        # Conservative lower bound of "<X" bucket is zero.
        return 0.0

    m_more = re.match(r"^more than\s+([0-9]+(?:\.[0-9]+)?)\s*([mb]?)$", text)
    if m_more:
        num = float(m_more.group(1))
        unit = m_more.group(2)
        # If no unit is provided in bucket labels, treat as millions by tape convention.
        return num * _mult(unit, default=1_000_000.0)

    m_range = re.match(
        r"^([0-9]+(?:\.[0-9]+)?)\s*([mb]?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*([mb]?)$",
        text,
    )
    if m_range:
        left = float(m_range.group(1))
        left_unit = m_range.group(2)
        right_unit = m_range.group(4)
        # If only one side has unit, reuse it; fallback to millions for bucket labels.
        unit = left_unit or right_unit
        return left * _mult(unit, default=1_000_000.0)

    if text.endswith("m"):
        return float(text[:-1]) * 1_000_000.0
    if text.endswith("b"):
        return float(text[:-1]) * 1_000_000_000.0
    return _as_float(value, "Turnover")


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in sorted(REQUIRED_TAPE_COLUMNS) if c not in df.columns]
    if missing:
        raise ConfigValidationError(f"Missing required tape columns: {missing}")


def build_loan_records(
    tape_df: pd.DataFrame,
    as_of_date: date,
    rating_mapping: Mapping[str, str],
    expected_currency: str | None = None,
) -> list[LoanRecord]:
    """Normalize and validate tape rows into typed LoanRecord objects.

    Spec 82/83: exclude loans matured at/before as-of and zero-balance loans.
    Spec 104/109: missing required fields are hard errors.
    Spec 55/56: LGD_econ and LGD_reg must be in [0,1].
    """
    _validate_required_columns(tape_df)
    records: list[LoanRecord] = []
    ccy_expected = expected_currency.strip().upper() if expected_currency else None

    for idx, row in tape_df.iterrows():
        loan_id = _as_str(row["Reference_Claim_ID"], "Reference_Claim_ID")
        debtor_id = _as_str(row["Debtor_ID"], "Debtor_ID")
        debtor_group_id = _as_str(row["Debtor_Group_ID"], "Debtor_Group_ID")
        currency = _as_str(row["Loan_Currency"], "Loan_Currency").upper()
        if ccy_expected and currency != ccy_expected:
            raise ConfigValidationError(
                f"Row {idx}: Loan_Currency={currency} differs from deal currency={ccy_expected}."
            )

        maturity = _as_date(row["Maturity_Date"], "Maturity_Date")
        balance = _as_float(row["Outstanding_Principal_Amount"], "Outstanding_Principal_Amount")

        if maturity <= as_of_date:
            continue
        if balance == 0.0:
            continue

        pd_1y = _as_float(row["PD"], "PD")
        lgd_reg = _as_float(row["LGD_reg"], "LGD_reg")
        lgd_econ = _as_float(row["LGD_econ"], "LGD_econ")
        _check_probability(pd_1y, "PD")
        _check_probability(lgd_reg, "LGD_reg")
        _check_probability(lgd_econ, "LGD_econ")

        internal = _as_str(row["Internal_Rating"], "Internal_Rating")
        internal_val = _as_float(internal, "Internal_Rating")
        external = map_internal_to_external_rating(internal, rating_mapping)
        survival_lookup = collapse_to_survival_bucket(external)
        amortisation_type = _as_amortisation_type(row["Amortisation_Type"])
        turnover_amount = _parse_turnover_to_amount(row["Turnover"])
        country = _as_str(row["Country"], "Country")
        moodys_industry = _as_str(row["Moodys_Industry"], "Moodys_Industry")
        wal_years = _as_float(row["WAL"], "WAL")

        records.append(
            LoanRecord(
                loan_id=loan_id,
                debtor_id=debtor_id,
                debtor_group_id=debtor_group_id,
                currency=currency,
                internal_rating=internal,
                internal_rating_value=internal_val,
                external_rating=external,
                survival_lookup_rating=survival_lookup,
                pd_1y=pd_1y,
                turnover_amount=turnover_amount,
                country=country,
                moodys_industry=moodys_industry,
                wal_years=wal_years,
                maturity_date=maturity,
                amortisation_type=amortisation_type,
                outstanding_principal=balance,
                lgd_reg=lgd_reg,
                lgd_econ=lgd_econ,
            )
        )
    return records


def build_debtor_curve_keys(loans: Iterable[LoanRecord]) -> dict[str, str]:
    """Assign one curve key per debtor using worst loan-level rating.

    Spec 86/87/88: group by Debtor_ID and use worst-rating obligor curve.
    """
    by_debtor: dict[str, list[LoanRecord]] = {}
    for loan in loans:
        by_debtor.setdefault(loan.debtor_id, []).append(loan)

    keys: dict[str, str] = {}
    for debtor_id, debtor_loans in by_debtor.items():
        worst = min(debtor_loans, key=lambda x: _RATING_RANK.get(x.external_rating, 10_000))
        keys[debtor_id] = collapse_to_survival_bucket(worst.external_rating)
    return keys


def validate_debtor_curve_coverage(
    debtor_curve_keys: Mapping[str, str],
    supported_ratings: Iterable[str],
) -> None:
    """Hard-validate that all debtor curve keys exist in loaded survival curves."""
    supported = {str(r).strip().upper() for r in supported_ratings}
    missing = sorted({r for r in debtor_curve_keys.values() if r.upper() not in supported})
    if missing:
        raise ConfigValidationError(
            f"Mapped debtor curve ratings missing from survival curve set: {missing}"
        )
