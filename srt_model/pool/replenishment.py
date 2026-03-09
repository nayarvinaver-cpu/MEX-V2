from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import pandas as pd

from srt_model.io.portfolio import LoanRecord
from srt_model.pool.ead import projected_balance_with_prepayment


@dataclass(frozen=True)
class ReplenishmentResult:
    pool_balance_sched_by_date: dict[date, float]
    stop_event_date: date | None
    stop_event_reason: str | None


def _cfg_float(cfg, key: str, default: float) -> float:
    try:
        return float(getattr(cfg, key))
    except Exception:
        return float(default)


def _cfg_date(cfg, key: str, default_iso: str) -> date:
    try:
        return pd.to_datetime(getattr(cfg, key), errors="raise").date()
    except Exception:
        return datetime.strptime(default_iso, "%Y-%m-%d").date()


def _active_non_defaulted_balances(
    loans: Iterable[LoanRecord],
    t: date,
    as_of_date: date,
    prepayment_date_by_loan: dict[str, date | None],
    debtor_notice_date: dict[str, date],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for loan in loans:
        nd = debtor_notice_date.get(loan.debtor_id)
        if nd is not None and t >= nd:
            continue
        bal = projected_balance_with_prepayment(
            loan=loan,
            t=t,
            as_of_date=as_of_date,
            prepayment_date=prepayment_date_by_loan.get(loan.loan_id),
        )
        out[loan.loan_id] = float(max(0.0, bal))
    return out


def _all_balances(
    loans: Iterable[LoanRecord],
    t: date,
    as_of_date: date,
    prepayment_date_by_loan: dict[str, date | None],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for loan in loans:
        bal = projected_balance_with_prepayment(
            loan=loan,
            t=t,
            as_of_date=as_of_date,
            prepayment_date=prepayment_date_by_loan.get(loan.loan_id),
        )
        out[loan.loan_id] = float(max(0.0, bal))
    return out


def _total_balance(
    loans: Iterable[LoanRecord],
    t: date,
    as_of_date: date,
    prepayment_date_by_loan: dict[str, date | None],
) -> float:
    total = 0.0
    for loan in loans:
        bal = projected_balance_with_prepayment(
            loan=loan,
            t=t,
            as_of_date=as_of_date,
            prepayment_date=prepayment_date_by_loan.get(loan.loan_id),
        )
        total += float(max(0.0, bal))
    return float(total)


def _weighted_average_pd(loans_by_id: dict[str, LoanRecord], balances: dict[str, float]) -> float:
    total = sum(balances.values())
    if total <= 0.0:
        return 0.0
    num = sum(balances[lid] * loans_by_id[lid].pd_1y for lid in balances.keys())
    return float(num / total)


def _guidelines_pass(
    cfg,
    loans_by_id: dict[str, LoanRecord],
    balances_non_defaulted: dict[str, float],
) -> bool:
    """Evaluate replenishment guideline limits on current non-defaulted pool."""
    active = {lid: bal for lid, bal in balances_non_defaulted.items() if bal > 0.0}
    total = sum(active.values())
    if total <= 0.0:
        return True

    active_loans = [loans_by_id[lid] for lid in active.keys()]

    # Eligibility: maturity min/max.
    max_mat = max(l.maturity_date for l in active_loans)
    min_mat = min(l.maturity_date for l in active_loans)
    if max_mat > _cfg_date(cfg, "ELIGIBILITY_FINAL_MATURITY_MAX_DATE", "2999-12-31"):
        return False
    if min_mat < _cfg_date(cfg, "ELIGIBILITY_FINAL_MATURITY_MIN_DATE", "1900-01-01"):
        return False

    # Eligibility: lowest rating / turnover.
    worst_internal = max(l.internal_rating_value for l in active_loans)
    if worst_internal > _cfg_float(cfg, "ELIGIBILITY_LOWEST_RATING_MAX_INTERNAL", 1e9):
        return False
    min_turnover = min(l.turnover_amount for l in active_loans)
    if min_turnover < _cfg_float(cfg, "ELIGIBILITY_LOWEST_DEBTOR_TURNOVER_MIN", 0.0):
        return False

    # Debtor concentration limits split by internal-rating quality.
    debtor_bal: dict[str, float] = {}
    debtor_worst_internal: dict[str, float] = {}
    for lid, bal in active.items():
        loan = loans_by_id[lid]
        debtor_bal[loan.debtor_id] = debtor_bal.get(loan.debtor_id, 0.0) + bal
        debtor_worst_internal[loan.debtor_id] = max(
            debtor_worst_internal.get(loan.debtor_id, -1e9),
            loan.internal_rating_value,
        )
    for debtor_id, bal in debtor_bal.items():
        share = bal / total
        if debtor_worst_internal[debtor_id] <= 3.8:
            if share > _cfg_float(cfg, "GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX", 1.0):
                return False
        else:
            if share > _cfg_float(cfg, "GUIDELINE_DEBTOR_CONC_R40_AND_WORSE_MAX", 1.0):
                return False

    # Moody's industry concentration.
    ind_bal: dict[str, float] = {}
    for lid, bal in active.items():
        ind = str(loans_by_id[lid].moodys_industry)
        ind_bal[ind] = ind_bal.get(ind, 0.0) + bal
    shares = sorted((x / total for x in ind_bal.values()), reverse=True)
    if shares and shares[0] > _cfg_float(cfg, "GUIDELINE_MOODYS_LARGEST_GROUP_MAX", 1.0):
        return False
    for s in shares[1:4]:
        if s > _cfg_float(cfg, "GUIDELINE_MOODYS_2_TO_4_MAX", 1.0):
            return False
    for s in shares[4:]:
        if s > _cfg_float(cfg, "GUIDELINE_MOODYS_OTHER_MAX", 1.0):
            return False

    # Country concentration.
    country_bal: dict[str, float] = {}
    for lid, bal in active.items():
        c = str(loans_by_id[lid].country).strip().upper()
        country_bal[c] = country_bal.get(c, 0.0) + bal
    germany_share = country_bal.get("GERMANY", 0.0) / total
    if germany_share < _cfg_float(cfg, "GUIDELINE_COUNTRY_GERMANY_MIN", 0.0):
        return False
    for c, bal in country_bal.items():
        if c == "GERMANY":
            continue
        if bal / total > _cfg_float(cfg, "GUIDELINE_COUNTRY_OTHER_MAX", 1.0):
            return False

    # Debtor groups with aggregate amount < 5m.
    group_bal: dict[str, float] = {}
    for lid, bal in active.items():
        grp = loans_by_id[lid].debtor_group_id
        group_bal[grp] = group_bal.get(grp, 0.0) + bal
    small_groups = sum(v for v in group_bal.values() if v < 5_000_000.0)
    if small_groups / total > _cfg_float(cfg, "GUIDELINE_DEBTOR_GROUPS_LT_5M_MAX", 1.0):
        return False

    # WAL limit.
    wal = sum(active[lid] * loans_by_id[lid].wal_years for lid in active.keys()) / total
    if wal > _cfg_float(cfg, "GUIDELINE_WAL_REPLENISHED_POOL_MAX", 1e9):
        return False
    return True


def build_path_pool_balance_schedule(
    *,
    cfg,
    loans: list[LoanRecord],
    event_dates: list[date],
    as_of_date: date,
    replenishment_end_date: date,
    cap_amount: float,
    prepayment_date_by_loan: dict[str, date | None],
    debtor_notice_date: dict[str, date],
    losses_by_notice: dict[date, float],
    n_ref_asof: float,
    shared_total_balance_cache: dict[date, float] | None = None,
    shared_all_balance_cache: dict[date, dict[str, float]] | None = None,
) -> ReplenishmentResult:
    """Build scheduled pool balance over event dates with replenishment mechanics.

    Spec 155/159: replenish amortization/maturity (and prepayment if enabled), not defaults.
    Spec 147/152: stop event checks use non-defaulted WAPD and cumulative loss threshold.
    """
    loans_by_id = {l.loan_id: l for l in loans}
    pool_sched: dict[date, float] = {}
    topup = 0.0
    stop_date: date | None = None
    stop_reason: str | None = None
    cumulative_loss = 0.0

    def _get_total_balance(t: date) -> float:
        if shared_total_balance_cache is not None and t in shared_total_balance_cache:
            return float(shared_total_balance_cache[t])
        total = _total_balance(loans, t, as_of_date, prepayment_date_by_loan)
        if shared_total_balance_cache is not None:
            shared_total_balance_cache[t] = float(total)
        return float(total)

    def _get_all_balances(t: date) -> dict[str, float]:
        if shared_all_balance_cache is not None and t in shared_all_balance_cache:
            return shared_all_balance_cache[t]
        balances = _all_balances(loans, t, as_of_date, prepayment_date_by_loan)
        if shared_all_balance_cache is not None:
            shared_all_balance_cache[t] = balances
        return balances

    prev_t = event_dates[0]
    prev_base = _get_total_balance(prev_t)
    for t in event_dates:
        base_bal = _get_total_balance(t)
        if t != prev_t:
            if prev_base > 0.0:
                runoff = max(0.0, min(1.0, base_bal / prev_base))
            else:
                runoff = 0.0
            topup *= runoff

        cumulative_loss += float(losses_by_notice.get(t, 0.0))
        replenish_allowed = t <= replenishment_end_date and stop_date is None
        if replenish_allowed:
            all_bal_t = _get_all_balances(t)
            non_def = {
                lid: bal
                for lid, bal in all_bal_t.items()
                if debtor_notice_date.get(loans_by_id[lid].debtor_id) is None
                or t < debtor_notice_date[loans_by_id[lid].debtor_id]
            }
            wapd = _weighted_average_pd(loans_by_id, non_def)
            if wapd > _cfg_float(cfg, "STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED", 1.0):
                stop_date = t
                stop_reason = "WAPD stop event"
            elif n_ref_asof > 0.0 and (cumulative_loss / n_ref_asof) > _cfg_float(
                cfg, "STOP_EVENT_CUMULATIVE_LOSS_MAX", 1.0
            ):
                stop_date = t
                stop_reason = "Cumulative loss stop event"

            if stop_date is None:
                if _guidelines_pass(cfg, loans_by_id, non_def):
                    needed = float(cap_amount) - (base_bal + topup)
                    if needed > 0.0:
                        topup += needed

        pool_sched[t] = float(base_bal + topup)
        prev_t = t
        prev_base = base_bal

    return ReplenishmentResult(
        pool_balance_sched_by_date=pool_sched,
        stop_event_date=stop_date,
        stop_event_reason=stop_reason,
    )
