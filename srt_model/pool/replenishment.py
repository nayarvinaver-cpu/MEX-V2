from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd

from srt_model.io.portfolio import LoanRecord
from srt_model.pool.ead import projected_balance_with_prepayment


REPLENISHMENT_MODE_SCALAR_TOPUP = "SCALAR_TOPUP"
REPLENISHMENT_MODE_VINTAGE_LOANS = "VINTAGE_LOANS"
_DAYS_IN_YEAR = 365.25
_SMALL_GROUP_THRESHOLD = 5_000_000.0


@dataclass(frozen=True)
class ReplenishmentResult:
    pool_balance_sched_by_date: dict[date, float]
    stop_event_date: date | None
    stop_event_reason: str | None


@dataclass(frozen=True)
class _SyntheticVintage:
    start_date: date
    initial_amount: float
    maturity_date: date
    pd_1y: float
    wal_years: float


@dataclass(frozen=True)
class _OriginalPoolState:
    total_balance: float
    pd_numerator: float
    wal_numerator: float
    group_balances: dict[str, float]
    group_worst_internal: dict[str, float]
    industry_balances: dict[str, float]
    small_group_balance: float


@dataclass(frozen=True)
class _SyntheticPoolState:
    total_balance: float
    pd_numerator: float
    wal_numerator: float


@dataclass
class _SyntheticPoolTracker:
    _active_by_maturity: list[tuple[date, int, _SyntheticVintage]]
    _next_seq: int = 0
    total_balance: float = 0.0
    pd_numerator: float = 0.0
    wal_numerator: float = 0.0

    def advance_to(self, t: date) -> None:
        while self._active_by_maturity and self._active_by_maturity[0][0] <= t:
            _, _, vintage = heapq.heappop(self._active_by_maturity)
            self.total_balance -= vintage.initial_amount
            self.pd_numerator -= vintage.initial_amount * vintage.pd_1y
            self.wal_numerator -= vintage.initial_amount * vintage.wal_years

    def add_vintage(self, vintage: _SyntheticVintage) -> None:
        if vintage.initial_amount <= 0.0:
            return
        self.total_balance += vintage.initial_amount
        self.pd_numerator += vintage.initial_amount * vintage.pd_1y
        self.wal_numerator += vintage.initial_amount * vintage.wal_years
        heapq.heappush(
            self._active_by_maturity,
            (vintage.maturity_date, self._next_seq, vintage),
        )
        self._next_seq += 1

    def snapshot(self) -> _SyntheticPoolState:
        return _SyntheticPoolState(
            total_balance=float(max(0.0, self.total_balance)),
            pd_numerator=float(max(0.0, self.pd_numerator)),
            wal_numerator=float(max(0.0, self.wal_numerator)),
        )


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


def _cfg_replenishment_mode(cfg) -> str:
    raw_value = getattr(cfg, "REPLENISHMENT_MODE", None)
    if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
        raise ValueError("REPLENISHMENT_MODE must be set in srt_model_config.py.")
    raw = str(raw_value).strip().upper()
    aliases = {
        "TOPUP": REPLENISHMENT_MODE_SCALAR_TOPUP,
        "POOL_TOPUP": REPLENISHMENT_MODE_SCALAR_TOPUP,
        "LEGACY": REPLENISHMENT_MODE_SCALAR_TOPUP,
        "VINTAGE": REPLENISHMENT_MODE_VINTAGE_LOANS,
        "VINTAGE_LOAN": REPLENISHMENT_MODE_VINTAGE_LOANS,
        "NOTIONAL_LOAN": REPLENISHMENT_MODE_VINTAGE_LOANS,
    }
    mode = aliases.get(raw, raw)
    if mode not in {REPLENISHMENT_MODE_SCALAR_TOPUP, REPLENISHMENT_MODE_VINTAGE_LOANS}:
        raise ValueError(
            f"Unsupported REPLENISHMENT_MODE '{raw}'. "
            f"Allowed: {REPLENISHMENT_MODE_SCALAR_TOPUP}, {REPLENISHMENT_MODE_VINTAGE_LOANS}."
        )
    return mode


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


def _pd_numerator(loans_by_id: dict[str, LoanRecord], balances: dict[str, float]) -> float:
    return float(sum(balances[lid] * loans_by_id[lid].pd_1y for lid in balances.keys()))


def _non_defaulted_balances(
    loans_by_id: dict[str, LoanRecord],
    balances: dict[str, float],
    t: date,
    debtor_default_event_date: dict[str, date],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lid, bal in balances.items():
        if bal <= 0.0:
            continue
        default_event = debtor_default_event_date.get(loans_by_id[lid].debtor_id)
        if default_event is not None and t >= default_event:
            continue
        out[lid] = float(bal)
    return out


def _group_share_cap(cfg, worst_internal: float) -> float:
    if worst_internal <= 3.8:
        return _cfg_float(cfg, "GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX", 1.0)
    return _cfg_float(cfg, "GUIDELINE_DEBTOR_CONC_R40_AND_WORSE_MAX", 1.0)


def _moodys_guidelines_pass(
    cfg,
    industry_balances: dict[str, float],
    total_balance: float,
) -> bool:
    if total_balance <= 0.0:
        return True
    shares = sorted((bal / total_balance for bal in industry_balances.values() if bal > 0.0), reverse=True)
    if shares and shares[0] > _cfg_float(cfg, "GUIDELINE_MOODYS_LARGEST_GROUP_MAX", 1.0):
        return False
    for share in shares[1:4]:
        if share > _cfg_float(cfg, "GUIDELINE_MOODYS_2_TO_4_MAX", 1.0):
            return False
    for share in shares[4:]:
        if share > _cfg_float(cfg, "GUIDELINE_MOODYS_OTHER_MAX", 1.0):
            return False
    return True


def _portfolio_guidelines_pass(
    cfg,
    loans_by_id: dict[str, LoanRecord],
    balances_non_defaulted: dict[str, float],
) -> bool:
    """Evaluate whole-portfolio guidelines on non-defaulted balances."""
    active = {lid: bal for lid, bal in balances_non_defaulted.items() if bal > 0.0}
    total = sum(active.values())
    if total <= 0.0:
        return True

    group_balances: dict[str, float] = {}
    group_worst_internal: dict[str, float] = {}
    industry_balances: dict[str, float] = {}
    wal_numerator = 0.0

    for lid, bal in active.items():
        loan = loans_by_id[lid]
        group_id = loan.debtor_group_id
        group_balances[group_id] = group_balances.get(group_id, 0.0) + bal
        group_worst_internal[group_id] = max(
            group_worst_internal.get(group_id, -1e9),
            loan.internal_rating_value,
        )
        industry = str(loan.moodys_industry)
        industry_balances[industry] = industry_balances.get(industry, 0.0) + bal
        wal_numerator += bal * loan.wal_years

    for group_id, bal in group_balances.items():
        if bal / total > _group_share_cap(cfg, group_worst_internal[group_id]):
            return False

    if not _moodys_guidelines_pass(cfg, industry_balances, total):
        return False

    small_groups = sum(v for v in group_balances.values() if v < _SMALL_GROUP_THRESHOLD)
    if small_groups / total > _cfg_float(cfg, "GUIDELINE_DEBTOR_GROUPS_LT_5M_MAX", 1.0):
        return False

    wal = wal_numerator / total
    if wal > _cfg_float(cfg, "GUIDELINE_WAL_REPLENISHED_POOL_MAX", 1e9):
        return False
    return True


def _build_original_pool_state(
    loans_by_id: dict[str, LoanRecord],
    balances_non_defaulted: dict[str, float],
) -> _OriginalPoolState:
    group_balances: dict[str, float] = {}
    group_worst_internal: dict[str, float] = {}
    industry_balances: dict[str, float] = {}
    total = 0.0
    pd_numerator = 0.0
    wal_numerator = 0.0

    for lid, bal in balances_non_defaulted.items():
        if bal <= 0.0:
            continue
        loan = loans_by_id[lid]
        total += bal
        pd_numerator += bal * loan.pd_1y
        wal_numerator += bal * loan.wal_years

        group_id = loan.debtor_group_id
        group_balances[group_id] = group_balances.get(group_id, 0.0) + bal
        group_worst_internal[group_id] = max(
            group_worst_internal.get(group_id, -1e9),
            loan.internal_rating_value,
        )

        industry = str(loan.moodys_industry)
        industry_balances[industry] = industry_balances.get(industry, 0.0) + bal

    small_group_balance = sum(v for v in group_balances.values() if v < _SMALL_GROUP_THRESHOLD)
    return _OriginalPoolState(
        total_balance=float(total),
        pd_numerator=float(pd_numerator),
        wal_numerator=float(wal_numerator),
        group_balances=group_balances,
        group_worst_internal=group_worst_internal,
        industry_balances=industry_balances,
        small_group_balance=float(small_group_balance),
    )


def _synthetic_vintage_balance(vintage: _SyntheticVintage, t: date) -> float:
    if vintage.initial_amount <= 0.0:
        return 0.0
    if t <= vintage.start_date:
        return float(vintage.initial_amount)
    if t >= vintage.maturity_date:
        return 0.0
    return float(vintage.initial_amount)


def _build_synthetic_pool_state(vintages: list[_SyntheticVintage], t: date) -> _SyntheticPoolState:
    total = 0.0
    pd_numerator = 0.0
    wal_numerator = 0.0
    for vintage in vintages:
        bal = _synthetic_vintage_balance(vintage, t)
        if bal <= 0.0:
            continue
        total += bal
        pd_numerator += bal * vintage.pd_1y
        wal_numerator += bal * vintage.wal_years
    return _SyntheticPoolState(
        total_balance=float(total),
        pd_numerator=float(pd_numerator),
        wal_numerator=float(wal_numerator),
    )


def _synthetic_group_cap_share(cfg) -> float:
    return _cfg_float(cfg, "GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX", 1.0)


def _minimum_synthetic_small_group_balance(
    total_balance: float,
    synthetic_total: float,
    synthetic_group_cap_share: float,
) -> float:
    if synthetic_total <= 0.0:
        return 0.0
    if synthetic_group_cap_share <= 0.0:
        return float(synthetic_total)
    max_group_balance = synthetic_group_cap_share * total_balance
    if max_group_balance < _SMALL_GROUP_THRESHOLD:
        return float(synthetic_total)
    if synthetic_total < _SMALL_GROUP_THRESHOLD:
        return float(synthetic_total)
    return 0.0


def _portfolio_guidelines_pass_with_synthetics(
    cfg,
    original_state: _OriginalPoolState,
    synthetic_state: _SyntheticPoolState,
) -> bool:
    total = original_state.total_balance + synthetic_state.total_balance
    if total <= 0.0:
        return True

    for group_id, bal in original_state.group_balances.items():
        if bal / total > _group_share_cap(cfg, original_state.group_worst_internal[group_id]):
            return False

    if not _moodys_guidelines_pass(cfg, original_state.industry_balances, total):
        return False

    synthetic_group_cap_share = _synthetic_group_cap_share(cfg)
    if synthetic_state.total_balance > 0.0 and synthetic_group_cap_share <= 0.0:
        return False

    synthetic_small_groups = _minimum_synthetic_small_group_balance(
        total_balance=total,
        synthetic_total=synthetic_state.total_balance,
        synthetic_group_cap_share=synthetic_group_cap_share,
    )
    total_small_groups = original_state.small_group_balance + synthetic_small_groups
    if total_small_groups / total > _cfg_float(cfg, "GUIDELINE_DEBTOR_GROUPS_LT_5M_MAX", 1.0):
        return False

    wal = (original_state.wal_numerator + synthetic_state.wal_numerator) / total
    if wal > _cfg_float(cfg, "GUIDELINE_WAL_REPLENISHED_POOL_MAX", 1e9):
        return False
    return True


def _candidate_synthetic_state(
    synthetic_state: _SyntheticPoolState,
    amount: float,
    pd_1y: float,
    wal_years: float,
) -> _SyntheticPoolState:
    return _SyntheticPoolState(
        total_balance=synthetic_state.total_balance + amount,
        pd_numerator=synthetic_state.pd_numerator + (amount * pd_1y),
        wal_numerator=synthetic_state.wal_numerator + (amount * wal_years),
    )


def _synthetic_small_group_switch_amount(
    total_balance_before: float,
    synthetic_total_before: float,
    synthetic_group_cap_share: float,
) -> float:
    if synthetic_group_cap_share <= 0.0:
        return float("inf")
    return max(
        (_SMALL_GROUP_THRESHOLD / synthetic_group_cap_share) - total_balance_before,
        _SMALL_GROUP_THRESHOLD - synthetic_total_before,
    )


def _max_feasible_new_vintage_amount(
    *,
    cfg,
    original_state: _OriginalPoolState,
    synthetic_state: _SyntheticPoolState,
    requested_amount: float,
    pd_1y: float,
    wal_min: float,
) -> float:
    requested = float(max(0.0, requested_amount))
    if requested <= 0.0:
        return 0.0
    if not _portfolio_guidelines_pass_with_synthetics(cfg, original_state, synthetic_state):
        return 0.0

    synthetic_group_cap_share = _synthetic_group_cap_share(cfg)
    if synthetic_group_cap_share <= 0.0:
        return 0.0

    total_before = original_state.total_balance + synthetic_state.total_balance
    wal_before = original_state.wal_numerator + synthetic_state.wal_numerator
    upper = requested

    wal_limit = _cfg_float(cfg, "GUIDELINE_WAL_REPLENISHED_POOL_MAX", 1e9)
    if wal_limit < 1e8 and wal_min > wal_limit:
        wal_headroom = (wal_limit * total_before) - wal_before
        if wal_headroom <= 0.0:
            return 0.0
        upper = min(upper, wal_headroom / (wal_min - wal_limit))

    if upper <= 0.0:
        return 0.0

    candidate_upper = _candidate_synthetic_state(synthetic_state, upper, pd_1y, wal_min)
    if _portfolio_guidelines_pass_with_synthetics(cfg, original_state, candidate_upper):
        return float(upper)

    small_group_limit = _cfg_float(cfg, "GUIDELINE_DEBTOR_GROUPS_LT_5M_MAX", 1.0)
    if not (0.0 <= small_group_limit < 1.0):
        return 0.0

    switch_amount = _synthetic_small_group_switch_amount(
        total_balance_before=total_before,
        synthetic_total_before=synthetic_state.total_balance,
        synthetic_group_cap_share=synthetic_group_cap_share,
    )
    if upper >= switch_amount:
        return 0.0

    low_regime_upper = (
        (small_group_limit * total_before)
        - original_state.small_group_balance
        - synthetic_state.total_balance
    ) / (1.0 - small_group_limit)
    if low_regime_upper <= 0.0:
        return 0.0

    issued = min(upper, low_regime_upper)
    candidate_low = _candidate_synthetic_state(synthetic_state, issued, pd_1y, wal_min)
    if _portfolio_guidelines_pass_with_synthetics(cfg, original_state, candidate_low):
        return float(issued)
    return 0.0


def _evaluate_stop_event_reason(
    *,
    cfg,
    total_non_defaulted_balance: float,
    pd_numerator: float,
    cumulative_loss: float,
    n_ref_asof: float,
) -> str | None:
    wapd = 0.0 if total_non_defaulted_balance <= 0.0 else (pd_numerator / total_non_defaulted_balance)
    if wapd > _cfg_float(cfg, "STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED", 1.0):
        return "WAPD stop event"
    if n_ref_asof > 0.0 and (cumulative_loss / n_ref_asof) > _cfg_float(
        cfg, "STOP_EVENT_CUMULATIVE_LOSS_MAX", 1.0
    ):
        return "Cumulative loss stop event"
    return None


def _build_path_pool_balance_schedule_scalar_topup(
    *,
    cfg,
    loans: list[LoanRecord],
    event_dates: list[date],
    as_of_date: date,
    replenishment_end_date: date,
    cap_amount: float,
    prepayment_date_by_loan: dict[str, date | None],
    debtor_default_event_date: dict[str, date],
    losses_by_default_event: dict[date, float],
    n_ref_asof: float,
    shared_total_balance_cache: dict[date, float] | None,
    shared_all_balance_cache: dict[date, dict[str, float]] | None,
) -> ReplenishmentResult:
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

        cumulative_loss += float(losses_by_default_event.get(t, 0.0))
        replenish_allowed = t <= replenishment_end_date and stop_date is None
        if replenish_allowed:
            non_defaulted = _non_defaulted_balances(
                loans_by_id,
                _get_all_balances(t),
                t,
                debtor_default_event_date,
            )
            stop_reason_now = _evaluate_stop_event_reason(
                cfg=cfg,
                total_non_defaulted_balance=sum(non_defaulted.values()),
                pd_numerator=_pd_numerator(loans_by_id, non_defaulted),
                cumulative_loss=cumulative_loss,
                n_ref_asof=n_ref_asof,
            )
            if stop_reason_now is not None:
                stop_date = t
                stop_reason = stop_reason_now
            elif _portfolio_guidelines_pass(cfg, loans_by_id, non_defaulted):
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


def _new_synthetic_profile_bounds(
    cfg,
    t: date,
) -> tuple[float, float, float, date, date] | None:
    new_pool_wapd = _cfg_float(cfg, "REPL_WAPD_REFERENCE_POOL_MAX", 1.0)
    previous_pool_wapd = _cfg_float(cfg, "REPL_WAPD_PREVIOUS_POOL", 1.0)
    if new_pool_wapd > previous_pool_wapd:
        return None

    max_internal = _cfg_float(cfg, "ELIGIBILITY_LOWEST_RATING_MAX_INTERNAL", 1e9)
    if max_internal < 0.0:
        return None

    min_maturity = max(
        _cfg_date(cfg, "ELIGIBILITY_FINAL_MATURITY_MIN_DATE", "1900-01-01"),
        t + timedelta(days=1),
    )
    max_maturity = _cfg_date(cfg, "ELIGIBILITY_FINAL_MATURITY_MAX_DATE", "2999-12-31")
    if min_maturity > max_maturity:
        return None

    wal_min = max((min_maturity - t).days / _DAYS_IN_YEAR, 1.0 / _DAYS_IN_YEAR)
    wal_max = max((max_maturity - t).days / _DAYS_IN_YEAR, wal_min)
    return (new_pool_wapd, wal_min, wal_max, min_maturity, max_maturity)


def _choose_new_vintage(
    *,
    cfg,
    t: date,
    original_state: _OriginalPoolState,
    synthetic_state: _SyntheticPoolState,
    requested_amount: float,
) -> _SyntheticVintage | None:
    requested = float(max(0.0, requested_amount))
    if requested <= 0.0:
        return None

    bounds = _new_synthetic_profile_bounds(cfg, t)
    if bounds is None:
        return None
    new_pool_wapd, wal_min, wal_max, min_maturity, max_maturity = bounds

    if not _portfolio_guidelines_pass_with_synthetics(cfg, original_state, synthetic_state):
        return None
    issued = _max_feasible_new_vintage_amount(
        cfg=cfg,
        original_state=original_state,
        synthetic_state=synthetic_state,
        requested_amount=requested,
        pd_1y=new_pool_wapd,
        wal_min=wal_min,
    )

    if issued <= 0.0:
        return None

    total_after = original_state.total_balance + synthetic_state.total_balance + issued
    wal_limit = _cfg_float(cfg, "GUIDELINE_WAL_REPLENISHED_POOL_MAX", 1e9)
    wal_bound = wal_max
    if wal_limit < 1e8:
        wal_bound = (
            (wal_limit * total_after)
            - original_state.wal_numerator
            - synthetic_state.wal_numerator
        ) / issued
    wal_years = min(wal_max, max(wal_min, wal_bound))

    maturity_days = max((min_maturity - t).days, int(wal_years * _DAYS_IN_YEAR))
    maturity_date = min(t + timedelta(days=max(1, maturity_days)), max_maturity)

    return _SyntheticVintage(
        start_date=t,
        initial_amount=float(issued),
        maturity_date=maturity_date,
        pd_1y=float(new_pool_wapd),
        wal_years=float(wal_years),
    )


def _build_path_pool_balance_schedule_vintage_loans(
    *,
    cfg,
    loans: list[LoanRecord],
    event_dates: list[date],
    as_of_date: date,
    replenishment_end_date: date,
    cap_amount: float,
    prepayment_date_by_loan: dict[str, date | None],
    debtor_default_event_date: dict[str, date],
    losses_by_default_event: dict[date, float],
    n_ref_asof: float,
    shared_total_balance_cache: dict[date, float] | None,
    shared_all_balance_cache: dict[date, dict[str, float]] | None,
) -> ReplenishmentResult:
    loans_by_id = {l.loan_id: l for l in loans}
    pool_sched: dict[date, float] = {}
    synthetic_tracker = _SyntheticPoolTracker(_active_by_maturity=[])
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

    for t in event_dates:
        synthetic_tracker.advance_to(t)
        base_bal = _get_total_balance(t)
        synthetic_state = synthetic_tracker.snapshot()

        cumulative_loss += float(losses_by_default_event.get(t, 0.0))
        replenish_allowed = t <= replenishment_end_date and stop_date is None
        if replenish_allowed:
            original_non_defaulted = _non_defaulted_balances(
                loans_by_id,
                _get_all_balances(t),
                t,
                debtor_default_event_date,
            )
            original_state = _build_original_pool_state(loans_by_id, original_non_defaulted)
            stop_reason_now = _evaluate_stop_event_reason(
                cfg=cfg,
                total_non_defaulted_balance=original_state.total_balance + synthetic_state.total_balance,
                pd_numerator=original_state.pd_numerator + synthetic_state.pd_numerator,
                cumulative_loss=cumulative_loss,
                n_ref_asof=n_ref_asof,
            )
            if stop_reason_now is not None:
                stop_date = t
                stop_reason = stop_reason_now
            else:
                needed = float(cap_amount) - (base_bal + synthetic_state.total_balance)
                vintage = _choose_new_vintage(
                    cfg=cfg,
                    t=t,
                    original_state=original_state,
                    synthetic_state=synthetic_state,
                    requested_amount=needed,
                )
                if vintage is not None:
                    synthetic_tracker.add_vintage(vintage)
                    synthetic_state = synthetic_tracker.snapshot()

        pool_sched[t] = float(base_bal + synthetic_state.total_balance)

    return ReplenishmentResult(
        pool_balance_sched_by_date=pool_sched,
        stop_event_date=stop_date,
        stop_event_reason=stop_reason,
    )


def build_path_pool_balance_schedule(
    *,
    cfg,
    loans: list[LoanRecord],
    event_dates: list[date],
    as_of_date: date,
    replenishment_end_date: date,
    cap_amount: float,
    prepayment_date_by_loan: dict[str, date | None],
    debtor_default_event_date: dict[str, date],
    losses_by_default_event: dict[date, float],
    n_ref_asof: float,
    shared_total_balance_cache: dict[date, float] | None = None,
    shared_all_balance_cache: dict[date, dict[str, float]] | None = None,
) -> ReplenishmentResult:
    """Build scheduled pool balance over event dates with replenishment mechanics.

    Spec 155/159: replenish amortization/maturity (and prepayment if enabled), not defaults.
    Spec 147/152: stop event checks use non-defaulted WAPD and cumulative loss threshold.
    """
    mode = _cfg_replenishment_mode(cfg)
    if mode == REPLENISHMENT_MODE_VINTAGE_LOANS:
        return _build_path_pool_balance_schedule_vintage_loans(
            cfg=cfg,
            loans=loans,
            event_dates=event_dates,
            as_of_date=as_of_date,
            replenishment_end_date=replenishment_end_date,
            cap_amount=cap_amount,
            prepayment_date_by_loan=prepayment_date_by_loan,
            debtor_default_event_date=debtor_default_event_date,
            losses_by_default_event=losses_by_default_event,
            n_ref_asof=n_ref_asof,
            shared_total_balance_cache=shared_total_balance_cache,
            shared_all_balance_cache=shared_all_balance_cache,
        )
    return _build_path_pool_balance_schedule_scalar_topup(
        cfg=cfg,
        loans=loans,
        event_dates=event_dates,
        as_of_date=as_of_date,
        replenishment_end_date=replenishment_end_date,
        cap_amount=cap_amount,
        prepayment_date_by_loan=prepayment_date_by_loan,
        debtor_default_event_date=debtor_default_event_date,
        losses_by_default_event=losses_by_default_event,
        n_ref_asof=n_ref_asof,
        shared_total_balance_cache=shared_total_balance_cache,
        shared_all_balance_cache=shared_all_balance_cache,
    )
