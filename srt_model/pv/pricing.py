from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from datetime import date, timedelta
import threading
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from srt_model.config import (
    ConfigValidationError,
    normalize_tranche_amortization_mode,
    resolve_tranche_band_points,
    resolve_calendar_selection,
)
from srt_model.grid.calendar import adjust_modified_following
from srt_model.grid.schedule import (
    build_payment_schedule,
    compute_default_event_date,
    effective_accrual_start,
)
from srt_model.grid.dates import add_months
from srt_model.pipeline import PreparedInputs, simulate_default_time_matrix
from srt_model.pool.ead import ead_at_default, pool_scheduled_balance
from srt_model.pool.prepayment import simulate_prepayment_dates
from srt_model.pool.replenishment import build_path_pool_balance_schedule
from srt_model.pv.discounting import pv_cashflows
from srt_model.pv.par_spread import solve_par_spread_closed_form
from srt_model.tranche.cashflows import (
    cumulative_tranche_loss,
    incremental_tranche_loss,
    premium_accrual_piecewise,
    redemption_cashflow,
    scheduled_tranche_band,
    scheduled_tranche_notional,
    tranche_outstanding_notional,
    write_down_cashflow,
)


@dataclass(frozen=True)
class PricingResult:
    pv_premium: float
    pv_write_down: float
    pv_redemption: float
    npv_mtm: float
    clean_price: float
    pv01: float
    par_spread: float
    expected_loss: float
    var99_loss: float
    es99_loss: float
    reconciliation_table: pd.DataFrame
    n_paths: int
    n_obligors: int
    tranche_notional_asof_full: float
    tranche_notional_asof_ours: float


@dataclass(frozen=True)
class _ValuationDates:
    as_of: date
    accrual_start: date
    accrual_end: date
    protection_start: date
    protection_end: date
    legal_final: date
    first_payment: date


@dataclass(frozen=True)
class _PathPricingContext:
    prepared: PreparedInputs
    dates: _ValuationDates
    calendar_selection: object
    start_eff: date
    payment_dates: tuple[date, ...]
    quarter_dates: tuple[date, ...]
    maturity_dates: tuple[date, ...]
    replenishment_end_date: date
    debtor_loans: dict[str, list]
    basis_days: float
    enable_prepayment: bool
    cpr_annual: float
    prepayment_none_map: dict[str, date | None]
    static_event_dates: tuple[date, ...]
    total_stack_asof: float
    attachment_point: float
    detachment_point: float
    tranche_amortization_mode: str
    our_share: float
    tau_matrix: np.ndarray


@dataclass(frozen=True)
class _PathChunkResult:
    start_idx: int
    pv_premium: np.ndarray
    pv_write_down: np.ndarray
    pv_redemption: np.ndarray
    pv01: np.ndarray
    tranche_loss: np.ndarray
    premium_sum_by_date: dict[date, float]
    write_down_sum_by_date: dict[date, float]
    redemption_sum_by_date: dict[date, float]


_PATH_PRICING_CONTEXT: _PathPricingContext | None = None


class _ProgressBar:
    """Lightweight terminal progress bar for long Monte Carlo pricing runs."""

    def __init__(
        self,
        total: int,
        enabled: bool,
        update_every: int = 0,
        width: int = 32,
        animate_spinner: bool = False,
        spinner_interval_seconds: float = 0.2,
        label: str = "Pricing paths",
    ):
        self.total = max(1, int(total))
        self.enabled = bool(enabled)
        self.width = max(10, int(width))
        self._update_every_raw = int(update_every)
        self.step = (
            self._update_every_raw
            if self._update_every_raw > 0
            else max(1, self.total // 100)  # default to roughly 1% updates
        )
        self._warmup_updates = min(10, self.total)
        self._next_tick = 0
        self._start = perf_counter()
        self._done = 0
        self._label = str(label)
        self._animate_spinner = bool(animate_spinner) and self.enabled
        self._spinner_chars = "|/-\\"
        self._spinner_idx = 0
        self._printed = False
        self._finished = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

        try:
            interval = float(spinner_interval_seconds)
        except (TypeError, ValueError):
            interval = 0.2
        self._spinner_interval_seconds = max(0.05, interval)

    def start(self) -> None:
        if not self._animate_spinner:
            return
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total = max(1, int(total))
            self.step = (
                self._update_every_raw
                if self._update_every_raw > 0
                else max(1, self.total // 100)
            )
            self._warmup_updates = min(10, self.total)
            self._next_tick = 0
            self._done = min(self._done, self.total)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._spinner_interval_seconds):
            with self._lock:
                if self._finished:
                    return
                self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
                self._render_locked(force=True)

    def _render_locked(self, force: bool = False) -> None:
        d = max(0, min(int(self._done), self.total))
        if not force and d < self._next_tick and d < self.total:
            return
        frac = d / self.total
        filled = int(self.width * frac)
        bar = "=" * filled + "-" * (self.width - filled)
        elapsed = perf_counter() - self._start
        rate = (d / elapsed) if elapsed > 0 else 0.0
        eta = ((self.total - d) / rate) if rate > 0 else float("inf")
        eta_text = f"{eta:6.1f}s" if np.isfinite(eta) else "  --.-s"
        spinner = self._spinner_chars[self._spinner_idx] if self._animate_spinner else " "
        print(
            f"\r{self._label} {spinner} [{bar}] {d:>6}/{self.total:<6} ({frac*100:5.1f}%) ETA {eta_text}",
            end="",
            flush=True,
        )
        self._printed = True
        if d < self._warmup_updates:
            # Show early movement quickly so long runs do not look frozen at startup.
            self._next_tick = d + 1
        else:
            self._next_tick = d + self.step
        if d >= self.total and not self._finished:
            print("", flush=True)
            self._finished = True

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._done = max(0, min(int(done), self.total))
            self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
            self._render_locked(force=False)

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        with self._lock:
            if not self._finished:
                self._render_locked(force=True)
                if self._printed:
                    print("", flush=True)
                self._finished = True


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_date(value: object, field_name: str) -> date:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ConfigValidationError(f"Invalid date for {field_name}: {value}")
    return ts.date()


def _tau_years_to_date(as_of_date: date, tau_years: float, basis_days: float) -> date | None:
    """Convert simulated tau in years to date-only default date.

    Spec 84/214: default time is calendar date only.
    User decision: conversion basis is config parameter TAU_YEAR_BASIS_DAYS (default 365.25).
    """
    if not np.isfinite(tau_years):
        return None
    days = int(max(0.0, float(tau_years)) * basis_days)
    return as_of_date + timedelta(days=days)


def _build_valuation_dates(prepared: PreparedInputs):
    cfg = prepared.config
    selection = resolve_calendar_selection(cfg)
    as_of = prepared.as_of_date
    accrual_start = _parse_date(cfg.ACCRUAL_START_DATE, "ACCRUAL_START_DATE")
    accrual_end = _parse_date(cfg.ACCRUAL_END_DATE, "ACCRUAL_END_DATE")
    protection_start_raw = _parse_date(cfg.PROTECTION_START_DATE, "PROTECTION_START_DATE")
    protection_end_raw = _parse_date(cfg.PROTECTION_END_DATE, "PROTECTION_END_DATE")
    legal_final_raw = _parse_date(cfg.LEGAL_FINAL_MATURITY_DATE, "LEGAL_FINAL_MATURITY_DATE")
    first_payment_raw = _parse_date(cfg.FIRST_PAYMENT_DATE, "FIRST_PAYMENT_DATE")

    # Spec 34/35/40/60: protection/legal key dates are business-day adjusted (Modified Following).
    protection_start = adjust_modified_following(protection_start_raw, selection)
    protection_end = adjust_modified_following(protection_end_raw, selection)
    legal_final = adjust_modified_following(legal_final_raw, selection)
    first_payment = adjust_modified_following(first_payment_raw, selection)
    return (
        _ValuationDates(
            as_of=as_of,
            accrual_start=accrual_start,
            accrual_end=accrual_end,
            protection_start=protection_start,
            protection_end=protection_end,
            legal_final=legal_final,
            first_payment=first_payment,
        ),
        selection,
    )


def _build_debtor_loans_map(prepared: PreparedInputs):
    out: dict[str, list] = {}
    for loan in prepared.loans:
        out.setdefault(loan.debtor_id, []).append(loan)
    return out


def _cum_loss_up_to(delta_loss_by_event_date: dict[date, float], t: date) -> float:
    return float(sum(loss for event_date, loss in delta_loss_by_event_date.items() if event_date <= t))


def _notional_at_date(
    t: date,
    pool_sched_by_date: dict[date, float],
    total_stack_asof: float,
    attachment_point: float,
    detachment_point: float,
    tranche_amortization_mode: str,
    delta_loss_by_event_date: dict[date, float],
) -> float:
    if t in pool_sched_by_date:
        pool_bal = float(pool_sched_by_date[t])
    else:
        known = sorted(pool_sched_by_date.keys())
        if t < known[0]:
            pool_bal = float(pool_sched_by_date[known[0]])
        else:
            prev = max(d for d in known if d <= t)
            pool_bal = float(pool_sched_by_date[prev])
    attach_notional, detach_notional = scheduled_tranche_band(
        total_stack_sched=pool_bal,
        total_stack_asof=total_stack_asof,
        attachment_point=attachment_point,
        detachment_point=detachment_point,
        tranche_amortization_mode=tranche_amortization_mode,
    )
    cum_loss = _cum_loss_up_to(delta_loss_by_event_date, t)
    return tranche_outstanding_notional(
        attachment_notional=attach_notional,
        detachment_notional=detach_notional,
        cumulative_portfolio_loss=cum_loss,
    )


def _path_losses_by_default_event(
    debtor_loans: dict[str, list],
    debtor_ids: list[str],
    tau_years_row: np.ndarray,
    dates: _ValuationDates,
    calendar_selection,
    prepayment_date_by_loan: dict[str, date | None],
    basis_days: float,
) -> tuple[dict[date, float], dict[str, date]]:
    losses: dict[date, float] = {}
    debtor_default_event: dict[str, date] = {}
    # Spec 99/140/262/312: coverage based on default date in [ProtStart, ProtEnd].
    for j, debtor_id in enumerate(debtor_ids):
        tau_date = _tau_years_to_date(dates.as_of, float(tau_years_row[j]), basis_days=basis_days)
        if tau_date is None:
            continue
        if tau_date < dates.protection_start or tau_date > dates.protection_end:
            continue

        default_event_date = compute_default_event_date(
            default_date=tau_date,
            calendar_selection=calendar_selection,
        )

        total_loss = 0.0
        for loan in debtor_loans.get(debtor_id, []):
            # Spec 81/204: if tau > maturity, loan is out of risk set.
            if tau_date > loan.maturity_date:
                continue
            # Spec 57/58: EAD frozen at default date.
            ead = ead_at_default(
                loan,
                tau_date=tau_date,
                as_of_date=dates.as_of,
                prepayment_date=prepayment_date_by_loan.get(loan.loan_id),
            )
            if ead <= 0.0:
                continue
            lgd = max(float(loan.lgd_econ), float(loan.lgd_reg))
            total_loss += ead * lgd

        if total_loss > 0.0:
            losses[default_event_date] = losses.get(default_event_date, 0.0) + total_loss
            if (
                debtor_id not in debtor_default_event
                or default_event_date < debtor_default_event[debtor_id]
            ):
                debtor_default_event[debtor_id] = default_event_date
    return losses, debtor_default_event


def _write_down_cashflows_from_losses(
    losses_by_event_date: dict[date, float],
    pool_sched_by_date: dict[date, float],
    total_stack_asof: float,
    attachment_point: float,
    detachment_point: float,
    tranche_amortization_mode: str,
) -> tuple[list[tuple[date, float]], float]:
    """Convert portfolio losses into tranche write-down cashflows.

    Spec 158/163: tranche loss increment is min(deltaLoss, N_tr(t-)); CF_wd is negative.
    """
    cum_portfolio_loss = 0.0
    cfs: list[tuple[date, float]] = []
    total_tranche_loss = 0.0
    for event_date in sorted(losses_by_event_date):
        delta_loss = float(losses_by_event_date[event_date])
        attach_notional, detach_notional = scheduled_tranche_band(
            total_stack_sched=float(pool_sched_by_date[event_date]),
            total_stack_asof=total_stack_asof,
            attachment_point=attachment_point,
            detachment_point=detachment_point,
            tranche_amortization_mode=tranche_amortization_mode,
        )
        delta_tr_loss = incremental_tranche_loss(
            delta_portfolio_loss=delta_loss,
            cumulative_portfolio_loss_before=cum_portfolio_loss,
            attachment_notional=attach_notional,
            detachment_notional=detach_notional,
        )
        total_tranche_loss += delta_tr_loss
        cfs.append((event_date, write_down_cashflow(delta_tr_loss)))
        cum_portfolio_loss += delta_loss
    return cfs, total_tranche_loss


def _quarter_dates(first_payment: date, legal_final: date, eom_on: bool) -> list[date]:
    out: list[date] = []
    d = first_payment
    while d <= legal_final:
        out.append(d)
        d = add_months(d, 3, eom_on=eom_on)
    return out


def _resolve_pricing_num_workers(cfg, n_paths_hint: int) -> int:
    raw = _coerce_int(getattr(cfg, "PRICING_NUM_WORKERS", 1), default=1)
    if raw == 1:
        return 1
    if raw <= 0:
        raw = max(1, os_cpu_count_or_one())
    if raw <= 1:
        return 1
    if "fork" not in mp.get_all_start_methods():
        return 1
    return max(1, min(raw, max(1, int(n_paths_hint))))


def os_cpu_count_or_one() -> int:
    count = mp.cpu_count()
    return count if count and count > 0 else 1


def _path_chunk_ranges(n_paths: int, workers: int) -> list[tuple[int, int]]:
    if workers <= 1 or n_paths <= 1:
        return [(0, n_paths)]
    # Use smaller chunks than the worker count alone so progress can advance
    # during long parallel runs instead of jumping only a few times.
    chunk_count = min(n_paths, max(workers * 8, workers))
    chunk_size = max(1, int(np.ceil(n_paths / chunk_count)))
    out: list[tuple[int, int]] = []
    start = 0
    while start < n_paths:
        end = min(n_paths, start + chunk_size)
        out.append((start, end))
        start = end
    return out


def _merge_cashflow_sums(target: dict[date, float], source: dict[date, float]) -> None:
    for cf_date, amount in source.items():
        target[cf_date] = target.get(cf_date, 0.0) + amount


def _build_path_context(
    *,
    prepared: PreparedInputs,
    dates: _ValuationDates,
    calendar_selection,
    start_eff: date,
    payment_dates: list[date],
    quarter_dates: list[date],
    maturity_dates: list[date],
    replenishment_end_date: date,
    debtor_loans: dict[str, list],
    basis_days: float,
    enable_prepayment: bool,
    cpr_annual: float,
    prepayment_none_map: dict[str, date | None],
    total_stack_asof: float,
    attachment_point: float,
    detachment_point: float,
    tranche_amortization_mode: str,
    our_share: float,
    tau_matrix: np.ndarray,
) -> _PathPricingContext:
    static_event_dates = tuple(
        sorted(
            {
                dates.as_of,
                start_eff,
                dates.accrual_end,
                dates.legal_final,
                *payment_dates,
                *quarter_dates,
                *maturity_dates,
            }
        )
    )
    return _PathPricingContext(
        prepared=prepared,
        dates=dates,
        calendar_selection=calendar_selection,
        start_eff=start_eff,
        payment_dates=tuple(payment_dates),
        quarter_dates=tuple(quarter_dates),
        maturity_dates=tuple(maturity_dates),
        replenishment_end_date=replenishment_end_date,
        debtor_loans=debtor_loans,
        basis_days=float(basis_days),
        enable_prepayment=bool(enable_prepayment),
        cpr_annual=float(cpr_annual),
        prepayment_none_map=prepayment_none_map,
        static_event_dates=static_event_dates,
        total_stack_asof=float(total_stack_asof),
        attachment_point=float(attachment_point),
        detachment_point=float(detachment_point),
        tranche_amortization_mode=str(tranche_amortization_mode),
        our_share=float(our_share),
        tau_matrix=np.asarray(tau_matrix, dtype=float),
    )


def _price_path_range(
    ctx: _PathPricingContext,
    start_idx: int,
    end_idx: int,
    progress_callback=None,
) -> _PathChunkResult:
    n_chunk = max(0, end_idx - start_idx)
    path_pv_prem = np.zeros(n_chunk, dtype=float)
    path_pv_wd = np.zeros(n_chunk, dtype=float)
    path_pv_red = np.zeros(n_chunk, dtype=float)
    path_pv01 = np.zeros(n_chunk, dtype=float)
    path_tranche_loss = np.zeros(n_chunk, dtype=float)
    prem_sum_by_date: dict[date, float] = {}
    wd_sum_by_date: dict[date, float] = {}
    red_sum_by_date: dict[date, float] = {}

    shared_total_balance_cache: dict[date, float] | None = {} if not ctx.enable_prepayment else None
    shared_all_balance_cache: dict[date, dict[str, float]] | None = {} if not ctx.enable_prepayment else None
    df = ctx.prepared.discount_curve.df
    premium_day_count = str(ctx.prepared.config.PREMIUM_DAY_COUNT)

    for offset, p in enumerate(range(start_idx, end_idx)):
        if ctx.enable_prepayment:
            rng_prepay = np.random.default_rng(int(ctx.prepared.config.RANDOM_SEED) + 1_000_000 + p)
            prepayment_date_by_loan = simulate_prepayment_dates(
                loans=ctx.prepared.loans,
                quarter_dates=list(ctx.quarter_dates),
                enable_prepayment=True,
                cpr_annual=ctx.cpr_annual,
                rng=rng_prepay,
            )
        else:
            prepayment_date_by_loan = ctx.prepayment_none_map

        losses_by_event_date, debtor_default_event = _path_losses_by_default_event(
            debtor_loans=ctx.debtor_loans,
            debtor_ids=ctx.prepared.debtor_ids,
            tau_years_row=ctx.tau_matrix[p],
            dates=ctx.dates,
            calendar_selection=ctx.calendar_selection,
            prepayment_date_by_loan=prepayment_date_by_loan,
            basis_days=ctx.basis_days,
        )
        sorted_event_dates = sorted(losses_by_event_date.keys())
        if sorted_event_dates:
            event_dates = sorted({*ctx.static_event_dates, *sorted_event_dates})
        else:
            event_dates = list(ctx.static_event_dates)

        repl_result = build_path_pool_balance_schedule(
            cfg=ctx.prepared.config,
            loans=ctx.prepared.loans,
            event_dates=event_dates,
            as_of_date=ctx.dates.as_of,
            replenishment_end_date=ctx.replenishment_end_date,
            cap_amount=float(getattr(ctx.prepared.config, "REPLENISHMENT_CAP_AMOUNT", ctx.total_stack_asof)),
            prepayment_date_by_loan=prepayment_date_by_loan,
            debtor_default_event_date=debtor_default_event,
            losses_by_default_event=losses_by_event_date,
            n_ref_asof=ctx.total_stack_asof,
            shared_total_balance_cache=shared_total_balance_cache,
            shared_all_balance_cache=shared_all_balance_cache,
        )
        pool_sched_by_date = repl_result.pool_balance_sched_by_date

        n_tr_asof = _notional_at_date(
            t=ctx.dates.as_of,
            pool_sched_by_date=pool_sched_by_date,
            total_stack_asof=ctx.total_stack_asof,
            attachment_point=ctx.attachment_point,
            detachment_point=ctx.detachment_point,
            tranche_amortization_mode=ctx.tranche_amortization_mode,
            delta_loss_by_event_date=losses_by_event_date,
        )
        if n_tr_asof <= 0.0:
            if progress_callback is not None:
                progress_callback(p + 1)
            continue

        wd_cfs_raw, path_tr_loss_full = _write_down_cashflows_from_losses(
            losses_by_event_date=losses_by_event_date,
            pool_sched_by_date=pool_sched_by_date,
            total_stack_asof=ctx.total_stack_asof,
            attachment_point=ctx.attachment_point,
            detachment_point=ctx.detachment_point,
            tranche_amortization_mode=ctx.tranche_amortization_mode,
        )
        path_tranche_loss[offset] = path_tr_loss_full * ctx.our_share
        wd_cfs = [(d, cf) for d, cf in wd_cfs_raw if d > ctx.dates.as_of]

        premium_cfs: list[tuple[date, float]] = []
        pv01_cfs: list[tuple[date, float]] = []
        prev = ctx.start_eff
        for pay_date in ctx.payment_dates:
            if pay_date <= ctx.dates.as_of:
                continue
            if pay_date <= prev:
                prev = pay_date
                continue
            period_events = [d for d in sorted_event_dates if prev < d < pay_date]

            def n_tr_at_start(d: date) -> float:
                return _notional_at_date(
                    t=d,
                    pool_sched_by_date=pool_sched_by_date,
                    total_stack_asof=ctx.total_stack_asof,
                    attachment_point=ctx.attachment_point,
                    detachment_point=ctx.detachment_point,
                    tranche_amortization_mode=ctx.tranche_amortization_mode,
                    delta_loss_by_event_date=losses_by_event_date,
                )

            prem_amt = premium_accrual_piecewise(
                period_start=prev,
                period_end=pay_date,
                event_dates_in_period=period_events,
                n_tr_at_start_of_date=n_tr_at_start,
                spread=float(ctx.prepared.config.PREMIUM_SPREAD),
                premium_day_count=premium_day_count,
            )
            pv01_amt = premium_accrual_piecewise(
                period_start=prev,
                period_end=pay_date,
                event_dates_in_period=period_events,
                n_tr_at_start_of_date=n_tr_at_start,
                spread=1.0,
                premium_day_count=premium_day_count,
            )
            premium_cfs.append((pay_date, prem_amt))
            pv01_cfs.append((pay_date, pv01_amt))
            prev = pay_date

        n_tr_lfm = _notional_at_date(
            t=ctx.dates.legal_final,
            pool_sched_by_date=pool_sched_by_date,
            total_stack_asof=ctx.total_stack_asof,
            attachment_point=ctx.attachment_point,
            detachment_point=ctx.detachment_point,
            tranche_amortization_mode=ctx.tranche_amortization_mode,
            delta_loss_by_event_date=losses_by_event_date,
        )
        red_cf = redemption_cashflow(n_tr_lfm)
        red_cfs = [(ctx.dates.legal_final, red_cf)] if ctx.dates.legal_final > ctx.dates.as_of else []

        path_pv_prem[offset] = pv_cashflows(premium_cfs, df)
        path_pv_wd[offset] = pv_cashflows(wd_cfs, df)
        path_pv_red[offset] = pv_cashflows(red_cfs, df)
        path_pv01[offset] = pv_cashflows(pv01_cfs, df)

        for cf_date, amount in premium_cfs:
            prem_sum_by_date[cf_date] = prem_sum_by_date.get(cf_date, 0.0) + amount * ctx.our_share
        for cf_date, amount in wd_cfs:
            wd_sum_by_date[cf_date] = wd_sum_by_date.get(cf_date, 0.0) + amount * ctx.our_share
        for cf_date, amount in red_cfs:
            red_sum_by_date[cf_date] = red_sum_by_date.get(cf_date, 0.0) + amount * ctx.our_share

        if progress_callback is not None:
            progress_callback(p + 1)

    return _PathChunkResult(
        start_idx=start_idx,
        pv_premium=path_pv_prem,
        pv_write_down=path_pv_wd,
        pv_redemption=path_pv_red,
        pv01=path_pv01,
        tranche_loss=path_tranche_loss,
        premium_sum_by_date=prem_sum_by_date,
        write_down_sum_by_date=wd_sum_by_date,
        redemption_sum_by_date=red_sum_by_date,
    )


def _price_path_chunk_worker(path_range: tuple[int, int]) -> _PathChunkResult:
    if _PATH_PRICING_CONTEXT is None:
        raise RuntimeError("Path pricing worker context is not initialized.")
    start_idx, end_idx = path_range
    return _price_path_range(_PATH_PRICING_CONTEXT, start_idx, end_idx)


def price_prepared_inputs(prepared: PreparedInputs) -> PricingResult:
    cfg = prepared.config
    spread = float(cfg.PREMIUM_SPREAD)
    our_share = float(cfg.OUR_PERCENTAGE)
    if our_share < 0.0 or our_share > 1.0:
        raise ConfigValidationError("OUR_PERCENTAGE must be in [0,1].")
    attachment_point, detachment_point = resolve_tranche_band_points(cfg)
    tranche_amortization_mode = normalize_tranche_amortization_mode(
        getattr(cfg, "TRANCHE_AMORTIZATION_MODE", None)
    )

    dates, calendar_selection = _build_valuation_dates(prepared)
    start_eff = effective_accrual_start(dates.as_of, dates.accrual_start)
    payment_dates = build_payment_schedule(
        first_payment_date=dates.first_payment,
        as_of_date=dates.as_of,
        accrual_start_date=dates.accrual_start,
        accrual_end_date=dates.accrual_end,
        eom_on=bool(cfg.EOM_ON),
        calendar_selection=calendar_selection,
    )
    quarter_dates = _quarter_dates(
        first_payment=dates.first_payment,
        legal_final=dates.legal_final,
        eom_on=bool(cfg.EOM_ON),
    )
    maturity_dates = [l.maturity_date for l in prepared.loans if l.maturity_date <= dates.legal_final]
    replenishment_end_date = _parse_date(cfg.REPLENISHMENT_END_DATE, "REPLENISHMENT_END_DATE")
    debtor_loans = _build_debtor_loans_map(prepared)
    basis_days = float(prepared.config.TAU_YEAR_BASIS_DAYS)
    enable_prepayment = bool(getattr(cfg, "ENABLE_PREPAYMENT", False))
    cpr_annual = float(getattr(cfg, "CPR_ANNUAL", 0.0))
    prepayment_none_map = {loan.loan_id: None for loan in prepared.loans}

    total_stack_asof = sum(loan.outstanding_principal for loan in prepared.loans)
    pool_asof = pool_scheduled_balance(prepared.loans, dates.as_of, dates.as_of)
    if pool_asof <= 0.0:
        raise ConfigValidationError("As-of scheduled pool balance must be > 0.")
    total_stack_asof = float(pool_asof)
    n_sched_asof_full = scheduled_tranche_notional(
        total_stack_sched=total_stack_asof,
        total_stack_asof=total_stack_asof,
        attachment_point=attachment_point,
        detachment_point=detachment_point,
        tranche_amortization_mode=tranche_amortization_mode,
    )

    requested_workers = _resolve_pricing_num_workers(cfg, int(getattr(cfg, "NUM_SIMULATIONS", 1)))
    progress_step = _coerce_int(getattr(cfg, "PROGRESS_UPDATE_EVERY_PATHS", 0), default=0)
    spinner_interval = _coerce_float(
        getattr(cfg, "ACTIVITY_SPINNER_INTERVAL_SECONDS", 0.2),
        default=0.2,
    )
    progress = _ProgressBar(
        total=max(1, _coerce_int(getattr(cfg, "NUM_SIMULATIONS", 1), default=1)),
        enabled=bool(getattr(cfg, "ENABLE_PROGRESS_BAR", False)),
        update_every=progress_step,
        animate_spinner=bool(getattr(cfg, "ENABLE_ACTIVITY_SPINNER", True)),
        spinner_interval_seconds=spinner_interval,
    )
    progress.start()
    progress.update(0)

    tau_matrix = simulate_default_time_matrix(prepared)
    n_paths, _ = tau_matrix.shape
    progress.set_total(n_paths)
    progress.update(0)
    worker_count = min(requested_workers, max(1, n_paths))

    ctx = _build_path_context(
        prepared=prepared,
        dates=dates,
        calendar_selection=calendar_selection,
        start_eff=start_eff,
        payment_dates=payment_dates,
        quarter_dates=quarter_dates,
        maturity_dates=maturity_dates,
        replenishment_end_date=replenishment_end_date,
        debtor_loans=debtor_loans,
        basis_days=basis_days,
        enable_prepayment=enable_prepayment,
        cpr_annual=cpr_annual,
        prepayment_none_map=prepayment_none_map,
        total_stack_asof=total_stack_asof,
        attachment_point=attachment_point,
        detachment_point=detachment_point,
        tranche_amortization_mode=tranche_amortization_mode,
        our_share=our_share,
        tau_matrix=tau_matrix,
    )

    path_pv_prem = np.zeros(n_paths, dtype=float)
    path_pv_wd = np.zeros(n_paths, dtype=float)
    path_pv_red = np.zeros(n_paths, dtype=float)
    path_pv01 = np.zeros(n_paths, dtype=float)
    path_tranche_loss = np.zeros(n_paths, dtype=float)
    prem_sum_by_date: dict[date, float] = {}
    wd_sum_by_date: dict[date, float] = {}
    red_sum_by_date: dict[date, float] = {}
    try:
        if worker_count <= 1:
            chunk = _price_path_range(ctx, 0, n_paths, progress_callback=progress.update)
            path_pv_prem[:] = chunk.pv_premium
            path_pv_wd[:] = chunk.pv_write_down
            path_pv_red[:] = chunk.pv_redemption
            path_pv01[:] = chunk.pv01
            path_tranche_loss[:] = chunk.tranche_loss
            _merge_cashflow_sums(prem_sum_by_date, chunk.premium_sum_by_date)
            _merge_cashflow_sums(wd_sum_by_date, chunk.write_down_sum_by_date)
            _merge_cashflow_sums(red_sum_by_date, chunk.redemption_sum_by_date)
        else:
            global _PATH_PRICING_CONTEXT
            _PATH_PRICING_CONTEXT = ctx
            completed_paths = 0
            path_ranges = _path_chunk_ranges(n_paths, worker_count)
            with mp.get_context("fork").Pool(processes=worker_count) as pool:
                for chunk in pool.imap_unordered(_price_path_chunk_worker, path_ranges):
                    chunk_size = len(chunk.pv_premium)
                    end_idx = chunk.start_idx + chunk_size
                    path_pv_prem[chunk.start_idx:end_idx] = chunk.pv_premium
                    path_pv_wd[chunk.start_idx:end_idx] = chunk.pv_write_down
                    path_pv_red[chunk.start_idx:end_idx] = chunk.pv_redemption
                    path_pv01[chunk.start_idx:end_idx] = chunk.pv01
                    path_tranche_loss[chunk.start_idx:end_idx] = chunk.tranche_loss
                    _merge_cashflow_sums(prem_sum_by_date, chunk.premium_sum_by_date)
                    _merge_cashflow_sums(wd_sum_by_date, chunk.write_down_sum_by_date)
                    _merge_cashflow_sums(red_sum_by_date, chunk.redemption_sum_by_date)
                    completed_paths += chunk_size
                    progress.update(completed_paths)
    finally:
        _PATH_PRICING_CONTEXT = None

    pv_premium_full = float(path_pv_prem.mean())
    pv_wd_full = float(path_pv_wd.mean())
    pv_red_full = float(path_pv_red.mean())
    pv01_full = float(path_pv01.mean())

    # Ownership scaling: simulate full tranche state, then scale investor cashflows by OUR_PERCENTAGE.
    pv_premium = pv_premium_full * our_share
    pv_wd = pv_wd_full * our_share
    pv_red = pv_red_full * our_share
    pv01 = pv01_full * our_share
    npv_mtm = pv_premium + pv_wd + pv_red

    n_asof_ours = n_sched_asof_full * our_share
    if n_asof_ours <= 0.0:
        clean = 0.0
    else:
        clean = 100.0 * npv_mtm / n_asof_ours

    pv_wd_positive = float(np.mean(-path_pv_wd)) * our_share
    try:
        par_spread = solve_par_spread_closed_form(pv01=pv01, pv_wd_positive=pv_wd_positive, pv_red=pv_red)
    except ConfigValidationError:
        par_spread = 0.0

    expected_loss = float(path_tranche_loss.mean())
    var99 = float(np.quantile(path_tranche_loss, 0.99))
    tail = path_tranche_loss[path_tranche_loss >= var99]
    es99 = float(tail.mean()) if tail.size > 0 else var99

    recon_dates = sorted(set(prem_sum_by_date) | set(wd_sum_by_date) | set(red_sum_by_date))
    reconciliation = pd.DataFrame(
        {
            "date": recon_dates,
            "expected_premium_cf": [prem_sum_by_date.get(d, 0.0) / n_paths for d in recon_dates],
            "expected_write_down_cf": [wd_sum_by_date.get(d, 0.0) / n_paths for d in recon_dates],
            "expected_redemption_cf": [red_sum_by_date.get(d, 0.0) / n_paths for d in recon_dates],
        }
    )

    progress.close()
    return PricingResult(
        pv_premium=pv_premium,
        pv_write_down=pv_wd,
        pv_redemption=pv_red,
        npv_mtm=npv_mtm,
        clean_price=clean,
        pv01=pv01,
        par_spread=par_spread,
        expected_loss=expected_loss,
        var99_loss=var99,
        es99_loss=es99,
        reconciliation_table=reconciliation,
        n_paths=n_paths,
        n_obligors=len(prepared.debtor_ids),
        tranche_notional_asof_full=n_sched_asof_full,
        tranche_notional_asof_ours=n_asof_ours,
    )
