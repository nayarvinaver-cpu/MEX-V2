#%% Imports
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from bisect import bisect_left
from typing import Dict, List, Optional, Sequence, Union
import warnings

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

DateLike = Union[date, datetime, str, pd.Timestamp]


#%% Basic helpers
def plot_discount_curve(curve: DiscountCurve, n_points: int = 800, include_nodes: bool = True) -> None:
    start = pd.Timestamp(curve.node_dates[0])
    end = pd.Timestamp(curve.node_dates[-1])

    grid = pd.date_range(start=start, end=end, periods=n_points)
    dfs = [curve.discount_factor(d) for d in grid]  # no extrapolation used

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, dfs, linewidth=1.5, label="Interpolated DF")

    if include_nodes:
        ax.plot(curve.node_dates, curve.node_dfs, "o", linestyle="None", label="Node DFs")

    ax.set_title("Discount Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Discount Factor")
    ax.grid(True, alpha=0.3)
    ax.legend()

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.tight_layout()
    plt.show()

def to_date(x: DateLike) -> date:
    """Convert supported input to datetime.date."""
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return pd.to_datetime(x, errors="raise").date()
    raise TypeError(f"Unsupported date type: {type(x)}")


def yearfrac_act360(d1: date, d2: date) -> float:
    """ACT/360 year fraction for [d1, d2)."""
    return (d2 - d1).days / 360.0


def parse_float_maybe_comma(x) -> Optional[float]:
    """Parse numeric values that may use comma decimal separators."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip().replace("\u00A0", "").replace(" ", "")
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def cell_norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


#%% Curve class (DI_FORW + EXCL_INCL + ACT360 + COMPOUND annual)
@dataclass
class DiscountCurve:
    node_dates: List[date]
    node_dfs: List[float]
    conventions: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if len(self.node_dates) != len(self.node_dfs):
            raise ValueError("node_dates and node_dfs must have same length.")
        if len(self.node_dates) < 2:
            raise ValueError("At least 2 curve nodes are required.")

        for i in range(1, len(self.node_dates)):
            if self.node_dates[i] <= self.node_dates[i - 1]:
                raise ValueError("node_dates must be strictly increasing.")

        for df in self.node_dfs:
            if df is None or df <= 0.0:
                raise ValueError("All discount factors must be > 0.")

        self.anchor_date: date = self.node_dates[0]
        self.anchor_df: float = self.node_dfs[0]

        # Segment forwards:
        # df_{i+1} = df_i / (1+f_i)^{dt_i}  =>  f_i = (df_i/df_{i+1})^(1/dt_i) - 1
        self._seg_fwd: List[float] = []
        for i in range(len(self.node_dates) - 1):
            d0, d1 = self.node_dates[i], self.node_dates[i + 1]
            dt = yearfrac_act360(d0, d1)
            if dt <= 0:
                raise ValueError(f"Non-positive ACT/360 year fraction between {d0} and {d1}.")
            df0, df1 = self.node_dfs[i], self.node_dfs[i + 1]
            fwd = (df0 / df1) ** (1.0 / dt) - 1.0
            self._seg_fwd.append(fwd)

    def discount_factor(self, q_date: DateLike) -> float:
        """
        Return DF for q_date using:
          - DI_FORW (piecewise-constant forward)
          - EXCL_INCL intervals: (t_i, t_{i+1}]
        No extrapolation is allowed.
        """
        q = to_date(q_date)

        # No extrapolation
        if q < self.node_dates[0] or q > self.node_dates[-1]:
            raise ValueError(
                f"Query date {q} is outside curve range "
                f"[{self.node_dates[0]}, {self.node_dates[-1]}]. Extrapolation is disabled."
            )

        # Exact node hit
        idx = bisect_left(self.node_dates, q)
        if idx < len(self.node_dates) and self.node_dates[idx] == q:
            return self.node_dfs[idx]

        # Internal point: find i such that node_dates[i] < q <= node_dates[i+1]
        j = bisect_left(self.node_dates, q)
        i = j - 1
        if i < 0 or i >= len(self._seg_fwd):
            raise RuntimeError("Failed to locate interpolation segment.")

        tau = yearfrac_act360(self.node_dates[i], q)
        return self.node_dfs[i] / ((1.0 + self._seg_fwd[i]) ** tau)

    def discount_factors(self, q_dates: Sequence[DateLike]) -> List[float]:
        return [self.discount_factor(d) for d in q_dates]

    def segment_forwards(self) -> List[float]:
        return list(self._seg_fwd)


#%% Excel parsing
def _extract_conventions(raw: pd.DataFrame) -> Dict[str, str]:
    key_names = {"disc ipol", "intpol conv", "cal conv", "irr conv", "pmt freq"}

    for r in range(len(raw) - 1):
        row_norm = [cell_norm(v) for v in raw.iloc[r].tolist()]
        if key_names.issubset(set(row_norm)):
            headers = [str(v).strip() if not pd.isna(v) else "" for v in raw.iloc[r].tolist()]
            values = [str(v).strip() if not pd.isna(v) else "" for v in raw.iloc[r + 1].tolist()]
            conv = {}
            for h, v in zip(headers, values):
                if h:
                    conv[h] = v
            return conv

    return {}


def _extract_curve_nodes(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = None
    factor_col = None
    start_row = None

    for r in range(len(raw)):
        row_norm = [cell_norm(v) for v in raw.iloc[r].tolist()]
        if "date" in row_norm and "factor" in row_norm:
            date_col = row_norm.index("date")
            factor_col = row_norm.index("factor")
            start_row = r + 1
            break

    if start_row is None or date_col is None or factor_col is None:
        raise ValueError("Could not find 'Date'/'Factor' headers in the sheet.")

    dates: List[date] = []
    factors: List[float] = []
    started = False

    for r in range(start_row, len(raw)):
        d_raw = raw.iat[r, date_col]
        f_raw = raw.iat[r, factor_col]

        d_parsed = pd.to_datetime(d_raw, errors="coerce")
        f_parsed = parse_float_maybe_comma(f_raw)

        if pd.notna(d_parsed) and f_parsed is not None:
            dates.append(d_parsed.date())
            factors.append(float(f_parsed))
            started = True
            continue

        if started and pd.isna(d_raw) and (f_parsed is None):
            break

    if len(dates) < 2:
        raise ValueError("Not enough curve node rows were parsed from 'Date'/'Factor' block.")

    return pd.DataFrame({"Date": dates, "Factor": factors})


def _validate_conventions(conventions: Dict[str, str]) -> None:
    if not conventions:
        warnings.warn("No conventions found in sheet; proceeding with implemented conventions.")
        return

    expected = {
        "Disc Ipol": "DI_FORW",
        "Intpol Conv": "EXCL_INCL",
        "Cal Conv": "ACT360",
        "Irr Conv": "COMPOUND",
        "Pmt Freq": "ANNUALLY",
    }

    for k, v_exp in expected.items():
        v = conventions.get(k, "")
        if str(v).strip().upper() != v_exp:
            warnings.warn(
                f"Convention mismatch for '{k}': found '{v}', expected '{v_exp}'. "
                "The code uses DI_FORW + EXCL_INCL + ACT360 + COMPOUND(annual)."
            )


#%% Public loader + callable function
def load_discount_curve_from_excel(
    file_path: str = "usd_quotes.xlsx",
    sheet_name: str = "Discount Function",
) -> DiscountCurve:
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    conventions = _extract_conventions(raw)
    _validate_conventions(conventions)

    nodes = _extract_curve_nodes(raw)

    return DiscountCurve(
        node_dates=nodes["Date"].tolist(),
        node_dfs=nodes["Factor"].tolist(),
        conventions=conventions,
    )


curve = load_discount_curve_from_excel("usd_quotes.xlsx", "Discount Function")


def get_discount_factor(q_date: DateLike) -> float:
    """Return discount factor for the given date (no extrapolation)."""
    return curve.discount_factor(q_date)

#%%
# Run:
plot_discount_curve(curve)

#%% Usage example
query_date = "2034-01-30"
df_value = get_discount_factor(query_date)
print(f"DF({query_date}) = {df_value:.10f}")

