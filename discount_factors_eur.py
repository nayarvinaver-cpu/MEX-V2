#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
FILE_PATH = "eur_quotes.xlsx"
SHEET_NAME = "Discount Function"
ALLOW_EXTRAPOLATION = False  # keep False for no extrapolation


#%%
def _to_float_maybe_comma(x) -> float:
    """
    Robust float parser:
    - handles numeric input
    - handles strings with decimal comma (e.g. '0,9987')
    - handles strings with decimal point (e.g. '0.9987')
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip().replace(" ", "")
    if s == "":
        return np.nan

    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s and "." in s:
        # assume comma is thousands separator
        s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return np.nan


#%%
def read_discount_function_sheet(file_path: str, sheet_name: str):
    """
    Reads the SCecon-style 'Discount Function' sheet with structure:
    row 1: title
    row 2: metadata labels (Disc Ipol, Intpol Conv, ...)
    row 3: metadata values
    row 4: data header ('Date', 'Factor')
    row 5+: curve points
    """
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Find the row that contains Date / Factor headers
    header_idx = None
    for i in range(len(raw)):
        c0 = str(raw.iat[i, 0]).strip().upper() if not pd.isna(raw.iat[i, 0]) else ""
        c1 = str(raw.iat[i, 1]).strip().upper() if not pd.isna(raw.iat[i, 1]) else ""
        if c0 == "DATE" and c1 == "FACTOR":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find 'Date'/'Factor' header row in the sheet.")

    if header_idx < 2:
        raise ValueError("Sheet layout unexpected: metadata rows not found above Date/Factor row.")

    # Metadata rows (expected directly above Date/Factor row)
    labels = raw.iloc[header_idx - 2, :5].tolist()
    values = raw.iloc[header_idx - 1, :5].tolist()

    meta = {}
    for k, v in zip(labels, values):
        key = str(k).strip() if not pd.isna(k) else ""
        val = str(v).strip() if not pd.isna(v) else ""
        if key:
            meta[key] = val

    # Data rows from below Date/Factor header
    data = raw.iloc[header_idx + 1:, [0, 1]].copy()
    data.columns = ["Date", "Factor"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Factor"] = data["Factor"].apply(_to_float_maybe_comma)

    data = data.dropna(subset=["Date", "Factor"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid curve points found under Date/Factor.")

    return meta, data


#%%
def yearfrac(start_date: pd.Timestamp, end_date: pd.Timestamp, cal_conv: str) -> float:
    """
    Year fraction according to calendar convention.
    Currently implemented: ACT360 (as required by your EUR sheet).
    """
    cal_conv_u = cal_conv.strip().upper()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if cal_conv_u == "ACT360":
        return (end - start).days / 360.0

    raise NotImplementedError(f"Calendar convention '{cal_conv}' is not implemented in this script.")


#%%
#%%
class DiscountCurveEUR:
    """
    Discount curve from node dates + DFs using:
    - Disc Ipol  : DI_SPOT
    - Intpol Conv: LINEAR_FLAT_END
    - Cal Conv   : ACT360
    - Irr Conv   : COMPOUND
    - Pmt Freq   : ANNUALLY
    """

    def __init__(
        self,
        node_dates: pd.Series,
        node_dfs: pd.Series,
        disc_ipol: str,
        intpol_conv: str,
        cal_conv: str,
        irr_conv: str,
        pmt_freq: str,
        allow_extrapolation: bool = False,
    ):
        self.disc_ipol = disc_ipol.strip().upper()
        self.intpol_conv = intpol_conv.strip().upper()
        self.cal_conv = cal_conv.strip().upper()
        self.irr_conv = irr_conv.strip().upper()
        self.pmt_freq = pmt_freq.strip().upper()
        self.allow_extrapolation = allow_extrapolation

        # Validate conventions expected for this EUR setup
        if self.disc_ipol != "DI_SPOT":
            raise NotImplementedError(f"Expected DI_SPOT, got {self.disc_ipol}.")
        if self.intpol_conv != "LINEAR_FLAT_END":
            raise NotImplementedError(f"Expected LINEAR_FLAT_END, got {self.intpol_conv}.")
        if self.cal_conv != "ACT360":
            raise NotImplementedError(f"Expected ACT360, got {self.cal_conv}.")
        if self.irr_conv != "COMPOUND":
            raise NotImplementedError(f"Expected COMPOUND, got {self.irr_conv}.")
        if self.pmt_freq != "ANNUALLY":
            raise NotImplementedError(f"Expected ANNUALLY, got {self.pmt_freq}.")

        df = pd.DataFrame({
            "Date": pd.to_datetime(node_dates, errors="coerce"),
            "DF": pd.to_numeric(node_dfs, errors="coerce")
        }).dropna().reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid nodes passed to curve constructor.")
        if (df["DF"] <= 0).any():
            raise ValueError("All discount factors must be strictly positive.")

        # Normalize date part, keep as DatetimeIndex (Timestamp elements)
        df["Date"] = df["Date"].dt.normalize()
        self.node_dates = pd.DatetimeIndex(df["Date"])
        self.node_df = df["DF"].to_numpy(dtype=float)

        self.anchor_date = self.node_dates[0]  # pandas Timestamp

        # Build times (ACT360) relative to anchor
        self.node_t = np.array(
            [yearfrac(self.anchor_date, d, self.cal_conv) for d in self.node_dates],
            dtype=float
        )

        # Validate monotonicity
        if not np.all(np.diff(self.node_t) >= 0):
            raise ValueError("Node dates must be non-decreasing.")
        if np.any(np.diff(self.node_t) == 0):
            raise ValueError("Duplicate node dates are not allowed.")
        if abs(self.node_t[0]) > 1e-14:
            raise ValueError("First node must be the anchor date (t=0).")

        # Convert node DFs -> node spot rates for DI_SPOT interpolation:
        # COMPOUND + ANNUALLY:
        #   DF(t) = (1 + r)^(-t)  =>  r = DF^(-1/t) - 1
        self.node_spot = np.zeros_like(self.node_t)
        mask = self.node_t > 0
        self.node_spot[mask] = np.power(self.node_df[mask], -1.0 / self.node_t[mask]) - 1.0

    def _exact_node_index(self, qd: pd.Timestamp):
        matches = np.where(self.node_dates.values == np.datetime64(qd))[0]
        return int(matches[0]) if matches.size > 0 else None

    def _interp_spot(self, t: float) -> float:
        t_min = self.node_t[0]
        t_max = self.node_t[-1]

        if not self.allow_extrapolation and (t < t_min or t > t_max):
            raise ValueError(
                f"Date outside curve range ({self.node_dates[0].date()} to {self.node_dates[-1].date()})."
            )

        # LINEAR_FLAT_END:
        # - linear between nodes
        # - flat outside range (only used if allow_extrapolation=True)
        return float(
            np.interp(
                t,
                self.node_t,
                self.node_spot,
                left=self.node_spot[0],
                right=self.node_spot[-1],
            )
        )

    def get_spot_rate(self, query_date) -> float:
        qd = pd.Timestamp(query_date).normalize()
        t = yearfrac(self.anchor_date, qd, self.cal_conv)

        # exact node shortcut
        i = self._exact_node_index(qd)
        if i is not None:
            return float(self.node_spot[i])

        return self._interp_spot(t)

    def get_discount_factor(self, query_date) -> float:
        qd = pd.Timestamp(query_date).normalize()
        t = yearfrac(self.anchor_date, qd, self.cal_conv)

        # exact node shortcut
        i = self._exact_node_index(qd)
        if i is not None:
            return float(self.node_df[i])

        r = self._interp_spot(t)

        if abs(t) < 1e-15:
            return 1.0

        base = 1.0 + r
        if base <= 0:
            raise ValueError("Invalid interpolated spot: (1+r) <= 0 for compounded discounting.")
        return float(np.power(base, -t))


#%%
def plot_discount_curve(curve: DiscountCurveEUR, n_points: int = 500, include_nodes: bool = True):
    """
    Plots interpolated discount curve across available maturity range.
    """
    t0 = curve.node_dates[0]
    t1 = curve.node_dates[-1]

    grid_dates = pd.date_range(start=t0, end=t1, periods=n_points)
    grid_df = [curve.get_discount_factor(d) for d in grid_dates]

    plt.figure(figsize=(10, 5))
    plt.plot(grid_dates, grid_df, label="Interpolated DF curve")

    if include_nodes:
        plt.scatter(curve.node_dates, curve.node_df, s=20, label="Input nodes")

    plt.title("EUR Discount Curve")
    plt.xlabel("Date")
    plt.ylabel("Discount Factor")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#%%
# Build curve from Excel
meta, curve_data = read_discount_function_sheet(FILE_PATH, SHEET_NAME)

disc_ipol = meta.get("Disc Ipol", "")
intpol_conv = meta.get("Intpol Conv", "")
cal_conv = meta.get("Cal Conv", "")
irr_conv = meta.get("Irr Conv", "")
pmt_freq = meta.get("Pmt Freq", "")

curve = DiscountCurveEUR(
    node_dates=curve_data["Date"],
    node_dfs=curve_data["Factor"],
    disc_ipol=disc_ipol,
    intpol_conv=intpol_conv,
    cal_conv=cal_conv,
    irr_conv=irr_conv,
    pmt_freq=pmt_freq,
    allow_extrapolation=ALLOW_EXTRAPOLATION,
)

print("Curve loaded successfully.")
print(f"Anchor date: {curve.anchor_date.date()}")
print(f"First DF: {curve.node_df[0]:.10f}")
print(f"Last node: {curve.node_dates[-1].date()} | Last DF: {curve.node_df[-1]:.10f}")


#%%
# Usage example
query_date = "2034-01-30"
try:
    df_q = curve.get_discount_factor(query_date)
    print(f"Discount factor on {pd.Timestamp(query_date).date()}: {df_q:.10f}")
except ValueError as e:
    print(f"Could not retrieve DF for {query_date}: {e}")


#%%
# Plot curve
plot_discount_curve(curve)
