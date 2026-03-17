"""User-editable configuration for the SRT valuation model.

Update this file for each deal. Any required field left empty triggers
a warning and program stop in config validation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Files and sheets
# ---------------------------------------------------------------------------
# Absolute or relative path to the Excel file containing the reference-loan tape used for valuation.
PORTFOLIO_TAPE_PATH = "2025_12_31_Datatape_CoCo_II-4_final.xlsx"
# Exact sheet name inside PORTFOLIO_TAPE_PATH that contains the active pool to be valued.
PORTFOLIO_SHEET_NAME = "Portfolio"
# Path to raw CDS market quotes used when (re)bootstrapping hazard/survival curves.
CDS_RAW_FILE_PATH = "Composite_CDS_Pricing_20251231.csv"
# Path to pre-bootstrapped EUR survival probability matrix (rows=ratings, columns=tenors in years).
SURVIVAL_PROBS_EUR_PATH = "bootstrapped_survival_probs_EUR.csv"
# Path to pre-bootstrapped USD survival probability matrix (rows=ratings, columns=tenors in years).
SURVIVAL_PROBS_USD_PATH = "bootstrapped_survival_probs_USD.csv"
# Path to pre-bootstrapped EUR hazard-rate matrix (rows=ratings, columns=tenors in years).
HAZARD_RATES_EUR_PATH = "bootstrapped_hazard_rates_EUR.csv"
# Path to pre-bootstrapped USD hazard-rate matrix (rows=ratings, columns=tenors in years).
HAZARD_RATES_USD_PATH = "bootstrapped_hazard_rates_USD.csv"
# Path to the EUR discount-curve source file used to query DF(date) for EUR cashflows.
DISCOUNT_CURVE_EUR_FILE = "eur_quotes.xlsx"
# Path to the USD discount-curve source file used to query DF(date) for USD cashflows.
DISCOUNT_CURVE_USD_FILE = "usd_quotes.xlsx"

# ---------------------------------------------------------------------------
# Core deal dates (YYYY-MM-DD)
# ---------------------------------------------------------------------------
# Valuation anchor date (market date); all forward cashflows and discounting are measured relative to this date.
# This date shoul be the same as the date on which all the data was retrieved (SRT data, survival curves, discount curves).
AS_OF_DATE = "2025-12-31"
# First contractual premium payment date in calendar-date form; schedule rolls forward from here in 3M steps.
FIRST_PAYMENT_DATE = "2026-01-29"
# Date from which premium accrual starts economically (may be before or after AS_OF_DATE depending on deal state, and is oftentimes same as Issue Date).
ACCRUAL_START_DATE = "2023-02-02"
# End of credit-protection window (inclusive boundary in the spec).
PROTECTION_END_DATE = "2030-01-29"
# Date at which premium accrual stops; if this is not on the regular schedule, a final stub payment date is created.
ACCRUAL_END_DATE = PROTECTION_END_DATE
# Start of credit-protection window; defaults before this date are not covered.
PROTECTION_START_DATE = ACCRUAL_START_DATE
# Last date on which synthetic replenishment is allowed; after this date no top-up is applied.
REPLENISHMENT_END_DATE = "2026-02-15"
# Hard legal maturity cap date; notice/write-down timing is capped so it cannot occur after this date.
LEGAL_FINAL_MATURITY_DATE = "2033-07-29"

# ---------------------------------------------------------------------------
# Premium / schedule conventions
# ---------------------------------------------------------------------------
# End-of-Month (EOM) roll toggle: if True and anchor is month-end (e.g., Jan 31), quarterly rolls stay month-end.
EOM_ON = False
# Premium spread as decimal per year (example: 0.05 means 5% = 500 bps).
PREMIUM_SPREAD = "0.098"
# Day-count basis used to compute accrued premium fractions (examples: ACT/360, 30E/360, ACT/365F).
PREMIUM_DAY_COUNT = "ACT/360"
# If True, business-day checks use the intersection of multiple calendars; if False, only ISSUER_COUNTRY calendar is used.
JOINT_CALENDARS_ENABLED = False
# Issuer country/calendar key used for business-day adjustment when JOINT_CALENDARS_ENABLED is False.
ISSUER_COUNTRY = "Germany"
# List of country/calendar keys to combine when JOINT_CALENDARS_ENABLED is True (date must be business day in all selected).
JOINT_CALENDAR_COUNTRIES: list[str] = []

# ---------------------------------------------------------------------------
# Tranche and position inputs
# ---------------------------------------------------------------------------
# Attachment point of the selected tranche as a decimal of the current live capital structure at AS_OF_DATE.
# Example: 0.03 means the tranche starts at 3% of the modeled live stack, measured from the bottom.
ATTACHMENT_POINT = 0.0
# Detachment point of the selected tranche as a decimal of the current live capital structure at AS_OF_DATE.
# Example: 0.06 means the tranche ends at 6% of the modeled live stack, measured from the bottom.
DETACHMENT_POINT = 0.05839050023236881
# Scheduled amortization rule for the selected tranche relative to the modeled full SRT stack.
# - "PRO_RATA": attachment and detachment scale proportionally with the total scheduled stack.
# - "SEQUENTIAL": amortization happens from the top down; the selected tranche is the surviving overlap between its AS_OF_DATE band and the remaining scheduled stack, and can refill back up to that band if replenishment later increases the total again.
TRANCHE_AMORTIZATION_MODE = "PRO_RATA"
# Investor ownership share of the selected tranche used to scale full-tranche cashflows to our position valuation.
# Example: 0.30 means we economically own 30% of the selected tranche cashflows/losses while tranche state is still tracked on the full tranche.
OUR_PERCENTAGE = 0.30

# ---------------------------------------------------------------------------
# Risk / simulation controls
# ---------------------------------------------------------------------------
# Recovery assumption used when transforming CDS spreads into hazard/survival curves.
FIXED_RECOVERY_CDS = 0.40
# Maximum CDS tenor (in years) retained when preparing market-implied credit curves.
MAX_YEARS = 10
# Legacy lag input retained for compatibility with prior scripts; set to 0 unless a model component explicitly uses it.
VERIFICATION_LAG_MONTHS = 0
# One-factor Gaussian copula correlation parameter rho (must satisfy 0 <= rho < 1).
RHO = 0.10
# Fixed random seed so Monte Carlo results are reproducible across runs and validation comparisons.
RANDOM_SEED = 42
# Number of Monte Carlo simulation paths; higher values improve stability but increase runtime.
NUM_SIMULATIONS = 1000
# Number of worker processes used for path pricing.
# Set to 1 for single-process pricing. Set to 0 to auto-use available CPU cores.
# Parallel pricing currently uses process-based workers on platforms that support the "fork" start method.
PRICING_NUM_WORKERS = 0
# If True, prints a live progress bar while Monte Carlo paths are being priced.
# This is useful for long runs so you can see that the model is still working and estimate time remaining.
ENABLE_PROGRESS_BAR = True
# If True, an activity spinner rotates continuously on the progress line, even when path count does not advance yet.
# This makes it visually obvious that the process is still alive (for example during heavy startup steps).
ENABLE_ACTIVITY_SPINNER = True
# Spinner refresh frequency in seconds; lower values rotate faster but write to terminal more often.
# Typical values are 0.1 to 0.5 seconds.
ACTIVITY_SPINNER_INTERVAL_SECONDS = 0.2
# How often the progress bar refreshes, measured in number of completed paths.
# Set to 0 to auto-select a sensible refresh cadence (~1% increments). Use larger values for less console spam.
PROGRESS_UPDATE_EVERY_PATHS = 0
# Conversion basis from simulated default-time years to calendar days (tau_days = tau_years * TAU_YEAR_BASIS_DAYS).
TAU_YEAR_BASIS_DAYS = 365.25

# ---------------------------------------------------------------------------
# Prepayment controls
# ---------------------------------------------------------------------------
# If True, apply stochastic loan prepayment; if False, no prepayment cashflow effect is modeled.
ENABLE_PREPAYMENT = False
# Annualized CPR (Conditional Prepayment Rate) in [0,1]; 0.0 means economically no prepayment even if toggle is on.
CPR_ANNUAL = 0.0

# ---------------------------------------------------------------------------
# Rating mapping (internal numeric -> external rating)
# ---------------------------------------------------------------------------
# User-editable mapping from tape Internal_Rating values to external ratings before survival-curve lookup.
INTERNAL_TO_EXTERNAL_RATING = {
    "1.0": "AAA",
    "1.2": "AA+",
    "1.4": "AA",
    "1.6": "AA-",
    "1.8": "A",
    "2.0": "A-",
    "2.2": "BBB+",
    "2.4": "BBB",
    "2.6": "BBB",
    "2.8": "BBB-",
    "3.0": "BB+",
    "3.2": "BB",
    "3.4": "BB",
    "3.6": "BB-",
    "3.8": "B+",
    "4.0": "B+",
    "4.2": "B",
    "4.4": "B",
    "4.6": "B",
    "4.8": "B-",
    "5.0": "CCC+",
    "5.2": "CCC",
    "5.4": "CCC-",
    "5.6": "CC",
    "5.8": "C",
    "6.0": "D",
    "6.2": "D",
    "6.3": "D",
    "6.4": "D",
    "6.5": "D",
}

# ---------------------------------------------------------------------------
# Replenishment limits (prefilled from provided deal screenshot)
# ---------------------------------------------------------------------------
# Replenishment engine mode:
# - "SCALAR_TOPUP" (legacy): one pooled top-up balance that runs off with the base-pool runoff ratio.
# - "VINTAGE_LOANS": each top-up is issued as a separate synthetic vintage with independently chosen characteristics that must satisfy replenishment eligibility and post-addition portfolio guidelines.
REPLENISHMENT_MODE = "SCALAR_TOPUP"
# Maximum allowed replenished reference-pool amount (deal cap) expressed in notional currency units.
# In many transactions this also equals the initial total SRT notional at inception; it can be used as a fallback reference if current live totals are not yet refreshed.
REPLENISHMENT_CAP_AMOUNT = 3_200_000_000.0
# Weighted-average PD assigned to each new replenishment vintage, stored as decimal (0.0342 = 3.42%).
REPL_WAPD_REFERENCE_POOL_MAX = 0.0342
# Previous-pool weighted-average PD benchmark; replenishment is only possible when REPL_WAPD_REFERENCE_POOL_MAX <= REPL_WAPD_PREVIOUS_POOL.
REPL_WAPD_PREVIOUS_POOL = 0.0371

# Stop-event threshold: max weighted-average PD excluding defaulted reference claims, decimal form (0.03 = 3.00%).
STOP_EVENT_WAPD_MAX_EXCL_DEFAULTED = 0.03
# Stop-event threshold: max cumulative loss ratio vs N_ref(as-of), decimal form (0.0175 = 1.75%).
STOP_EVENT_CUMULATIVE_LOSS_MAX = 0.0175

# Latest allowed final maturity date for replenished exposures.
ELIGIBILITY_FINAL_MATURITY_MAX_DATE = "2033-01-31"
# Earliest allowed final maturity date for replenished exposures.
ELIGIBILITY_FINAL_MATURITY_MIN_DATE = "2026-03-31"
# Worst permitted internal rating number in replenished pool (larger numbers are weaker credit quality).
ELIGIBILITY_LOWEST_RATING_MAX_INTERNAL = 4.6

# Max debtor concentration for rating 3.8 and better, decimal form (0.005 = 0.50%).
GUIDELINE_DEBTOR_CONC_R38_AND_BETTER_MAX = 0.005
# Max debtor concentration for rating 4.0 and worse, decimal form (0.003 = 0.30%).
GUIDELINE_DEBTOR_CONC_R40_AND_WORSE_MAX = 0.003
# Max concentration for the single largest Moody's industry group, decimal form (0.12 = 12%).
GUIDELINE_MOODYS_LARGEST_GROUP_MAX = 0.12
# Max concentration for the 2nd to 4th Moody's industry groups, decimal form (0.10 = 10%).
GUIDELINE_MOODYS_2_TO_4_MAX = 0.10
# Max concentration for all other Moody's industry groups, decimal form (0.08 = 8%).
GUIDELINE_MOODYS_OTHER_MAX = 0.08
# Maximum share of debtor groups with aggregate limits < 5,000,000 EUR, decimal form (0.10 = 10%).
GUIDELINE_DEBTOR_GROUPS_LT_5M_MAX = 0.10
# Maximum weighted-average life (WAL, in years) permitted for replenished pool.
GUIDELINE_WAL_REPLENISHED_POOL_MAX = 2.50
