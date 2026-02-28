# =============================================================================
# srt_config.py - Core Parameters and Mechanics
# =============================================================================
from datetime import datetime

# =============================================================================
# 1. FILE PATHS & DIRECTORIES
# =============================================================================
CDS_RAW_FILE = "Composite_CDS_Pricing_20251231.csv"
PORTFOLIO_RAW_FILE = "2025_12_31_Datatape_CoCo_II-4_final.xlsx"

DISCOUNT_CURVE_EUR_FILE = "bootstrapped_discount_curve_EUR.csv" 
DISCOUNT_CURVE_USD_FILE = "bootstrapped_discount_curve_USD.csv" 

PORTFOLIO_CLEAN_FILE = "Cleaned_CoCo_Portfolio.csv"
SURVIVAL_EUR_FILE = "bootstrapped_survival_probs_EUR.csv"
SURVIVAL_USD_FILE = "bootstrapped_survival_probs_USD.csv"

# =============================================================================
# 2. RISK & PRICING PARAMETERS
# =============================================================================
FIXED_RECOVERY_CDS = 0.40
MAX_YEARS = 10
VERIFICATION_LAG_MONTHS = 0  # Time from Credit Event to Principal Writedown
NUM_SIMULATIONS = 10000        # Legacy Monte Carlo parameter
RHO = 0.1                # Asset Correlation assumption (20%)

# =============================================================================
# 3. SRT TRANCHE PARAMETERS (CoCo II-4)
# =============================================================================
# --- Original Issuance Structure (Term Sheet Display Only) ---
ORIGINAL_PORTFOLIO_NOTIONAL = 3200000000
TRANCHE_ATTACHMENT = 0.00
TRANCHE_DETACHMENT = 0.061

ALECTA_SHARE = 0.30
PGGM_SHARE = 0.70
REPLENISHMENT_STOP_TRIGGER = 0.0175

# --- Current Valuation State (Drives Calculation) ---
CURRENT_PORTFOLIO_NOTIONAL = 3191131781   # Portfolio EAD at valuation date
CURRENT_TRANCHE_NOTIONAL = 186331781     # Remaining Tranche Principal

# =============================================================================
# 4. STRUCTURAL TIMELINES
# =============================================================================
PORTFOLIO_AS_OF_DATE = "2025-12-31"       # Base date for valuation
REPLENISHMENT_END_DATE = "2026-02-15"     
TRANSACTION_CALL_DATE = "2033-01-31"      # Scheduled Maturity Date 
LEGAL_FINAL_MATURITY_DATE = "2033-07-29"  # Hard maturity drop dead date

# --- Date to Year Conversion ---
def _calc_years(start_str, end_str):
    from datetime import datetime
    d1 = datetime.strptime(start_str, "%Y-%m-%d")
    d2 = datetime.strptime(end_str, "%Y-%m-%d")
    return max(0.0, (d2 - d1).days / 365.25)

REPLENISHMENT_YEARS = _calc_years(PORTFOLIO_AS_OF_DATE, REPLENISHMENT_END_DATE)
TRANSACTION_MATURITY_YEARS = _calc_years(PORTFOLIO_AS_OF_DATE, TRANSACTION_CALL_DATE)
LEGAL_FINAL_MATURITY_YEARS = _calc_years(PORTFOLIO_AS_OF_DATE, LEGAL_FINAL_MATURITY_DATE)