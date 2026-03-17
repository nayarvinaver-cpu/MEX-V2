from __future__ import annotations

from srt_model.pipeline import build_prepared_inputs_from_module
from srt_model.pv.pricing import price_prepared_inputs


def main() -> None:
    prepared = build_prepared_inputs_from_module("srt_model_config")
    result = price_prepared_inputs(prepared)
    print("=== SRT Pricing Result ===")
    print(f"Paths: {result.n_paths}")
    print(f"Obligors: {result.n_obligors}")
    print(f"PV Premium: {result.pv_premium:,.2f}")
    print(f"PV Write-down: {result.pv_write_down:,.2f}")
    print(f"PV Redemption: {result.pv_redemption:,.2f}")
    print(f"NPV MTM: {result.npv_mtm:,.2f}")
    print(f"PV01: {result.pv01:,.2f}")
    print(f"Par Spread (decimal): {result.par_spread:.8f}")
    print(f"Expected Loss: {result.expected_loss:,.2f}")
    print(f"VaR99 Loss: {result.var99_loss:,.2f}")
    print(f"ES99 Loss: {result.es99_loss:,.2f}")
    print(f"Clean Price: {result.clean_price:.6f}")


if __name__ == "__main__":
    main()
