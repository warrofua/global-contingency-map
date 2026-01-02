#!/usr/bin/env python3
"""
Phase 1 Demo: Financial Surface Analysis

Demonstrates:
- SPD extraction from options
- Feature computation
- HMM regime detection
- Early warning signals
- Dashboard visualization
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gcm.financial.pipeline import FinancialSurfacePipeline
from gcm.dashboard.dashboard import RegimeDashboard
from gcm.utils.config import config

def main():
    print("=" * 60)
    print("GCM Phase 1: Financial Surface Demo")
    print("=" * 60)

    # Initialize pipeline
    ticker = "SPY"
    print(f"\nInitializing pipeline for {ticker}...")
    pipeline = FinancialSurfacePipeline(ticker=ticker)

    # Run daily update
    print("\nRunning daily update...")
    try:
        result = pipeline.run_daily_update()

        print("\n--- Current State ---")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Ticker: {result['ticker']}")

        print("\n--- Features ---")
        for key, value in result['features'].items():
            if key != 'timestamp':
                print(f"  {key}: {value}")

        print("\n--- Regime ---")
        print(f"  Current state: {result['regime']['current_state']}")
        print(f"  State probabilities: {result['regime']['state_probabilities']}")

        print("\n--- Early Warnings ---")
        ar1 = result['early_warning']['ar1']
        variance_ratio = result['early_warning']['variance_ratio']
        alert_level = result['early_warning']['alert_level']

        print(f"  AR(1): {ar1:.3f}" if ar1 is not None else "  AR(1): N/A (insufficient data)")
        print(f"  Variance ratio: {variance_ratio:.3f}" if variance_ratio is not None else "  Variance ratio: N/A (insufficient data)")
        print(f"  Alert level: {alert_level}")

        # Create dashboard
        print("\n--- Generating Dashboard ---")
        dashboard = RegimeDashboard()
        fig = dashboard.create_daily_report(result)

        output_path = config.DATA_DIR / "phase1_dashboard.html"
        dashboard.save_html(fig, str(output_path))
        print(f"Dashboard saved to: {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This demo requires live market data and may fail outside trading hours.")
        print("Run backtest demo for historical analysis.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
