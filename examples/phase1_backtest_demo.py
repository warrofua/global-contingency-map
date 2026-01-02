#!/usr/bin/env python3
"""
Phase 1 Backtest Demo

Backtest regime detection on 2008 and 2020 crises
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gcm.financial.pipeline import FinancialSurfacePipeline
from gcm.dashboard.dashboard import RegimeDashboard
from gcm.utils.config import config

def main():
    print("=" * 60)
    print("GCM Phase 1: Backtest Demo")
    print("=" * 60)

    ticker = "SPY"
    print(f"\nBacktesting {ticker}...")

    pipeline = FinancialSurfacePipeline(ticker=ticker)

    # Backtest periods
    periods = [
        ("2019-01-01", "2020-06-01", "2020 COVID Crisis"),
        ("2007-06-01", "2009-01-01", "2008 Financial Crisis")
    ]

    for start_date, end_date, label in periods:
        print(f"\n--- {label} ({start_date} to {end_date}) ---")

        try:
            results = pipeline.backtest(
                start_date=start_date,
                end_date=end_date,
                train_window=50
            )

            print(f"Backtest complete: {len(results)} datapoints")

            # Find high alert periods
            high_alerts = results[results['alert_level'] >= 2]
            print(f"High alert periods: {len(high_alerts)}")

            if len(high_alerts) > 0:
                print("\nFirst few alerts:")
                print(high_alerts[['regime', 'ar1', 'alert_level']].head())

            # Create dashboard
            dashboard = RegimeDashboard()

            # Key events
            events = {
                "2020-02-20": "COVID Crash Start",
                "2020-03-23": "Market Bottom",
                "2008-09-15": "Lehman Collapse",
                "2008-10-10": "Crisis Peak"
            }

            fig = dashboard.create_backtest_report(results, events=events)

            output_path = config.DATA_DIR / f"backtest_{label.replace(' ', '_')}.html"
            dashboard.save_html(fig, str(output_path))
            print(f"Dashboard saved to: {output_path}")

        except Exception as e:
            print(f"Backtest failed: {e}")
            print("Note: Historical data may not be available")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
