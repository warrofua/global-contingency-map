"""
Phase 1 Financial Surface Pipeline
Integrates SPD extraction, feature computation, regime detection, and early warnings
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict

from .features import FinancialFeatureExtractor
from ..regime.hmm_detector import HMMRegimeDetector
from ..early_warning.signals import EarlyWarningSignals
from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class FinancialSurfacePipeline:
    """
    Complete Phase 1 pipeline for financial surface analysis

    Workflow:
    1. Extract features from options data
    2. Detect regimes with HMM
    3. Compute early warning signals
    4. Generate regime report
    """

    def __init__(self, ticker: str = None):
        """
        Initialize pipeline

        Args:
            ticker: Stock ticker (default: config.DEFAULT_TICKER)
        """
        self.ticker = ticker or config.DEFAULT_TICKER
        self.feature_extractor = FinancialFeatureExtractor(self.ticker)
        self.regime_detector = HMMRegimeDetector()
        self.early_warning = EarlyWarningSignals()
        logger.info(f"Initialized financial surface pipeline for {self.ticker}")

    def run_daily_update(self) -> Dict:
        """
        Run daily pipeline update

        Returns:
            Dictionary with current state and alerts
        """
        logger.info("Running daily financial surface update...")

        # Extract today's features
        today_features = self.feature_extractor.extract_features()

        # Load historical features from cache if available
        cache_path = config.CACHE_DIR / f"financial_features_{self.ticker}.parquet"

        if cache_path.exists():
            historical_features = pd.read_parquet(cache_path)
            # Append today's features
            features_df = pd.concat([
                historical_features,
                pd.DataFrame([today_features]).set_index('timestamp')
            ])
        else:
            features_df = pd.DataFrame([today_features]).set_index('timestamp')

        # Train/update HMM if we have enough data
        if len(features_df) >= 50:
            self.regime_detector.fit(features_df)

            # Predict current regime
            regime_proba = self.regime_detector.predict(features_df, return_proba=True)
            current_regime = np.argmax(regime_proba[-1])
            current_regime_prob = regime_proba[-1]

            # Compute transition probabilities
            trans_5day = self.regime_detector.get_transition_probabilities(
                current_regime, horizon=5
            )
            trans_20day = self.regime_detector.get_transition_probabilities(
                current_regime, horizon=20
            )

            # Compute regime entropy
            regime_entropy = self.regime_detector.compute_regime_entropy(features_df)

            # Compute early warning signals
            # Use tail mass as primary indicator
            tail_mass_series = features_df['tail_mass_total']
            ew_signals = self.early_warning.compute_all_signals(
                tail_mass_series,
                regime_entropy=regime_entropy
            )

            current_signals = ew_signals.iloc[-1]

        else:
            logger.warning("Insufficient data for regime detection (need 50+ samples)")
            current_regime = None
            current_regime_prob = None
            trans_5day = None
            trans_20day = None
            current_signals = None

        # Save updated features
        features_df.to_parquet(cache_path)

        # Prepare output
        output = {
            'timestamp': datetime.now(),
            'ticker': self.ticker,
            'features': today_features,
            'regime': {
                'current_state': int(current_regime) if current_regime is not None else None,
                'state_probabilities': current_regime_prob.tolist() if current_regime_prob is not None else None,
                'transition_5day': trans_5day.tolist() if trans_5day is not None else None,
                'transition_20day': trans_20day.tolist() if trans_20day is not None else None
            },
            'early_warning': {
                'ar1': float(current_signals['ar1']) if current_signals is not None else None,
                'variance_ratio': float(current_signals.get('variance_ratio', np.nan)) if current_signals is not None else None,
                'alert_level': int(current_signals['alert_level']) if current_signals is not None else 0
            }
        }

        logger.info(f"Daily update complete. Regime: {current_regime}, Alert: {output['early_warning']['alert_level']}")

        return output

    def backtest(
        self,
        start_date: str,
        end_date: str,
        train_window: int = 100
    ) -> pd.DataFrame:
        """
        Backtest regime detection and early warnings

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            train_window: Number of days for initial training

        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Generate date range (business days only)
        date_range = pd.bdate_range(start=start_date, end=end_date)

        # Extract features for all dates
        logger.info(f"Extracting features for {len(date_range)} days...")
        features_list = []

        for date in date_range:
            try:
                features = self.feature_extractor.extract_features(timestamp=date)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for {date}: {e}")
                # Continue with NaN features
                features_list.append(self.feature_extractor._get_nan_features(date))

        features_df = pd.DataFrame(features_list)
        features_df.set_index('timestamp', inplace=True)

        # Rolling window training and prediction
        results = []

        for i in range(train_window, len(features_df)):
            train_data = features_df.iloc[:i]

            # Train HMM
            try:
                self.regime_detector.fit(train_data)

                # Predict current state
                current_features = features_df.iloc[i:i + 1]
                regime = self.regime_detector.predict(current_features)[0]
                regime_proba = self.regime_detector.predict(current_features, return_proba=True)[0]

                # Compute early warnings
                tail_mass_series = features_df['tail_mass_total'].iloc[:i + 1]
                regime_entropy = self.regime_detector.compute_regime_entropy(train_data)

                ew_signals = self.early_warning.compute_all_signals(
                    tail_mass_series,
                    regime_entropy=regime_entropy
                )

                current_ew = ew_signals.iloc[-1]

                results.append({
                    'date': features_df.index[i],
                    'regime': regime,
                    'regime_prob': regime_proba.max(),
                    'ar1': current_ew['ar1'],
                    'variance': current_ew['variance'],
                    'alert_level': current_ew['alert_level'],
                    **features_df.iloc[i].to_dict()
                })

            except Exception as e:
                logger.error(f"Backtest failed at {features_df.index[i]}: {e}")
                continue

        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)

        logger.info("Backtest complete")

        return results_df
