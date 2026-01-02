"""
Financial surface feature computation
Extracts features from market-implied contingency surface
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

from .spd_extractor import SPDExtractor
from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class FinancialFeatureExtractor:
    """
    Extract feature vector x^F_t from financial surface

    Features include:
    - Tail mass (±2σ)
    - Skew proxy
    - Kurtosis proxy
    - Cross-asset correlation regime (future)
    - Term structure slope (future)
    """

    def __init__(self, ticker: str = None):
        """
        Initialize feature extractor

        Args:
            ticker: Stock ticker (defaults to config.DEFAULT_TICKER)
        """
        self.ticker = ticker or config.DEFAULT_TICKER
        self.spd_extractor = SPDExtractor(self.ticker)
        logger.info(f"Initialized financial feature extractor for {self.ticker}")

    def extract_features(
        self,
        timestamp: Optional[datetime] = None,
        expiry_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract full feature vector for a given timestamp

        Args:
            timestamp: Timestamp for feature extraction (default: now)
            expiry_date: Specific option expiry date (default: auto-select)

        Returns:
            Dictionary of features
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"Extracting features for {self.ticker} at {timestamp}")

        try:
            # Fetch option data
            calls, puts, spot_price, expiry_dt = self.spd_extractor.fetch_option_chain(
                expiry_date=expiry_date,
                min_days=config.OPTION_EXPIRY_DAYS_MIN,
                max_days=config.OPTION_EXPIRY_DAYS_MAX
            )

            # Extract SPD
            strikes, spd = self.spd_extractor.extract_spd(
                calls, spot_price, expiry_dt
            )

            # Compute SPD moments
            spd_moments = self.spd_extractor.compute_spd_moments(
                strikes, spd, spot_price
            )

            # Compute skew proxy from options
            skew_proxy = self._compute_skew_proxy(calls, puts, spot_price)

            # Compute term structure slope (placeholder for MVP)
            term_structure_slope = 0.0  # TODO: Implement with multiple expiries

            # Assemble feature vector
            features = {
                'timestamp': timestamp,
                'spot_price': spot_price,
                'tail_mass_left': spd_moments['tail_mass_left'],
                'tail_mass_right': spd_moments['tail_mass_right'],
                'tail_mass_total': spd_moments['tail_mass_total'],
                'skew_proxy': skew_proxy,
                'kurtosis': spd_moments['kurtosis'],
                'spd_std': spd_moments['std'],
                'spd_mean': spd_moments['mean'],
                'spd_skewness': spd_moments['skewness'],
                'term_structure_slope': term_structure_slope
            }

            logger.info(f"Extracted {len(features)} features")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return NaN features on failure
            return self._get_nan_features(timestamp)

    def _compute_skew_proxy(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        spot_price: float
    ) -> float:
        """
        Compute skew proxy from implied volatility difference

        Skew = IV(25Δ put) - IV(25Δ call)
        Approximated by comparing OTM put vs OTM call IVs

        Args:
            calls: Call options DataFrame
            puts: Put options DataFrame
            spot_price: Current stock price

        Returns:
            Skew proxy value
        """
        try:
            # Find near-the-money options (around 25 delta equivalent)
            # 25 delta call: strike ~ spot * 1.05
            # 25 delta put: strike ~ spot * 0.95

            call_strike_target = spot_price * 1.05
            put_strike_target = spot_price * 0.95

            # Find closest strikes
            call_idx = (calls['strike'] - call_strike_target).abs().idxmin()
            put_idx = (puts['strike'] - put_strike_target).abs().idxmin()

            call_iv = calls.loc[call_idx, 'impliedVolatility']
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            skew = put_iv - call_iv

            return skew

        except Exception as e:
            logger.warning(f"Skew computation failed: {e}")
            return 0.0

    def _get_nan_features(self, timestamp: datetime) -> Dict[str, float]:
        """Return feature dict with NaN values"""
        return {
            'timestamp': timestamp,
            'spot_price': np.nan,
            'tail_mass_left': np.nan,
            'tail_mass_right': np.nan,
            'tail_mass_total': np.nan,
            'skew_proxy': np.nan,
            'kurtosis': np.nan,
            'spd_std': np.nan,
            'spd_mean': np.nan,
            'spd_skewness': np.nan,
            'term_structure_slope': np.nan
        }

    def extract_time_series(
        self,
        dates: pd.DatetimeIndex,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract features for multiple timestamps

        Args:
            dates: DatetimeIndex of timestamps
            cache: Whether to cache results

        Returns:
            DataFrame with features as columns, dates as index
        """
        features_list = []

        for date in dates:
            features = self.extract_features(timestamp=date)
            features_list.append(features)

        df = pd.DataFrame(features_list)
        df.set_index('timestamp', inplace=True)

        if cache:
            cache_path = config.CACHE_DIR / f"financial_features_{self.ticker}.parquet"
            df.to_parquet(cache_path)
            logger.info(f"Cached features to {cache_path}")

        return df
