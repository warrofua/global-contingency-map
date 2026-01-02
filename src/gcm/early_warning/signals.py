"""
Early warning signals for critical transitions

Implements Scheffer et al. (2009) Nature indicators:
1. Critical slowing down (increasing autocorrelation)
2. Variance increase
3. Flickering (regime switching)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats
from statsmodels.tsa.stattools import acf

from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class EarlyWarningSignals:
    """
    Compute early warning signals for phase transitions

    Critical transition theory predicts generic signatures:
    - Slowing recovery (AR1 → 1)
    - Increased variance
    - Flickering between states
    """

    def __init__(self, window: int = None):
        """
        Initialize early warning detector

        Args:
            window: Rolling window size (default: config.ROLLING_WINDOW)
        """
        self.window = window or config.ROLLING_WINDOW
        logger.info(f"Initialized early warning signals with window={self.window}")

    def compute_all_signals(
        self,
        time_series: pd.Series,
        regime_entropy: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute all early warning signals for a time series

        Args:
            time_series: Time series data
            regime_entropy: Optional HMM regime entropy for flickering signal

        Returns:
            DataFrame with all signals
        """
        signals = pd.DataFrame(index=time_series.index)

        # Critical slowing down
        signals['ar1'] = self.compute_ar1(time_series)

        # Variance increase
        signals['variance'] = self.compute_rolling_variance(time_series)
        signals['variance_ratio'] = self._compute_variance_ratio(signals['variance'])

        # Detrended fluctuation analysis
        signals['dfa_alpha'] = self.compute_dfa(time_series)

        # Flickering (if regime entropy provided)
        if regime_entropy is not None:
            signals['flickering'] = regime_entropy
            signals['flickering_ratio'] = self._compute_flickering_ratio(signals['flickering'])

        # Composite alert level
        signals['alert_level'] = self._compute_alert_level(signals)

        return signals

    def compute_ar1(
        self,
        time_series: pd.Series,
        detrend: bool = True
    ) -> pd.Series:
        """
        Compute rolling lag-1 autocorrelation (AR1)

        Increasing AR1 indicates critical slowing down

        Args:
            time_series: Input time series
            detrend: Whether to detrend before computing ACF

        Returns:
            Series of AR1 values
        """
        ar1_values = []

        for i in range(len(time_series)):
            if i < self.window:
                ar1_values.append(np.nan)
                continue

            # Extract window
            window_data = time_series.iloc[i - self.window:i].values

            # Remove NaN
            window_clean = window_data[~np.isnan(window_data)]

            if len(window_clean) < 10:
                ar1_values.append(np.nan)
                continue

            # Detrend if requested
            if detrend:
                # Linear detrend
                x = np.arange(len(window_clean))
                slope, intercept = np.polyfit(x, window_clean, 1)
                window_clean = window_clean - (slope * x + intercept)

            # Compute ACF
            try:
                acf_vals = acf(window_clean, nlags=1, fft=False)
                ar1 = acf_vals[1]
            except:
                ar1 = np.nan

            ar1_values.append(ar1)

        return pd.Series(ar1_values, index=time_series.index)

    def compute_rolling_variance(
        self,
        time_series: pd.Series,
        detrend: bool = True
    ) -> pd.Series:
        """
        Compute rolling variance

        Increasing variance indicates loss of stability

        Args:
            time_series: Input time series
            detrend: Whether to detrend first

        Returns:
            Series of variance values
        """
        if detrend:
            # Detrend using rolling linear fit
            detrended = self._rolling_detrend(time_series)
        else:
            detrended = time_series

        variance = detrended.rolling(window=self.window, center=True).var()

        return variance

    def compute_dfa(
        self,
        time_series: pd.Series,
        scales: Optional[list] = None
    ) -> pd.Series:
        """
        Compute Detrended Fluctuation Analysis (DFA) exponent

        DFA alpha > 0.5 indicates long-range correlations
        Increasing alpha indicates approaching transition

        Args:
            time_series: Input time series
            scales: Window scales to use (default: [4, 8, 16, 32])

        Returns:
            Series of DFA alpha values
        """
        if scales is None:
            scales = [4, 8, 16, 32]

        dfa_values = []

        for i in range(len(time_series)):
            if i < max(scales) * 2:
                dfa_values.append(np.nan)
                continue

            # Extract window
            window_data = time_series.iloc[max(0, i - self.window * 2):i].values
            window_clean = window_data[~np.isnan(window_data)]

            if len(window_clean) < max(scales) * 2:
                dfa_values.append(np.nan)
                continue

            try:
                alpha = self._compute_dfa_alpha(window_clean, scales)
            except:
                alpha = np.nan

            dfa_values.append(alpha)

        return pd.Series(dfa_values, index=time_series.index)

    def _compute_dfa_alpha(
        self,
        data: np.ndarray,
        scales: list
    ) -> float:
        """
        Compute DFA scaling exponent

        Args:
            data: Time series data
            scales: Window scales

        Returns:
            DFA alpha exponent
        """
        # Integrate the signal (cumulative sum of deviations from mean)
        y = np.cumsum(data - np.mean(data))

        fluctuations = []

        for scale in scales:
            # Divide into segments
            n_segments = len(y) // scale

            if n_segments < 2:
                continue

            F_scale = 0

            for seg in range(n_segments):
                segment = y[seg * scale:(seg + 1) * scale]
                x = np.arange(len(segment))

                # Fit linear trend
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)

                # Fluctuation
                F_scale += np.mean((segment - trend) ** 2)

            F_scale = np.sqrt(F_scale / n_segments)
            fluctuations.append(F_scale)

        if len(fluctuations) < 2:
            return np.nan

        # Power law: F(s) ~ s^alpha
        # log(F) = alpha * log(s) + const
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluct = np.log(fluctuations)

        # Fit line
        slope, _ = np.polyfit(log_scales, log_fluct, 1)

        return slope

    def _rolling_detrend(self, time_series: pd.Series) -> pd.Series:
        """Apply rolling linear detrend"""
        detrended = time_series.copy()

        for i in range(self.window, len(time_series)):
            window_data = time_series.iloc[i - self.window:i].values
            x = np.arange(len(window_data))

            # Fit line
            mask = ~np.isnan(window_data)
            if mask.sum() < 5:
                continue

            slope, intercept = np.polyfit(x[mask], window_data[mask], 1)
            trend_value = slope * (len(window_data) - 1) + intercept

            detrended.iloc[i] = time_series.iloc[i] - trend_value

        return detrended

    def _compute_variance_ratio(self, variance: pd.Series) -> pd.Series:
        """
        Compute variance ratio relative to baseline

        Baseline = median of first 50% of data
        """
        baseline_window = len(variance) // 2
        baseline = variance.iloc[:baseline_window].median()

        if baseline == 0 or np.isnan(baseline):
            return pd.Series(np.nan, index=variance.index)

        return variance / baseline

    def _compute_flickering_ratio(self, flickering: pd.Series) -> pd.Series:
        """Compute flickering ratio relative to baseline"""
        baseline_window = len(flickering) // 2
        baseline = np.nanmedian(flickering[:baseline_window])

        if baseline == 0 or np.isnan(baseline):
            return pd.Series(np.nan, index=flickering.index)

        return flickering / baseline

    def _compute_alert_level(self, signals: pd.DataFrame) -> pd.Series:
        """
        Compute composite alert level (0-3)

        0: No alert
        1: Yellow alert
        2: Orange alert
        3: Red alert

        Based on thresholds from config
        """
        alert = pd.Series(0, index=signals.index)

        # Yellow: AR1 > threshold OR variance ratio > threshold
        yellow_cond = (
            (signals['ar1'] > config.AR1_YELLOW_THRESHOLD) |
            (signals.get('variance_ratio', 0) > config.VARIANCE_MULTIPLIER_THRESHOLD)
        )
        alert[yellow_cond] = 1

        # Orange: AR1 > higher threshold AND variance high AND flickering high
        orange_cond = (
            (signals['ar1'] > config.AR1_ORANGE_THRESHOLD) &
            (signals.get('variance_ratio', 0) > config.VARIANCE_MULTIPLIER_THRESHOLD) &
            (signals.get('flickering_ratio', 0) > config.VARIANCE_MULTIPLIER_THRESHOLD)
        )
        alert[orange_cond] = 2

        # Red: Orange conditions met (will be combined with CSAI in main pipeline)
        # For single surface, we use orange as highest level
        # In multi-surface setup, this will be upgraded to red when CSAI is high

        return alert

    def detect_transitions(
        self,
        signals: pd.DataFrame,
        threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Detect probable regime transitions

        Args:
            signals: DataFrame of early warning signals
            threshold: AR1 threshold for detection

        Returns:
            DataFrame with transition events
        """
        # Transitions occur when AR1 crosses threshold
        ar1 = signals['ar1']
        crossings = (ar1 > threshold) & (ar1.shift(1) <= threshold)

        transitions = signals[crossings].copy()
        transitions['transition_detected'] = True

        return transitions
