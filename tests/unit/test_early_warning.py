"""
Unit tests for early warning signals
"""
import pytest
import numpy as np
import pandas as pd

from gcm.early_warning.signals import EarlyWarningSignals


class TestEarlyWarningSignals:
    """Test early warning signal computation"""

    def test_initialization(self):
        """Test signal detector initialization"""
        ews = EarlyWarningSignals(window=20)
        assert ews.window == 20

    def test_ar1_computation(self):
        """Test AR1 autocorrelation"""
        ews = EarlyWarningSignals(window=20)

        # Create AR(1) process
        np.random.seed(42)
        n = 100
        phi = 0.8
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = phi * data[i-1] + np.random.normal(0, 1)

        ts = pd.Series(data)
        ar1 = ews.compute_ar1(ts, detrend=False)

        # AR1 should be close to phi in later windows
        assert np.nanmean(ar1[-20:]) == pytest.approx(phi, abs=0.2)

    def test_variance_computation(self):
        """Test rolling variance"""
        ews = EarlyWarningSignals(window=20)

        # Create time series with increasing variance
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(0, 3, 50)
        ])

        ts = pd.Series(data)
        variance = ews.compute_rolling_variance(ts, detrend=False)

        # Variance should increase in second half
        assert np.nanmean(variance[-20:]) > np.nanmean(variance[20:40])

    def test_alert_levels(self):
        """Test alert level computation"""
        ews = EarlyWarningSignals(window=10)

        # Create mock signals
        signals = pd.DataFrame({
            'ar1': [0.5, 0.75, 0.85],
            'variance_ratio': [1.0, 2.5, 3.0],
            'flickering_ratio': [1.0, 1.5, 2.5]
        })

        alert = ews._compute_alert_level(signals)

        # Check alert progression
        assert alert.iloc[0] == 0  # No alert
        assert alert.iloc[1] >= 1  # Yellow
        assert alert.iloc[2] >= 2  # Orange
