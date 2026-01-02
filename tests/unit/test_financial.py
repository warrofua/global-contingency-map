"""
Unit tests for financial surface components
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from gcm.financial.spd_extractor import SPDExtractor
from gcm.financial.features import FinancialFeatureExtractor


class TestSPDExtractor:
    """Test SPD extraction"""

    def test_initialization(self):
        """Test SPD extractor initialization"""
        extractor = SPDExtractor(ticker="SPY")
        assert extractor.ticker == "SPY"

    def test_spd_moments(self):
        """Test SPD moment computation"""
        extractor = SPDExtractor()

        # Mock SPD
        strikes = np.linspace(300, 500, 100)
        spot = 400

        # Gaussian SPD
        spd = np.exp(-(strikes - spot)**2 / (2 * 20**2))
        spd = spd / np.trapz(spd, strikes)

        moments = extractor.compute_spd_moments(strikes, spd, spot)

        assert 'mean' in moments
        assert 'variance' in moments
        assert 'skewness' in moments
        assert 'kurtosis' in moments
        assert moments['mean'] == pytest.approx(spot, abs=5)


class TestFinancialFeatures:
    """Test financial feature extraction"""

    def test_initialization(self):
        """Test feature extractor initialization"""
        extractor = FinancialFeatureExtractor(ticker="SPY")
        assert extractor.ticker == "SPY"

    def test_nan_features(self):
        """Test NaN feature generation"""
        extractor = FinancialFeatureExtractor()
        timestamp = datetime.now()
        features = extractor._get_nan_features(timestamp)

        assert features['timestamp'] == timestamp
        assert np.isnan(features['tail_mass_total'])
