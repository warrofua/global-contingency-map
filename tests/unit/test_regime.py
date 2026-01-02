"""
Unit tests for regime detection
"""
import pytest
import numpy as np
import pandas as pd

from gcm.regime.hmm_detector import HMMRegimeDetector


class TestHMMDetector:
    """Test HMM regime detection"""

    def test_initialization(self):
        """Test detector initialization"""
        detector = HMMRegimeDetector(n_regimes=3)
        assert detector.n_regimes == 3

    def test_fit_predict(self):
        """Test HMM fit and predict"""
        detector = HMMRegimeDetector(n_regimes=2)

        # Create synthetic data with two regimes
        np.random.seed(42)
        data1 = np.random.normal(0, 1, (50, 3))
        data2 = np.random.normal(5, 1, (50, 3))
        data = np.vstack([data1, data2])

        df = pd.DataFrame(data, columns=['f1', 'f2', 'f3'])

        # Fit
        detector.fit(df)

        # Predict
        states = detector.predict(df)

        assert len(states) == 100
        assert len(np.unique(states[states >= 0])) <= 2

    def test_transition_matrix(self):
        """Test transition matrix properties"""
        detector = HMMRegimeDetector(n_regimes=2)

        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        df = pd.DataFrame(data, columns=['f1', 'f2', 'f3'])

        detector.fit(df)

        trans_matrix = detector.get_transition_matrix()

        # Check shape
        assert trans_matrix.shape == (2, 2)

        # Check rows sum to 1
        row_sums = trans_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])
