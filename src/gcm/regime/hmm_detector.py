"""
Hidden Markov Model (HMM) regime detection
Identifies latent market regimes from financial features
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from hmmlearn import hmm
from scipy.stats import entropy

from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class HMMRegimeDetector:
    """
    HMM-based regime detection

    Learns latent states from feature time series and detects regime transitions
    """

    def __init__(self, n_regimes: int = None, random_state: int = 42):
        """
        Initialize HMM regime detector

        Args:
            n_regimes: Number of latent regimes (default: config.N_REGIMES)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes or config.N_REGIMES
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.feature_names: Optional[list] = None
        logger.info(f"Initialized HMM detector with {self.n_regimes} regimes")

    def fit(
        self,
        features: pd.DataFrame,
        feature_cols: Optional[list] = None
    ) -> 'HMMRegimeDetector':
        """
        Fit HMM to feature time series

        Args:
            features: DataFrame with features as columns
            feature_cols: List of column names to use (default: all numeric)

        Returns:
            Self (fitted detector)
        """
        # Select feature columns
        if feature_cols is None:
            feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names = feature_cols

        # Prepare data
        X = features[feature_cols].values

        # Remove NaN rows
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]

        if len(X_clean) < 10:
            raise ValueError("Insufficient data for HMM training (need at least 10 samples)")

        # Standardize features
        self.feature_mean_ = X_clean.mean(axis=0)
        self.feature_std_ = X_clean.std(axis=0) + 1e-8  # Avoid division by zero
        X_scaled = (X_clean - self.feature_mean_) / self.feature_std_

        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=config.HMM_N_ITER,
            random_state=self.random_state,
            verbose=False
        )

        logger.info(f"Fitting HMM with {len(X_scaled)} samples...")
        self.model.fit(X_scaled)

        # Check for degenerate transition matrix
        self._validate_transition_matrix()

        logger.info("HMM training complete")
        logger.info(f"Log-likelihood: {self.model.score(X_scaled):.2f}")

        return self

    def predict(
        self,
        features: pd.DataFrame,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Predict regime states

        Args:
            features: DataFrame with features
            return_proba: If True, return state probabilities instead of hard labels

        Returns:
            Array of state labels or probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare data
        X = features[self.feature_names].values
        mask = ~np.isnan(X).any(axis=1)
        X_scaled = (X - self.feature_mean_) / self.feature_std_

        if return_proba:
            # Return posterior probabilities
            proba = np.full((len(X), self.n_regimes), np.nan)
            proba[mask] = self.model.predict_proba(X_scaled[mask])
            return proba
        else:
            # Return hard state assignments
            states = np.full(len(X), -1)
            states[mask] = self.model.predict(X_scaled[mask])
            return states

    def get_transition_matrix(self) -> np.ndarray:
        """Get learned transition matrix"""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.transmat_

    def get_regime_statistics(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute statistics for each regime

        Args:
            features: DataFrame with features

        Returns:
            DataFrame with regime statistics
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        states = self.predict(features)
        X = features[self.feature_names].values

        stats = []
        for regime in range(self.n_regimes):
            mask = states == regime
            if mask.sum() == 0:
                continue

            X_regime = X[mask]
            regime_stats = {
                'regime': regime,
                'count': mask.sum(),
                'frequency': mask.sum() / len(states)
            }

            # Feature means in this regime
            for i, col in enumerate(self.feature_names):
                regime_stats[f'{col}_mean'] = np.nanmean(X_regime[:, i])
                regime_stats[f'{col}_std'] = np.nanstd(X_regime[:, i])

            stats.append(regime_stats)

        return pd.DataFrame(stats)

    def compute_regime_entropy(
        self,
        features: pd.DataFrame,
        window: int = None
    ) -> np.ndarray:
        """
        Compute entropy of regime posterior distribution (flickering indicator)

        High entropy indicates uncertainty / regime switching

        Args:
            features: DataFrame with features
            window: Rolling window size (default: config.ROLLING_WINDOW)

        Returns:
            Array of entropy values
        """
        if window is None:
            window = config.ROLLING_WINDOW

        proba = self.predict(features, return_proba=True)

        # Compute entropy at each timepoint
        entropy_values = np.array([
            entropy(p) if not np.any(np.isnan(p)) else np.nan
            for p in proba
        ])

        # Rolling average for smoothing
        entropy_series = pd.Series(entropy_values)
        entropy_smooth = entropy_series.rolling(window, center=True).mean().values

        return entropy_smooth

    def get_transition_probabilities(
        self,
        current_state: int,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Compute transition probabilities for given horizon

        Args:
            current_state: Current regime state
            horizon: Number of steps ahead

        Returns:
            Array of probabilities for each state at horizon
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        # Matrix power for multi-step transitions
        trans_matrix = self.model.transmat_
        trans_horizon = np.linalg.matrix_power(trans_matrix, horizon)

        return trans_horizon[current_state]

    def _validate_transition_matrix(self):
        """Check for degenerate transition matrix"""
        trans_matrix = self.model.transmat_

        # Check if any state is absorbing (probability ~1.0)
        for i in range(self.n_regimes):
            if trans_matrix[i, i] > 0.99:
                logger.warning(f"State {i} appears to be absorbing (self-transition prob: {trans_matrix[i, i]:.3f})")

        # Check if matrix is too sparse
        n_significant = (trans_matrix > 0.01).sum()
        if n_significant < self.n_regimes * 2:
            logger.warning(f"Transition matrix is very sparse ({n_significant} significant transitions)")

    def save(self, path: str):
        """Save trained model"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_mean': self.feature_mean_,
                'feature_std': self.feature_std_,
                'n_regimes': self.n_regimes
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load trained model"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.feature_mean_ = data['feature_mean']
        self.feature_std_ = data['feature_std']
        self.n_regimes = data['n_regimes']
        logger.info(f"Model loaded from {path}")
