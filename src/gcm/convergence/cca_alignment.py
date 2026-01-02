"""
Multi-view alignment using CCA and PLS

Learns shared latent space z(t) from multiple view feature vectors
Phase 1: CCA/PLS (linear)
Phase 2+: Contrastive learning (non-linear)
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.preprocessing import StandardScaler

from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class MultiViewAlignment:
    """
    Align multiple views into shared latent space

    Methods:
    - CCA (Canonical Correlation Analysis)
    - PLS (Partial Least Squares)

    For N views, learns projections:
    z^X_t = W_X @ x^X_t for each view X

    Optimizes correlation between projected views
    """

    def __init__(
        self,
        n_components: int = None,
        method: str = "cca"
    ):
        """
        Initialize multi-view alignment

        Args:
            n_components: Dimension of shared latent space
            method: "cca" or "pls"
        """
        self.n_components = n_components or config.CCA_N_COMPONENTS
        self.method = method
        self.models: Dict = {}  # {(view1, view2): model}
        self.scalers: Dict = {}  # {view_name: scaler}
        logger.info(f"Initialized {method.upper()} alignment with {self.n_components} components")

    def fit_two_view(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        view1_name: str = "view1",
        view2_name: str = "view2"
    ) -> 'MultiViewAlignment':
        """
        Fit CCA/PLS for two views

        Args:
            X1: First view features (n_samples, n_features1)
            X2: Second view features (n_samples, n_features2)
            view1_name: Name of first view
            view2_name: Name of second view

        Returns:
            Self (fitted)
        """
        # Remove NaN rows
        mask = ~(np.isnan(X1).any(axis=1) | np.isnan(X2).any(axis=1))
        X1_clean = X1[mask]
        X2_clean = X2[mask]

        if len(X1_clean) < 10:
            raise ValueError("Insufficient samples for CCA/PLS (need at least 10)")

        logger.info(f"Fitting {self.method.upper()} with {len(X1_clean)} samples")

        # Standardize
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        X1_scaled = scaler1.fit_transform(X1_clean)
        X2_scaled = scaler2.fit_transform(X2_clean)

        self.scalers[view1_name] = scaler1
        self.scalers[view2_name] = scaler2

        # Fit model
        if self.method == "cca":
            model = CCA(n_components=self.n_components)
        elif self.method == "pls":
            model = PLSCanonical(n_components=self.n_components)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        model.fit(X1_scaled, X2_scaled)

        self.models[(view1_name, view2_name)] = model

        # Compute canonical correlations
        X1_c, X2_c = model.transform(X1_scaled, X2_scaled)
        correlations = [
            np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1]
            for i in range(self.n_components)
        ]

        logger.info(f"Canonical correlations: {correlations}")

        return self

    def transform_two_view(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        view1_name: str = "view1",
        view2_name: str = "view2"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform views to shared latent space

        Args:
            X1, X2: View features
            view1_name, view2_name: View names

        Returns:
            (Z1, Z2) transformed features in shared space
        """
        key = (view1_name, view2_name)
        if key not in self.models:
            raise ValueError(f"Model not fitted for views ({view1_name}, {view2_name})")

        model = self.models[key]
        scaler1 = self.scalers[view1_name]
        scaler2 = self.scalers[view2_name]

        # Handle NaN
        mask = ~(np.isnan(X1).any(axis=1) | np.isnan(X2).any(axis=1))

        Z1 = np.full((len(X1), self.n_components), np.nan)
        Z2 = np.full((len(X2), self.n_components), np.nan)

        if mask.sum() > 0:
            X1_scaled = scaler1.transform(X1[mask])
            X2_scaled = scaler2.transform(X2[mask])

            Z1_clean, Z2_clean = model.transform(X1_scaled, X2_scaled)

            Z1[mask] = Z1_clean
            Z2[mask] = Z2_clean

        return Z1, Z2

    def fit_transform_two_view(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        view1_name: str = "view1",
        view2_name: str = "view2"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step"""
        self.fit_two_view(X1, X2, view1_name, view2_name)
        return self.transform_two_view(X1, X2, view1_name, view2_name)

    def fit_three_view(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray,
        view_names: Tuple[str, str, str] = ("financial", "narrative", "social")
    ) -> 'MultiViewAlignment':
        """
        Fit three-view alignment

        Strategy: Fit pairwise CCA and average latent representations

        Args:
            X1, X2, X3: Feature matrices
            view_names: Tuple of view names

        Returns:
            Self (fitted)
        """
        logger.info("Fitting three-view alignment...")

        # Fit all pairs
        self.fit_two_view(X1, X2, view_names[0], view_names[1])
        self.fit_two_view(X2, X3, view_names[1], view_names[2])
        self.fit_two_view(X1, X3, view_names[0], view_names[2])

        logger.info("Three-view alignment complete")

        return self

    def transform_three_view(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray,
        view_names: Tuple[str, str, str] = ("financial", "narrative", "social")
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform three views to shared latent space

        Averages pairwise transformations for each view

        Args:
            X1, X2, X3: Feature matrices
            view_names: View names

        Returns:
            (Z1, Z2, Z3) latent representations
        """
        # Get all pairwise transformations
        Z1_from_12, Z2_from_12 = self.transform_two_view(X1, X2, view_names[0], view_names[1])
        Z2_from_23, Z3_from_23 = self.transform_two_view(X2, X3, view_names[1], view_names[2])
        Z1_from_13, Z3_from_13 = self.transform_two_view(X1, X3, view_names[0], view_names[2])

        # Average overlapping transformations
        Z1 = np.nanmean([Z1_from_12, Z1_from_13], axis=0)
        Z2 = np.nanmean([Z2_from_12, Z2_from_23], axis=0)
        Z3 = np.nanmean([Z3_from_23, Z3_from_13], axis=0)

        return Z1, Z2, Z3

    def get_canonical_correlations(
        self,
        view1_name: str,
        view2_name: str
    ) -> np.ndarray:
        """
        Get canonical correlations for a view pair

        Args:
            view1_name, view2_name: View names

        Returns:
            Array of canonical correlations
        """
        key = (view1_name, view2_name)
        if key not in self.models:
            raise ValueError(f"Model not fitted for views ({view1_name}, {view2_name})")

        model = self.models[key]

        # Get training data correlations from model
        # Note: This requires accessing training data, so we return weights instead
        # True correlations would need to be computed on training data

        return model.x_weights_[:, 0]  # Placeholder


class CSAIComputer:
    """
    Cross-Surface Alignment Index (CSAI) computation

    CSAI_t = cos(Δz^F, Δz^N) · cos(Δz^N, Δz^S) · cos(Δz^S, Δz^F)

    Where Δz^X_t is drift vector in shared latent space
    """

    def __init__(self):
        """Initialize CSAI computer"""
        logger.info("Initialized CSAI computer")

    def compute_drift_vector(
        self,
        Z: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Compute drift vector from latent trajectory

        Δz_t = z_t - z_{t-window}

        Args:
            Z: Latent trajectory (n_timesteps, n_components)
            window: Lookback window

        Returns:
            Drift vectors (n_timesteps, n_components)
        """
        drift = np.full_like(Z, np.nan)

        for i in range(window, len(Z)):
            drift[i] = Z[i] - Z[i - window]

        return drift

    def cosine_similarity(
        self,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between vectors

        Args:
            v1, v2: Vectors

        Returns:
            Cosine similarity
        """
        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
            return np.nan

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return np.nan

        return np.dot(v1, v2) / (norm1 * norm2)

    def compute_two_surface_csai(
        self,
        Z1: np.ndarray,
        Z2: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Compute two-surface CSAI

        CSAI_t = cos(Δz^1_t, Δz^2_t)

        Args:
            Z1, Z2: Latent trajectories
            window: Drift window

        Returns:
            CSAI time series
        """
        drift1 = self.compute_drift_vector(Z1, window)
        drift2 = self.compute_drift_vector(Z2, window)

        csai = np.array([
            self.cosine_similarity(drift1[i], drift2[i])
            for i in range(len(drift1))
        ])

        return csai

    def compute_three_surface_csai(
        self,
        Z1: np.ndarray,
        Z2: np.ndarray,
        Z3: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Compute three-surface CSAI

        CSAI_t = cos(Δz^1, Δz^2) · cos(Δz^2, Δz^3) · cos(Δz^3, Δz^1)

        Args:
            Z1, Z2, Z3: Latent trajectories
            window: Drift window

        Returns:
            CSAI time series
        """
        drift1 = self.compute_drift_vector(Z1, window)
        drift2 = self.compute_drift_vector(Z2, window)
        drift3 = self.compute_drift_vector(Z3, window)

        csai = np.array([
            self.cosine_similarity(drift1[i], drift2[i]) *
            self.cosine_similarity(drift2[i], drift3[i]) *
            self.cosine_similarity(drift3[i], drift1[i])
            for i in range(len(drift1))
        ])

        return csai
