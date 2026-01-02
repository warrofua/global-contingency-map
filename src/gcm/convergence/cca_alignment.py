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
        self.feature_masks: Dict = {}  # {view_name: mask of non-constant features}
        logger.info(f"Initialized {method.upper()} alignment with {self.n_components} components")

    def _remove_constant_features(
        self,
        X: np.ndarray,
        view_name: str
    ) -> Tuple[np.ndarray, int]:
        """
        Remove constant features (zero variance)

        Args:
            X: Feature matrix
            view_name: Name of view

        Returns:
            (X_filtered, n_removed)
        """
        variances = np.var(X, axis=0)
        non_constant_mask = variances > 1e-10

        n_removed = (~non_constant_mask).sum()
        if n_removed > 0:
            logger.warning(f"Removing {n_removed} constant features from {view_name}")

        self.feature_masks[view_name] = non_constant_mask
        return X[:, non_constant_mask], n_removed

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

        # Check for constant features and remove them
        X1_clean, removed_features1 = self._remove_constant_features(X1_clean, view1_name)
        X2_clean, removed_features2 = self._remove_constant_features(X2_clean, view2_name)

        if X1_clean.shape[1] == 0 or X2_clean.shape[1] == 0:
            logger.warning(f"All features are constant in {view1_name} or {view2_name}, cannot fit CCA")
            # Create dummy model that returns zeros
            self.models[(view1_name, view2_name)] = None
            # Don't overwrite scalers if they were already set by a previous successful fit
            # Just set them to None if they don't exist yet
            if view1_name not in self.scalers:
                self.scalers[view1_name] = None
            if view2_name not in self.scalers:
                self.scalers[view2_name] = None
            return self

        # Standardize
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        X1_scaled = scaler1.fit_transform(X1_clean)
        X2_scaled = scaler2.fit_transform(X2_clean)

        self.scalers[view1_name] = scaler1
        self.scalers[view2_name] = scaler2

        # Fit model
        if self.method == "cca":
            model = CCA(n_components=min(self.n_components, X1_clean.shape[1], X2_clean.shape[1]))
        elif self.method == "pls":
            model = PLSCanonical(n_components=min(self.n_components, X1_clean.shape[1], X2_clean.shape[1]))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        model.fit(X1_scaled, X2_scaled)

        self.models[(view1_name, view2_name)] = model

        # Compute canonical correlations
        X1_c, X2_c = model.transform(X1_scaled, X2_scaled)
        correlations = [
            np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1]
            for i in range(model.n_components)
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
        scaler1 = self.scalers.get(view1_name)
        scaler2 = self.scalers.get(view2_name)

        # Handle case where model is None (all constant features) or scalers are None
        if model is None or scaler1 is None or scaler2 is None:
            logger.warning(f"Model or scalers are None for views ({view1_name}, {view2_name}), returning NaN")
            Z1 = np.full((len(X1), self.n_components), np.nan)
            Z2 = np.full((len(X2), self.n_components), np.nan)
            return Z1, Z2

        # Handle NaN
        mask = ~(np.isnan(X1).any(axis=1) | np.isnan(X2).any(axis=1))

        Z1 = np.full((len(X1), self.n_components), np.nan)
        Z2 = np.full((len(X2), self.n_components), np.nan)

        if mask.sum() > 0:
            # Filter constant features
            X1_filtered = X1[mask]
            X2_filtered = X2[mask]

            if view1_name in self.feature_masks:
                X1_filtered = X1_filtered[:, self.feature_masks[view1_name]]
            if view2_name in self.feature_masks:
                X2_filtered = X2_filtered[:, self.feature_masks[view2_name]]

            X1_scaled = scaler1.transform(X1_filtered)
            X2_scaled = scaler2.transform(X2_filtered)

            Z1_clean, Z2_clean = model.transform(X1_scaled, X2_scaled)

            # Pad to n_components if needed
            if Z1_clean.shape[1] < self.n_components:
                Z1_clean = np.pad(Z1_clean, ((0, 0), (0, self.n_components - Z1_clean.shape[1])), constant_values=0)
            if Z2_clean.shape[1] < self.n_components:
                Z2_clean = np.pad(Z2_clean, ((0, 0), (0, self.n_components - Z2_clean.shape[1])), constant_values=0)

            Z1[mask] = Z1_clean[:, :self.n_components]
            Z2[mask] = Z2_clean[:, :self.n_components]

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
        with np.errstate(all='ignore'):  # Suppress warnings for all-NaN slices
            Z1 = np.nanmean([Z1_from_12, Z1_from_13], axis=0)
            Z2 = np.nanmean([Z2_from_12, Z2_from_23], axis=0)
            Z3 = np.nanmean([Z3_from_23, Z3_from_13], axis=0)

        # Log warnings for views with all NaN values
        nan_count1 = np.isnan(Z1).all(axis=1).sum()
        nan_count2 = np.isnan(Z2).all(axis=1).sum()
        nan_count3 = np.isnan(Z3).all(axis=1).sum()

        if nan_count1 > 0:
            logger.warning(f"{view_names[0]} has {nan_count1}/{len(Z1)} timesteps with all NaN latent values")
        if nan_count2 > 0:
            logger.warning(f"{view_names[1]} has {nan_count2}/{len(Z2)} timesteps with all NaN latent values")
        if nan_count3 > 0:
            logger.warning(f"{view_names[2]} has {nan_count3}/{len(Z3)} timesteps with all NaN latent values")

        return Z1, Z2, Z3

    def fit_transform_three_view(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray,
        view_names: Tuple[str, str, str] = ("financial", "narrative", "social")
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit and transform three views in one step

        Args:
            X1, X2, X3: Feature matrices
            view_names: View names

        Returns:
            (Z1, Z2, Z3) latent representations
        """
        self.fit_three_view(X1, X2, X3, view_names)
        return self.transform_three_view(X1, X2, X3, view_names)

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
