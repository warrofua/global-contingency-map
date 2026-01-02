"""
Unit tests for convergence layer
"""
import pytest
import numpy as np
import pandas as pd

from gcm.convergence.cca_alignment import MultiViewAlignment, CSAIComputer


class TestMultiViewAlignment:
    """Test multi-view CCA alignment"""

    def test_initialization(self):
        """Test alignment initialization"""
        aligner = MultiViewAlignment(n_components=5, method="cca")
        assert aligner.n_components == 5
        assert aligner.method == "cca"

    def test_two_view_alignment(self):
        """Test two-view CCA"""
        aligner = MultiViewAlignment(n_components=3, method="cca")

        # Create correlated views
        np.random.seed(42)
        n = 100
        latent = np.random.normal(0, 1, (n, 3))

        X1 = latent + np.random.normal(0, 0.5, (n, 3))
        X2 = latent + np.random.normal(0, 0.5, (n, 3))

        # Fit
        aligner.fit_two_view(X1, X2, "view1", "view2")

        # Transform
        Z1, Z2 = aligner.transform_two_view(X1, X2, "view1", "view2")

        assert Z1.shape == (n, 3)
        assert Z2.shape == (n, 3)

    def test_three_view_alignment(self):
        """Test three-view alignment"""
        aligner = MultiViewAlignment(n_components=2, method="cca")

        np.random.seed(42)
        n = 50
        latent = np.random.normal(0, 1, (n, 2))

        X1 = latent + np.random.normal(0, 0.3, (n, 2))
        X2 = latent + np.random.normal(0, 0.3, (n, 2))
        X3 = latent + np.random.normal(0, 0.3, (n, 2))

        # Fit
        aligner.fit_three_view(X1, X2, X3)

        # Transform
        Z1, Z2, Z3 = aligner.transform_three_view(X1, X2, X3)

        assert Z1.shape == (n, 2)
        assert Z2.shape == (n, 2)
        assert Z3.shape == (n, 2)


class TestCSAIComputer:
    """Test CSAI computation"""

    def test_initialization(self):
        """Test CSAI computer initialization"""
        csai = CSAIComputer()
        assert csai is not None

    def test_drift_computation(self):
        """Test drift vector computation"""
        csai = CSAIComputer()

        # Create trajectory
        Z = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0]
        ])

        drift = csai.compute_drift_vector(Z, window=1)

        # Drift should be [1, 0] for each step
        assert drift[1][0] == pytest.approx(1.0)
        assert drift[1][1] == pytest.approx(0.0)

    def test_cosine_similarity(self):
        """Test cosine similarity"""
        csai = CSAIComputer()

        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        v3 = np.array([0, 1, 0])

        # Same direction
        sim_same = csai.cosine_similarity(v1, v2)
        assert sim_same == pytest.approx(1.0)

        # Orthogonal
        sim_orth = csai.cosine_similarity(v1, v3)
        assert sim_orth == pytest.approx(0.0)

    def test_two_surface_csai(self):
        """Test two-surface CSAI"""
        csai = CSAIComputer()

        # Create aligned trajectories
        t = np.linspace(0, 10, 50)
        Z1 = np.column_stack([np.sin(t), np.cos(t)])
        Z2 = np.column_stack([np.sin(t), np.cos(t)])

        csai_values = csai.compute_two_surface_csai(Z1, Z2, window=5)

        # CSAI should be close to 1 (aligned)
        assert np.nanmean(csai_values) > 0.9
