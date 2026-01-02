"""
Gaussian cluster modeling for narrative embeddings

Models each narrative cluster as N(μ_k(t), Σ_k(t)) in embedding space
Enables Fisher-Rao distance and KL divergence computation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cosine

from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class NarrativeEmbeddingClusters:
    """
    Embed narratives and model clusters as Gaussian distributions

    Workflow:
    1. Embed texts using sentence-transformers
    2. Cluster embeddings (KMeans or GMM)
    3. Model each cluster as Gaussian: N(μ_k, Σ_k)
    4. Compute drift, coherence, and inter-cluster distances
    """

    def __init__(
        self,
        embedding_model: str = None,
        n_clusters: int = None
    ):
        """
        Initialize embedding cluster model

        Args:
            embedding_model: Sentence transformer model name
            n_clusters: Number of clusters
        """
        self.model_name = embedding_model or config.EMBEDDING_MODEL
        self.n_clusters = n_clusters or config.N_CLUSTERS

        logger.info(f"Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.cluster_model = None
        self.cluster_params: Dict[int, Dict] = {}  # {cluster_id: {mu, sigma}}

        logger.info(f"Initialized narrative embeddings with {self.n_clusters} clusters")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts into vector space

        Args:
            texts: List of text strings

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings

    def fit_clusters(
        self,
        embeddings: np.ndarray,
        method: str = "kmeans"
    ) -> np.ndarray:
        """
        Fit cluster model to embeddings

        Args:
            embeddings: Embedding array
            method: "kmeans" or "gmm"

        Returns:
            Cluster labels
        """
        logger.info(f"Fitting {method} with {self.n_clusters} clusters...")

        if method == "kmeans":
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            labels = self.cluster_model.fit_predict(embeddings)

        elif method == "gmm":
            self.cluster_model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=42,
                covariance_type='full'
            )
            labels = self.cluster_model.fit_predict(embeddings)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute Gaussian parameters for each cluster
        self._compute_cluster_gaussians(embeddings, labels)

        logger.info("Clustering complete")
        return labels

    def _compute_cluster_gaussians(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ):
        """
        Compute Gaussian parameters (μ, Σ) for each cluster

        Args:
            embeddings: Embedding array
            labels: Cluster labels
        """
        self.cluster_params = {}

        for k in range(self.n_clusters):
            mask = labels == k
            cluster_embeddings = embeddings[mask]

            if len(cluster_embeddings) < 2:
                logger.warning(f"Cluster {k} has insufficient samples")
                continue

            # Compute mean and covariance
            mu = cluster_embeddings.mean(axis=0)
            sigma = np.cov(cluster_embeddings.T)

            # Add regularization to ensure positive definite
            sigma = sigma + np.eye(self.embedding_dim) * 1e-6

            self.cluster_params[k] = {
                'mu': mu,
                'sigma': sigma,
                'n_samples': len(cluster_embeddings)
            }

        logger.info(f"Computed Gaussian parameters for {len(self.cluster_params)} clusters")

    def compute_cluster_drift(
        self,
        mu_prev: np.ndarray,
        mu_curr: np.ndarray
    ) -> float:
        """
        Compute drift between cluster centroids

        Args:
            mu_prev: Previous centroid
            mu_curr: Current centroid

        Returns:
            Euclidean distance
        """
        return np.linalg.norm(mu_curr - mu_prev)

    def compute_cluster_coherence(self, sigma: np.ndarray) -> float:
        """
        Compute intra-cluster coherence

        Coherence = tr(Σ) / d (average variance per dimension)
        Lower values = more coherent

        Args:
            sigma: Covariance matrix

        Returns:
            Coherence metric
        """
        return np.trace(sigma) / self.embedding_dim

    def compute_fisher_rao_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray
    ) -> float:
        """
        Compute Fisher-Rao distance between two Gaussians

        For multivariate Gaussians, we use approximation:
        d_FR ≈ √(0.5 * (μ1-μ2)^T * Σ_avg^{-1} * (μ1-μ2))

        Args:
            mu1, sigma1: First Gaussian parameters
            mu2, sigma2: Second Gaussian parameters

        Returns:
            Fisher-Rao distance (approximation)
        """
        # Average covariance
        sigma_avg = (sigma1 + sigma2) / 2

        # Regularize
        sigma_avg = sigma_avg + np.eye(self.embedding_dim) * 1e-6

        # Compute distance
        diff = mu1 - mu2

        try:
            # Mahalanobis distance
            sigma_inv = np.linalg.inv(sigma_avg)
            distance = np.sqrt(0.5 * diff.T @ sigma_inv @ diff)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean if matrix is singular
            logger.warning("Singular covariance matrix, using Euclidean distance")
            distance = np.linalg.norm(diff)

        return distance

    def compute_kl_divergence(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray
    ) -> float:
        """
        Compute KL divergence between two Gaussians

        KL(N1 || N2) = 0.5 * [log(|Σ2|/|Σ1|) - d + tr(Σ2^{-1}Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1)]

        Args:
            mu1, sigma1: First Gaussian (P)
            mu2, sigma2: Second Gaussian (Q)

        Returns:
            KL divergence KL(P || Q)
        """
        d = self.embedding_dim
        diff = mu2 - mu1

        try:
            sigma2_inv = np.linalg.inv(sigma2)

            # Log determinants
            sign1, logdet1 = np.linalg.slogdet(sigma1)
            sign2, logdet2 = np.linalg.slogdet(sigma2)

            kl = 0.5 * (
                logdet2 - logdet1 - d +
                np.trace(sigma2_inv @ sigma1) +
                diff.T @ sigma2_inv @ diff
            )

            return kl

        except np.linalg.LinAlgError:
            logger.warning("Singular covariance in KL computation")
            return np.inf

    def extract_features(
        self,
        narratives_df: pd.DataFrame,
        text_column: str = 'text',
        timestamp_column: str = 'timestamp',
        time_window: str = '1D'
    ) -> pd.DataFrame:
        """
        Extract narrative features over time

        Args:
            narratives_df: DataFrame with narratives
            text_column: Column containing text
            timestamp_column: Column containing timestamps
            time_window: Aggregation window (e.g., '1D', '1H')

        Returns:
            DataFrame with features per time window
        """
        # Embed all texts
        texts = narratives_df[text_column].tolist()
        embeddings = self.embed_texts(texts)

        # Add embeddings to dataframe
        narratives_df = narratives_df.copy()
        narratives_df['embedding'] = list(embeddings)

        # Group by time window
        narratives_df.set_index(timestamp_column, inplace=True)
        grouped = narratives_df.groupby(pd.Grouper(freq=time_window))

        features_list = []
        prev_cluster_params = None

        for timestamp, group in grouped:
            if len(group) < 10:  # Need minimum samples
                continue

            group_embeddings = np.vstack(group['embedding'].values)

            # Fit clusters
            labels = self.fit_clusters(group_embeddings, method='kmeans')

            # Extract features
            features = {
                'timestamp': timestamp,
                'n_narratives': len(group)
            }

            # Cluster-level metrics
            cluster_drifts = []
            cluster_coherences = []
            inter_cluster_distances = []

            for k, params in self.cluster_params.items():
                mu = params['mu']
                sigma = params['sigma']

                # Coherence
                coherence = self.compute_cluster_coherence(sigma)
                cluster_coherences.append(coherence)

                # Drift (if we have previous parameters)
                if prev_cluster_params and k in prev_cluster_params:
                    mu_prev = prev_cluster_params[k]['mu']
                    drift = self.compute_cluster_drift(mu_prev, mu)
                    cluster_drifts.append(drift)

            # Inter-cluster distances
            cluster_ids = list(self.cluster_params.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    k1, k2 = cluster_ids[i], cluster_ids[j]
                    params1 = self.cluster_params[k1]
                    params2 = self.cluster_params[k2]

                    dist = self.compute_fisher_rao_distance(
                        params1['mu'], params1['sigma'],
                        params2['mu'], params2['sigma']
                    )
                    inter_cluster_distances.append(dist)

            # Aggregate metrics
            features['avg_cluster_coherence'] = np.mean(cluster_coherences) if cluster_coherences else np.nan
            features['avg_cluster_drift'] = np.mean(cluster_drifts) if cluster_drifts else np.nan
            features['avg_inter_cluster_distance'] = np.mean(inter_cluster_distances) if inter_cluster_distances else np.nan

            # Topic entropy (distribution of cluster sizes)
            cluster_sizes = [params['n_samples'] for params in self.cluster_params.values()]
            cluster_probs = np.array(cluster_sizes) / sum(cluster_sizes)
            topic_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
            features['topic_entropy'] = topic_entropy

            features_list.append(features)

            # Save for next iteration
            prev_cluster_params = self.cluster_params.copy()

        features_df = pd.DataFrame(features_list)
        features_df.set_index('timestamp', inplace=True)

        logger.info(f"Extracted features for {len(features_df)} time windows")

        return features_df
