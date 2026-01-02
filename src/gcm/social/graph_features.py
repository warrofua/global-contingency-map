"""
Social network graph construction and feature extraction

Extracts coalition topology metrics from social graphs
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class SocialGraphFeatures:
    """
    Construct and analyze social network graphs

    Features (x^S_t):
    - Modularity Q: Community structure strength
    - Assortativity r: Similar-degree connection preference
    - Bridge count: Inter-community edges
    - k-core decomposition: Elite network density
    - Giant component fraction: Network connectivity
    - Cascade potential: SIS/SIR spreading rate
    """

    def __init__(self):
        """Initialize social graph analyzer"""
        logger.info("Initialized social graph features")

    def create_graph_from_edges(
        self,
        edges: List[Tuple],
        directed: bool = False,
        weighted: bool = True
    ) -> nx.Graph:
        """
        Create graph from edge list

        Args:
            edges: List of (source, target) or (source, target, weight) tuples
            directed: Whether graph is directed
            weighted: Whether edges have weights

        Returns:
            NetworkX graph
        """
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        if weighted and len(edges) > 0 and len(edges[0]) == 3:
            G.add_weighted_edges_from(edges)
        else:
            G.add_edges_from(edges)

        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        return G

    def create_mock_graph(
        self,
        n_nodes: int = 100,
        n_communities: int = 5,
        p_within: float = 0.3,
        p_between: float = 0.05
    ) -> nx.Graph:
        """
        Create mock social graph with community structure

        Args:
            n_nodes: Number of nodes
            n_communities: Number of communities
            p_within: Edge probability within community
            p_between: Edge probability between communities

        Returns:
            NetworkX graph with communities
        """
        # Create stochastic block model
        sizes = [n_nodes // n_communities] * n_communities
        probs = [[p_within if i == j else p_between
                 for j in range(n_communities)]
                for i in range(n_communities)]

        G = nx.stochastic_block_model(sizes, probs, seed=42)

        logger.info(f"Created mock graph with {G.number_of_nodes()} nodes in {n_communities} communities")

        return G

    def compute_modularity(
        self,
        G: nx.Graph,
        communities: Optional[List[set]] = None
    ) -> float:
        """
        Compute modularity Q

        Q ∈ [-0.5, 1]: Higher = stronger community structure

        Args:
            G: NetworkX graph
            communities: List of node sets (auto-detect if None)

        Returns:
            Modularity value
        """
        if communities is None:
            # Detect communities using Louvain
            try:
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(G)
            except:
                logger.warning("Community detection failed")
                return 0.0

        try:
            modularity = nx.algorithms.community.modularity(G, communities)
            return modularity
        except:
            return 0.0

    def compute_assortativity(self, G: nx.Graph) -> float:
        """
        Compute degree assortativity coefficient

        r ∈ [-1, 1]: Positive = similar-degree nodes connect

        Args:
            G: NetworkX graph

        Returns:
            Assortativity coefficient
        """
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
            return assortativity
        except:
            return 0.0

    def count_bridges(
        self,
        G: nx.Graph,
        communities: Optional[List[set]] = None
    ) -> int:
        """
        Count bridge edges between communities

        Args:
            G: NetworkX graph
            communities: Community partition

        Returns:
            Number of bridge edges
        """
        if communities is None:
            try:
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(G)
            except:
                return 0

        # Create community membership dict
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i

        # Count inter-community edges
        bridge_count = 0
        for u, v in G.edges():
            if node_to_comm.get(u) != node_to_comm.get(v):
                bridge_count += 1

        return bridge_count

    def compute_k_core_distribution(
        self,
        G: nx.Graph
    ) -> Dict[int, int]:
        """
        Compute k-core decomposition distribution

        k-core: Maximal subgraph where all nodes have degree ≥ k

        Args:
            G: NetworkX graph

        Returns:
            Dict of {k: count} - number of nodes in k-core
        """
        try:
            core_numbers = nx.core_number(G)

            # Count distribution
            k_distribution = defaultdict(int)
            for node, k in core_numbers.items():
                k_distribution[k] += 1

            return dict(k_distribution)

        except:
            return {0: G.number_of_nodes()}

    def compute_giant_component_fraction(self, G: nx.Graph) -> float:
        """
        Compute fraction of nodes in giant component

        Args:
            G: NetworkX graph

        Returns:
            Fraction in [0, 1]
        """
        if G.number_of_nodes() == 0:
            return 0.0

        if G.is_directed():
            components = list(nx.weakly_connected_components(G))
        else:
            components = list(nx.connected_components(G))

        if not components:
            return 0.0

        giant_size = len(max(components, key=len))
        fraction = giant_size / G.number_of_nodes()

        return fraction

    def compute_cascade_potential(
        self,
        G: nx.Graph,
        beta: float = 0.1,
        gamma: float = 0.05
    ) -> float:
        """
        Estimate information cascade potential

        Using SIS model: spreading rate β / recovery rate γ
        Cascade occurs if β/γ > 1/λ_max (largest eigenvalue of adjacency matrix)

        Args:
            G: NetworkX graph
            beta: Infection rate
            gamma: Recovery rate

        Returns:
            Cascade potential (β/γ) / (1/λ_max)
        """
        try:
            # Get largest eigenvalue of adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)
            eigenvalues = np.linalg.eigvalsh(adj_matrix.todense())
            lambda_max = np.max(np.abs(eigenvalues))

            if lambda_max == 0:
                return 0.0

            # Critical threshold
            threshold = 1.0 / lambda_max

            # Cascade potential
            spreading_rate = beta / gamma
            potential = spreading_rate / threshold

            return potential

        except:
            return 0.0

    def extract_features(self, G: nx.Graph) -> Dict[str, float]:
        """
        Extract all social surface features

        Args:
            G: NetworkX graph

        Returns:
            Feature dictionary
        """
        logger.info("Extracting social graph features...")

        # Detect communities once
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G)
        except:
            communities = None

        features = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'modularity': self.compute_modularity(G, communities),
            'assortativity': self.compute_assortativity(G),
            'bridge_count': self.count_bridges(G, communities),
            'giant_component_fraction': self.compute_giant_component_fraction(G),
            'cascade_potential': self.compute_cascade_potential(G)
        }

        # k-core distribution stats
        k_dist = self.compute_k_core_distribution(G)
        if k_dist:
            features['max_k_core'] = max(k_dist.keys())
            features['avg_k_core'] = np.average(
                list(k_dist.keys()),
                weights=list(k_dist.values())
            )
        else:
            features['max_k_core'] = 0
            features['avg_k_core'] = 0

        logger.info(f"Extracted {len(features)} features")

        return features

    def extract_time_series(
        self,
        graph_snapshots: List[Tuple[pd.Timestamp, nx.Graph]]
    ) -> pd.DataFrame:
        """
        Extract features for multiple graph snapshots

        Args:
            graph_snapshots: List of (timestamp, graph) tuples

        Returns:
            DataFrame with features over time
        """
        features_list = []

        for timestamp, G in graph_snapshots:
            features = self.extract_features(G)
            features['timestamp'] = timestamp
            features_list.append(features)

        df = pd.DataFrame(features_list)
        df.set_index('timestamp', inplace=True)

        logger.info(f"Extracted features for {len(df)} time snapshots")

        return df
