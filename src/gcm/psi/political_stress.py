"""
Political Stress Index (PSI)
Based on Turchin's structural-demographic theory

Maps PSI components to GCM surfaces:
- Mass Mobilization Potential: financial + narrative
- Elite Competition: financial + social
- State Fiscal Distress: financial + narrative
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class PoliticalStressIndex:
    """
    Compute Political Stress Index from multi-surface features

    Per Turchin (2010, 2020), PSI tracks structural-demographic pressures:
    1. Mass Mobilization Potential (MMP)
    2. Elite Competition (EC)
    3. State Fiscal Distress (SFD)

    High PSI predicts increased instability risk
    """

    def __init__(self):
        """Initialize PSI computer"""
        logger.info("Initialized Political Stress Index computer")

    def _safe_normalize(self, series: pd.Series) -> pd.Series:
        """
        Safely normalize a series to [0, 1]

        Args:
            series: Input series

        Returns:
            Normalized series
        """
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val

        if pd.isna(min_val) or pd.isna(max_val) or range_val < 1e-8:
            # Constant or all NaN - return zeros with same index
            return pd.Series(0.0, index=series.index)

        return (series - min_val) / (range_val + 1e-8)

    def compute_mass_mobilization_potential(
        self,
        financial_features: pd.DataFrame,
        narrative_features: pd.DataFrame
    ) -> pd.Series:
        """
        Compute Mass Mobilization Potential

        Proxies:
        - Financial: market stress (tail mass, volatility)
        - Narrative: grievance narrative prevalence (cluster analysis)

        Args:
            financial_features: Financial surface features
            narrative_features: Narrative surface features

        Returns:
            MMP time series
        """
        # Get common index
        common_idx = financial_features.index.intersection(narrative_features.index)

        # Financial component: economic stress
        # High tail mass = market anticipating shocks
        if 'tail_mass_total' in financial_features.columns:
            financial_stress = financial_features.loc[common_idx, 'tail_mass_total']
        else:
            financial_stress = pd.Series(0.0, index=common_idx)

        # Normalize to [0, 1]
        financial_stress_norm = self._safe_normalize(financial_stress)

        # Narrative component: grievance prevalence
        # High topic entropy = diverse grievances
        # High drift = rapid narrative shift
        if 'topic_entropy' in narrative_features.columns:
            narrative_stress = narrative_features.loc[common_idx, 'topic_entropy']
        else:
            narrative_stress = pd.Series(0.0, index=common_idx)

        if 'avg_cluster_drift' in narrative_features.columns:
            narrative_drift = narrative_features.loc[common_idx, 'avg_cluster_drift']
        else:
            narrative_drift = pd.Series(0.0, index=common_idx)

        narrative_stress_norm = self._safe_normalize(narrative_stress)
        narrative_drift_norm = self._safe_normalize(narrative_drift)

        # Combine (weighted average)
        mmp = 0.5 * financial_stress_norm + 0.3 * narrative_stress_norm + 0.2 * narrative_drift_norm

        logger.info("Computed Mass Mobilization Potential")

        return mmp

    def compute_elite_competition(
        self,
        financial_features: pd.DataFrame,
        social_features: pd.DataFrame
    ) -> pd.Series:
        """
        Compute Elite Competition index

        Proxies:
        - Financial: market regime instability
        - Social: k-core elite density, modularity fragmentation

        Args:
            financial_features: Financial surface features
            social_features: Social surface features

        Returns:
            EC time series
        """
        # Get common index
        common_idx = financial_features.index.intersection(social_features.index)

        # Financial: volatility / skew as elite uncertainty
        if 'kurtosis' in financial_features.columns:
            financial_elite = financial_features.loc[common_idx, 'kurtosis']
        else:
            financial_elite = pd.Series(0.0, index=common_idx)
        financial_elite_norm = self._safe_normalize(financial_elite)

        # Social: high k-core indicates concentrated elite networks
        # High modularity indicates elite fragmentation
        if 'max_k_core' in social_features.columns:
            elite_density = social_features.loc[common_idx, 'max_k_core']
        else:
            elite_density = pd.Series(0.0, index=common_idx)

        if 'modularity' in social_features.columns:
            fragmentation = social_features.loc[common_idx, 'modularity']
        else:
            fragmentation = pd.Series(0.0, index=common_idx)

        elite_density_norm = self._safe_normalize(elite_density)
        fragmentation_norm = self._safe_normalize(fragmentation)

        # High competition = high density + high fragmentation
        ec = 0.4 * financial_elite_norm + 0.3 * elite_density_norm + 0.3 * fragmentation_norm

        logger.info("Computed Elite Competition")

        return ec

    def compute_state_fiscal_distress(
        self,
        financial_features: pd.DataFrame,
        narrative_features: pd.DataFrame
    ) -> pd.Series:
        """
        Compute State Fiscal Distress

        Proxies:
        - Financial: market stress, term structure
        - Narrative: legitimacy crisis narratives

        Args:
            financial_features: Financial surface features
            narrative_features: Narrative surface features

        Returns:
            SFD time series
        """
        # Get common index
        common_idx = financial_features.index.intersection(narrative_features.index)

        # Financial: skew (risk premium), term structure slope
        if 'skew_proxy' in financial_features.columns:
            skew = financial_features.loc[common_idx, 'skew_proxy']
        else:
            skew = pd.Series(0.0, index=common_idx)

        if 'term_structure_slope' in financial_features.columns:
            term_slope = financial_features.loc[common_idx, 'term_structure_slope']
        else:
            term_slope = pd.Series(0.0, index=common_idx)

        skew_norm = self._safe_normalize(skew)
        term_norm = self._safe_normalize(term_slope)

        # Narrative: coherence loss indicates legitimacy crisis
        if 'avg_cluster_coherence' in narrative_features.columns:
            coherence_loss = narrative_features.loc[common_idx, 'avg_cluster_coherence']
        else:
            coherence_loss = pd.Series(1.0, index=common_idx)
        coherence_loss_norm = 1 - self._safe_normalize(coherence_loss)

        # Combine
        sfd = 0.4 * skew_norm + 0.3 * term_norm + 0.3 * coherence_loss_norm

        logger.info("Computed State Fiscal Distress")

        return sfd

    def compute_psi(
        self,
        financial_features: pd.DataFrame,
        narrative_features: pd.DataFrame,
        social_features: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute composite Political Stress Index

        Args:
            financial_features: Financial surface features
            narrative_features: Narrative surface features
            social_features: Social surface features
            weights: Component weights (default: equal)

        Returns:
            DataFrame with PSI components and composite
        """
        if weights is None:
            weights = {'mmp': 1/3, 'ec': 1/3, 'sfd': 1/3}

        logger.info("Computing Political Stress Index...")

        # Align indices
        common_index = financial_features.index.intersection(narrative_features.index).intersection(social_features.index)

        if len(common_index) == 0:
            logger.warning("No common timestamps across surfaces")
            return pd.DataFrame()

        financial_aligned = financial_features.loc[common_index]
        narrative_aligned = narrative_features.loc[common_index]
        social_aligned = social_features.loc[common_index]

        # Compute components
        mmp = self.compute_mass_mobilization_potential(financial_aligned, narrative_aligned)
        ec = self.compute_elite_competition(financial_aligned, social_aligned)
        sfd = self.compute_state_fiscal_distress(financial_aligned, narrative_aligned)

        # Composite PSI
        psi_composite = weights['mmp'] * mmp + weights['ec'] * ec + weights['sfd'] * sfd

        # Assemble output
        psi_df = pd.DataFrame({
            'mmp': mmp,
            'ec': ec,
            'sfd': sfd,
            'psi': psi_composite
        }, index=common_index)

        logger.info(f"PSI computed for {len(psi_df)} timepoints")

        return psi_df

    def detect_instability_risk(
        self,
        psi_df: pd.DataFrame,
        threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        Detect high instability risk periods

        Args:
            psi_df: PSI DataFrame
            threshold: PSI threshold for high risk

        Returns:
            DataFrame with risk periods
        """
        high_risk = psi_df[psi_df['psi'] > threshold].copy()
        high_risk['risk_level'] = 'HIGH'

        logger.info(f"Detected {len(high_risk)} high-risk periods")

        return high_risk
