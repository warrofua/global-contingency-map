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
        # Financial component: economic stress
        # High tail mass = market anticipating shocks
        financial_stress = financial_features.get('tail_mass_total', pd.Series(0))

        # Normalize to [0, 1]
        financial_stress_norm = (financial_stress - financial_stress.min()) / (financial_stress.max() - financial_stress.min() + 1e-8)

        # Narrative component: grievance prevalence
        # High topic entropy = diverse grievances
        # High drift = rapid narrative shift
        narrative_stress = narrative_features.get('topic_entropy', pd.Series(0))
        narrative_drift = narrative_features.get('avg_cluster_drift', pd.Series(0))

        narrative_stress_norm = (narrative_stress - narrative_stress.min()) / (narrative_stress.max() - narrative_stress.min() + 1e-8)
        narrative_drift_norm = (narrative_drift - narrative_drift.min()) / (narrative_drift.max() - narrative_drift.min() + 1e-8)

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
        # Financial: volatility / skew as elite uncertainty
        financial_elite = financial_features.get('kurtosis', pd.Series(0))
        financial_elite_norm = (financial_elite - financial_elite.min()) / (financial_elite.max() - financial_elite.min() + 1e-8)

        # Social: high k-core indicates concentrated elite networks
        # High modularity indicates elite fragmentation
        elite_density = social_features.get('max_k_core', pd.Series(0))
        fragmentation = social_features.get('modularity', pd.Series(0))

        elite_density_norm = (elite_density - elite_density.min()) / (elite_density.max() - elite_density.min() + 1e-8)
        fragmentation_norm = (fragmentation - fragmentation.min()) / (fragmentation.max() - fragmentation.min() + 1e-8)

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
        # Financial: skew (risk premium), term structure slope
        skew = financial_features.get('skew_proxy', pd.Series(0))
        term_slope = financial_features.get('term_structure_slope', pd.Series(0))

        skew_norm = (skew - skew.min()) / (skew.max() - skew.min() + 1e-8)
        term_norm = (term_slope - term_slope.min()) / (term_slope.max() - term_slope.min() + 1e-8)

        # Narrative: coherence loss indicates legitimacy crisis
        coherence_loss = narrative_features.get('avg_cluster_coherence', pd.Series(1))
        coherence_loss_norm = 1 - (coherence_loss - coherence_loss.min()) / (coherence_loss.max() - coherence_loss.min() + 1e-8)

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
