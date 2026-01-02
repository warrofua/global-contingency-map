#!/usr/bin/env python3
"""
Full Three-Surface System Demo

Demonstrates complete GCM pipeline:
1. Financial surface analysis
2. Narrative surface analysis
3. Social surface analysis
4. Multi-view CCA alignment
5. CSAI computation
6. PSI computation
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gcm.financial.features import FinancialFeatureExtractor
from gcm.narrative.data_ingestion import NarrativeDataIngestion
from gcm.narrative.embedding_clusters import NarrativeEmbeddingClusters
from gcm.social.graph_features import SocialGraphFeatures
from gcm.convergence.cca_alignment import MultiViewAlignment, CSAIComputer
from gcm.psi.political_stress import PoliticalStressIndex
from gcm.utils.config import config


def main():
    print("=" * 70)
    print("GCM Full System Demo: Three-Surface Integration")
    print("=" * 70)

    # ========== SURFACE 1: FINANCIAL ==========
    print("\n[1/5] Extracting Financial Surface Features...")
    print("-" * 70)

    financial_extractor = FinancialFeatureExtractor(ticker="SPY")

    try:
        # Try to get real features (may fail outside market hours)
        fin_features = financial_extractor.extract_features()
        print(f"✓ Extracted real-time features for SPY")
    except Exception as e:
        print(f"⚠ Real-time extraction failed: {e}")
        print("  Using mock data...")
        fin_features = financial_extractor._get_nan_features(datetime.now())

    # Create mock time series for demo
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    financial_df = pd.DataFrame({
        'tail_mass_total': np.random.uniform(0.1, 0.3, len(dates)),
        'skew_proxy': np.random.uniform(-0.1, 0.1, len(dates)),
        'kurtosis': np.random.uniform(0, 2, len(dates))
    }, index=dates)

    print(f"  Financial features: {list(financial_df.columns)}")
    print(f"  Time range: {financial_df.index[0]} to {financial_df.index[-1]}")

    # ========== SURFACE 2: NARRATIVE ==========
    print("\n[2/5] Extracting Narrative Surface Features...")
    print("-" * 70)

    # Use mock data for demo
    narrative_ingestion = NarrativeDataIngestion()
    narratives = narrative_ingestion.create_mock_narrative_data(
        n_samples=200,
        start_date=dates[0],
        end_date=dates[-1]
    )

    print(f"  Collected {len(narratives)} narratives")

    # Embed and cluster
    narrative_embedder = NarrativeEmbeddingClusters(n_clusters=5)
    narrative_df = narrative_embedder.extract_features(
        narratives,
        time_window='1D'
    )

    print(f"  Narrative features: {list(narrative_df.columns)}")
    print(f"  Time range: {narrative_df.index[0]} to {narrative_df.index[-1]}")

    # ========== SURFACE 3: SOCIAL ==========
    print("\n[3/5] Extracting Social Surface Features...")
    print("-" * 70)

    social_extractor = SocialGraphFeatures()

    # Create mock graph for each day with time-varying parameters
    social_snapshots = []
    for i, date in enumerate(dates):
        # Vary network parameters over time to simulate evolution
        p_within = 0.25 + 0.1 * np.sin(i / 5)  # Oscillate between 0.15 and 0.35
        p_between = 0.03 + 0.04 * (i / len(dates))  # Gradual increase 0.03 to 0.07

        G = social_extractor.create_mock_graph(
            n_nodes=100,
            n_communities=5,
            p_within=p_within,
            p_between=p_between,
            seed=1000 + i  # Different seed for each snapshot
        )
        social_snapshots.append((date, G))

    social_df = social_extractor.extract_time_series(social_snapshots)

    print(f"  Social features: {list(social_df.columns)}")
    print(f"  Time range: {social_df.index[0]} to {social_df.index[-1]}")

    # ========== CONVERGENCE: MULTI-VIEW ALIGNMENT ==========
    print("\n[4/5] Learning Shared Latent Space (CCA)...")
    print("-" * 70)

    # Align indices
    common_idx = financial_df.index.intersection(narrative_df.index).intersection(social_df.index)
    print(f"  Common timestamps: {len(common_idx)}")

    if len(common_idx) < 10:
        print("  ⚠ Insufficient overlapping data for alignment")
        return

    fin_aligned = financial_df.loc[common_idx]
    nar_aligned = narrative_df.loc[common_idx]
    soc_aligned = social_df.loc[common_idx]

    # Three-view alignment
    aligner = MultiViewAlignment(n_components=3, method="cca")

    Z_financial, Z_narrative, Z_social = aligner.fit_transform_three_view(
        fin_aligned.values,
        nar_aligned.values,
        soc_aligned.values,
        view_names=("financial", "narrative", "social")
    )

    print(f"  ✓ Learned shared latent space (dim={Z_financial.shape[1]})")
    print(f"  Financial latent: {Z_financial.shape}")
    print(f"  Narrative latent: {Z_narrative.shape}")
    print(f"  Social latent: {Z_social.shape}")

    # Compute CSAI
    csai_computer = CSAIComputer()
    csai_3surface = csai_computer.compute_three_surface_csai(
        Z_financial,
        Z_narrative,
        Z_social,
        window=3
    )

    print(f"\n  CSAI Statistics:")
    print(f"    Mean: {np.nanmean(csai_3surface):.3f}")
    print(f"    Max: {np.nanmax(csai_3surface):.3f}")
    print(f"    Min: {np.nanmin(csai_3surface):.3f}")

    # ========== POLITICAL STRESS INDEX ==========
    print("\n[5/5] Computing Political Stress Index (PSI)...")
    print("-" * 70)

    psi_computer = PoliticalStressIndex()
    psi_df = psi_computer.compute_psi(
        fin_aligned,
        nar_aligned,
        soc_aligned
    )

    print(f"  ✓ PSI computed for {len(psi_df)} timepoints")
    print(f"\n  PSI Components:")
    print(f"    Mass Mobilization Potential: {psi_df['mmp'].mean():.3f}")
    print(f"    Elite Competition: {psi_df['ec'].mean():.3f}")
    print(f"    State Fiscal Distress: {psi_df['sfd'].mean():.3f}")
    print(f"    Composite PSI: {psi_df['psi'].mean():.3f}")

    # Detect high-risk periods
    high_risk = psi_computer.detect_instability_risk(psi_df, threshold=0.7)
    print(f"\n  High-risk periods: {len(high_risk)}")

    if len(high_risk) > 0:
        print("\n  Sample high-risk dates:")
        print(high_risk.head())

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nSystem Status:")
    print(f"  ✓ Financial surface: {len(fin_aligned)} features")
    print(f"  ✓ Narrative surface: {len(nar_aligned)} features")
    print(f"  ✓ Social surface: {len(soc_aligned)} features")
    print(f"  ✓ Shared latent space: {Z_financial.shape[1]} dimensions")
    print(f"  ✓ CSAI: {np.sum(~np.isnan(csai_3surface))} valid values")
    print(f"  ✓ PSI: {len(psi_df)} timepoints")

    print("\nNext Steps:")
    print("  1. Run with real market data during trading hours")
    print("  2. Configure GDELT/Twitter APIs for narrative data")
    print("  3. Provide social network data for coalition analysis")
    print("  4. Backtest on historical crisis periods (2008, 2020)")
    print("  5. Deploy continuous monitoring system")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
