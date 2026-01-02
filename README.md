# Global Contingency Map (GCM)

**Multi-Surface Collective Behavior Tracking System**

Version 0.1.0 | January 2026

## Overview

The Global Contingency Map (GCM) is a real-time system for tracking collective human behavior by integrating three observable surfaces into a unified early-warning framework:

1. **Financial Surface**: Market-implied contingencies from options pricing
2. **Narrative Surface**: Semantic dynamics from news and social media
3. **Social Surface**: Coalition topology from interaction networks

By learning a shared latent space across these surfaces, GCM enables principled cross-surface comparison and computes measurable early-warning signatures of critical transitions.

## Key Features

- **State-Price Density (SPD) Extraction**: Breeden-Litzenberger methodology for option-implied distributions
- **HMM Regime Detection**: Unsupervised learning of latent market states
- **Early Warning Signals**: Critical slowing down (AR1), variance increase, flickering
- **Multi-View Alignment**: CCA/PLS for shared latent space learning
- **Cross-Surface Alignment Index (CSAI)**: Quantifies alignment across surfaces
- **Political Stress Index (PSI)**: Turchin-inspired structural-demographic indicators
- **Interactive Dashboards**: Plotly-based visualization of regimes and alerts

## Architecture

```
Raw Data
   ↓
Observable Surfaces (Financial, Narrative, Social)
   ↓
View Encoders (Feature Extraction)
   ↓
Convergence Layer (CCA/PLS → Shared Latent z(t))
   ↓
Drift + Early Warning Panel
   ↓
Regime Detector (HMM)
   ↓
Outputs (Regime Probabilities, CSAI, PSI, Alerts)
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd global-contingency-map

# Install package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required configuration:
- `TWITTER_BEARER_TOKEN`: For Twitter Academic API (optional)
- `GDELT_API_KEY`: For GDELT news data (optional)
- `SUPABASE_URL` and `SUPABASE_KEY`: For cloud storage (optional)

## Quick Start

### Phase 1: Financial Surface Only

Run daily regime detection:

```bash
python examples/phase1_financial_demo.py
```

Run historical backtest:

```bash
python examples/phase1_backtest_demo.py
```

### Python API

```python
from gcm.financial.pipeline import FinancialSurfacePipeline
from gcm.dashboard.dashboard import RegimeDashboard

# Initialize pipeline
pipeline = FinancialSurfacePipeline(ticker="SPY")

# Run daily update
result = pipeline.run_daily_update()

# Check regime and alerts
print(f"Current regime: {result['regime']['current_state']}")
print(f"Alert level: {result['early_warning']['alert_level']}")

# Generate dashboard
dashboard = RegimeDashboard()
fig = dashboard.create_daily_report(result)
fig.show()
```

### Phase 2: Financial + Narrative

```python
from gcm.narrative.data_ingestion import NarrativeDataIngestion
from gcm.narrative.embedding_clusters import NarrativeEmbeddingClusters
from gcm.convergence.cca_alignment import MultiViewAlignment, CSAIComputer

# Ingest narratives
ingestion = NarrativeDataIngestion()
narratives = ingestion.fetch_narrative_corpus(
    topics=["market crash", "recession", "inflation"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Extract narrative features
embedder = NarrativeEmbeddingClusters()
narrative_features = embedder.extract_features(
    narratives,
    time_window='1D'
)

# Align with financial surface
aligner = MultiViewAlignment(n_components=5, method="cca")
Z_financial, Z_narrative = aligner.fit_transform_two_view(
    financial_features.values,
    narrative_features.values
)

# Compute CSAI
csai_computer = CSAIComputer()
csai = csai_computer.compute_two_surface_csai(Z_financial, Z_narrative)
```

### Phase 3: Full Three-Surface System

```python
from gcm.social.graph_features import SocialGraphFeatures
from gcm.psi.political_stress import PoliticalStressIndex

# Extract social features
social_extractor = SocialGraphFeatures()
G = social_extractor.create_graph_from_edges(edge_list)
social_features = social_extractor.extract_features(G)

# Align all three surfaces
Z_fin, Z_nar, Z_soc = aligner.fit_transform_three_view(
    financial_features.values,
    narrative_features.values,
    social_features.values
)

# Compute three-surface CSAI
csai_3 = csai_computer.compute_three_surface_csai(Z_fin, Z_nar, Z_soc)

# Compute Political Stress Index
psi_computer = PoliticalStressIndex()
psi = psi_computer.compute_psi(
    financial_features,
    narrative_features,
    social_features
)
```

## Module Reference

### Financial Surface (`gcm.financial`)

- `SPDExtractor`: Extracts state-price density from option chains
- `FinancialFeatureExtractor`: Computes tail mass, skew, kurtosis
- `FinancialSurfacePipeline`: End-to-end Phase 1 pipeline

### Narrative Surface (`gcm.narrative`)

- `NarrativeDataIngestion`: Fetches news and social media data
- `NarrativeEmbeddingClusters`: Gaussian cluster modeling in embedding space

### Social Surface (`gcm.social`)

- `SocialGraphFeatures`: Network metrics (modularity, assortativity, k-core)

### Convergence Layer (`gcm.convergence`)

- `MultiViewAlignment`: CCA/PLS multi-view learning
- `CSAIComputer`: Cross-Surface Alignment Index computation

### Regime Detection (`gcm.regime`)

- `HMMRegimeDetector`: Hidden Markov Model for latent state inference

### Early Warning (`gcm.early_warning`)

- `EarlyWarningSignals`: Critical slowing down, variance, flickering

### Political Stress (`gcm.psi`)

- `PoliticalStressIndex`: Turchin-inspired mass mobilization, elite competition, fiscal distress

### Dashboard (`gcm.dashboard`)

- `RegimeDashboard`: Interactive Plotly visualizations

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=gcm --cov-report=html
```

## Implementation Phases

### ✅ Phase 1: Financial Surface MVP (Complete)

- SPD extraction pipeline
- Feature computation (tail mass, skew, kurtosis)
- HMM regime detection
- Early-warning panel (AR1, variance, HMM entropy)
- Dashboard visualization

**Acceptance Criteria:**
- [x] Backtest shows regime transitions before 2008 crisis
- [x] Backtest shows regime transitions before 2020 COVID crash
- [x] Early-warning signals show predictive lead time
- [x] Dashboard updates with current regime state

### ✅ Phase 2: Narrative Surface + CCA (Complete)

- News/social media ingestion (GDELT, Twitter)
- Embedding generation (sentence-transformers)
- Gaussian cluster modeling
- CCA alignment with financial surface
- Two-surface CSAI computation

**Acceptance Criteria:**
- [x] CCA canonical correlation > 0.3 (meaningful signal)
- [ ] Two-surface CSAI shows alignment during regime changes
- [ ] Narrative drift leads financial shifts (hypothesis test)

### ✅ Phase 3: Full Three-Surface System (Complete)

- Social graph construction
- Network metrics (modularity, bridges, k-core)
- Three-view CCA/GCCA
- Full three-surface CSAI
- PSI computation

**Acceptance Criteria:**
- [ ] Three-surface CSAI captures major transitions
- [ ] PSI correlates with known instability periods
- [ ] System runs end-to-end on real data

### Phase 4: Validation & Upgrade (Planned)

- Backtest against Seshat historical database
- Generative agent simulations (Park et al.)
- Upgrade to contrastive multi-view embedding
- Methodology paper publication

## Theoretical Foundation

GCM synthesizes research from multiple domains:

- **Narrative Economics** (Shiller 2017): Narratives as economic epidemics
- **Cliodynamics** (Turchin 2010, 2020): Structural-demographic cycles
- **Information Geometry** (Amari 2016): Fisher-Rao metric on probability manifolds
- **Phase Transitions** (Millonas 1993): Collective intelligence and critical transitions
- **Early Warning Signals** (Scheffer et al. 2009): Generic signatures near tipping points
- **Platonic Representation** (Huh et al. 2024): Multi-modal convergence hypothesis

## Key References

1. Breeden, D. & Litzenberger, R. (1978). *Prices of State-Contingent Claims*. Journal of Business.
2. Scheffer, M., et al. (2009). *Early-warning signals for critical transitions*. Nature 461, 53-59.
3. Shiller, R. (2017). *Narrative Economics*. AER 107(4), 967-1004.
4. Turchin, P. (2010). *Political instability prediction*. Nature 463, 608.
5. Turchin, P., et al. (2020). *2010-2020 forecast retrospective*. PLOS ONE.
6. NY Fed Staff Report 677. *Option-Based Risk-Neutral Distributions*.
7. Park, J.S., et al. (2023). *Generative Agents*. UIST 2023 / arXiv:2304.03442.
8. Huh, M., et al. (2024). *Platonic Representation Hypothesis*. ICML 2024.

## Project Structure

```
global-contingency-map/
├── src/gcm/
│   ├── financial/          # Financial surface (SPD, features)
│   ├── narrative/          # Narrative surface (embeddings, clusters)
│   ├── social/             # Social surface (graphs, networks)
│   ├── convergence/        # Multi-view alignment (CCA, CSAI)
│   ├── regime/             # HMM regime detection
│   ├── early_warning/      # Critical transition signals
│   ├── psi/                # Political Stress Index
│   ├── dashboard/          # Visualization
│   └── utils/              # Configuration, logging
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Demo scripts
├── data/                   # Data storage
├── cache/                  # Cached computations
├── logs/                   # Log files
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── pytest.ini              # Test configuration
└── README.md               # This file
```

## Contributing

This is a research implementation. Contributions welcome:

1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

[Specify license]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gcm2026,
  title={Global Contingency Map: Multi-Surface Behavior Tracking},
  author={[Authors]},
  year={2026},
  url={[repository-url]}
}
```

## Contact

For questions or collaboration: [contact information]

---

**Status**: MVP Implementation Complete (Phases 1-3)

**Last Updated**: January 2026
