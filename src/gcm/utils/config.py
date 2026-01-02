"""Configuration management for GCM"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Global configuration"""

    # Directories
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "cache"))
    LOG_DIR = BASE_DIR / "logs"

    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # API Keys
    TWITTER_BEARER_TOKEN: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN")
    GDELT_API_KEY: Optional[str] = os.getenv("GDELT_API_KEY")
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Financial surface config
    DEFAULT_TICKER: str = "SPY"
    OPTION_EXPIRY_DAYS_MIN: int = 7
    OPTION_EXPIRY_DAYS_MAX: int = 60

    # Regime detection config
    N_REGIMES: int = 3
    HMM_N_ITER: int = 100

    # Early warning config
    ROLLING_WINDOW: int = 20  # days
    AR1_YELLOW_THRESHOLD: float = 0.7
    AR1_ORANGE_THRESHOLD: float = 0.8
    VARIANCE_MULTIPLIER_THRESHOLD: float = 2.0
    CSAI_RED_THRESHOLD: float = 0.7

    # Convergence layer config
    LATENT_DIM: int = 10  # Shared latent space dimension
    CCA_N_COMPONENTS: int = 5

    # Narrative config
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    N_CLUSTERS: int = 10

    # Social network config
    MIN_EDGE_WEIGHT: float = 0.1


config = Config()
