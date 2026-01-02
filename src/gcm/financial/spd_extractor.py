"""
State-Price Density (SPD) / Risk-Neutral Density (RND) extraction from options
Based on Breeden-Litzenberger methodology and NY Fed Staff Report 677
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.stats import norm

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class SPDExtractor:
    """
    Extract State-Price Density from option chains

    Implements Breeden-Litzenberger method:
    SPD(K) = e^(rT) * ∂²C/∂K² where C is call price
    """

    def __init__(self, ticker: str = "SPY"):
        """
        Initialize SPD extractor

        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        logger.info(f"Initialized SPD extractor for {ticker}")

    def fetch_option_chain(
        self,
        expiry_date: Optional[str] = None,
        min_days: int = 7,
        max_days: int = 60
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float, datetime]:
        """
        Fetch option chain data

        Args:
            expiry_date: Specific expiry date (YYYY-MM-DD) or None for nearest
            min_days: Minimum days to expiry
            max_days: Maximum days to expiry

        Returns:
            (calls_df, puts_df, spot_price, expiry_datetime)
        """
        # Get current stock price
        spot_price = self.stock.history(period="1d")['Close'].iloc[-1]

        # Get available expiry dates
        expiry_dates = self.stock.options

        if not expiry_dates:
            raise ValueError(f"No options data available for {self.ticker}")

        # Select appropriate expiry
        if expiry_date is None:
            # Find expiry within desired range
            today = datetime.now()
            selected_expiry = None

            for exp in expiry_dates:
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                days_to_expiry = (exp_date - today).days

                if min_days <= days_to_expiry <= max_days:
                    selected_expiry = exp
                    break

            if selected_expiry is None:
                selected_expiry = expiry_dates[0]
                logger.warning(f"No expiry in range [{min_days}, {max_days}] days, using {selected_expiry}")
        else:
            selected_expiry = expiry_date

        expiry_datetime = datetime.strptime(selected_expiry, "%Y-%m-%d")
        logger.info(f"Using expiry date: {selected_expiry}")

        # Fetch option chain
        opt_chain = self.stock.option_chain(selected_expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Clean data: remove rows with zero volume or bid/ask
        calls = calls[(calls['volume'] > 0) & (calls['bid'] > 0) & (calls['ask'] > 0)]
        puts = puts[(puts['volume'] > 0) & (puts['bid'] > 0) & (puts['ask'] > 0)]

        # Calculate mid prices
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        puts['mid_price'] = (puts['bid'] + puts['ask']) / 2

        logger.info(f"Fetched {len(calls)} calls and {len(puts)} puts")

        return calls, puts, spot_price, expiry_datetime

    def extract_spd(
        self,
        calls: pd.DataFrame,
        spot_price: float,
        expiry_datetime: datetime,
        risk_free_rate: float = 0.05,
        smoothing: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract SPD using Breeden-Litzenberger formula

        Args:
            calls: DataFrame of call options
            spot_price: Current stock price
            expiry_datetime: Option expiry date
            risk_free_rate: Risk-free rate (annualized)
            smoothing: Smoothing parameter for spline (higher = smoother)

        Returns:
            (strikes, spd_values) arrays
        """
        # Time to expiry in years
        T = (expiry_datetime - datetime.now()).days / 365.25

        if T <= 0:
            raise ValueError("Expiry date must be in the future")

        # Sort by strike
        calls_sorted = calls.sort_values('strike')
        strikes = calls_sorted['strike'].values
        prices = calls_sorted['mid_price'].values

        # Filter to reasonable strike range (e.g., 70% to 130% of spot)
        mask = (strikes >= 0.7 * spot_price) & (strikes <= 1.3 * spot_price)
        strikes = strikes[mask]
        prices = prices[mask]

        if len(strikes) < 10:
            logger.warning("Insufficient strikes for reliable SPD extraction")
            return np.array([]), np.array([])

        # Interpolate call prices with cubic spline
        # Use smoothing spline to avoid overfitting noise
        spline = UnivariateSpline(strikes, prices, s=smoothing * len(strikes))

        # Create fine grid for evaluation
        strike_grid = np.linspace(strikes.min(), strikes.max(), 200)

        # Compute second derivative (SPD)
        # SPD(K) = e^(rT) * ∂²C/∂K²
        second_deriv = spline.derivative(n=2)(strike_grid)
        spd = np.exp(risk_free_rate * T) * second_deriv

        # SPD should be non-negative (theoretical property)
        # Negative values indicate noise/estimation error
        spd = np.maximum(spd, 0)

        # Normalize to integrate to 1 (probability density)
        spd_integral = np.trapz(spd, strike_grid)
        if spd_integral > 0:
            spd = spd / spd_integral

        logger.info(f"Extracted SPD over {len(strike_grid)} strikes")

        return strike_grid, spd

    def compute_spd_moments(
        self,
        strikes: np.ndarray,
        spd: np.ndarray,
        spot_price: float
    ) -> dict:
        """
        Compute moments and statistics of the SPD

        Args:
            strikes: Strike price grid
            spd: SPD values
            spot_price: Current stock price

        Returns:
            Dictionary of SPD statistics
        """
        if len(strikes) == 0 or len(spd) == 0:
            return {
                'mean': np.nan,
                'variance': np.nan,
                'std': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'tail_mass_left': np.nan,
                'tail_mass_right': np.nan
            }

        # Mean (expected value under RND)
        mean = np.trapz(strikes * spd, strikes)

        # Variance
        variance = np.trapz((strikes - mean)**2 * spd, strikes)
        std = np.sqrt(variance)

        # Skewness
        skewness = np.trapz((strikes - mean)**3 * spd, strikes) / (std**3) if std > 0 else 0

        # Excess kurtosis
        kurtosis = np.trapz((strikes - mean)**4 * spd, strikes) / (std**4) - 3 if std > 0 else 0

        # Tail mass beyond ±2σ
        left_threshold = spot_price - 2 * std
        right_threshold = spot_price + 2 * std

        tail_mass_left = np.trapz(
            spd[strikes < left_threshold],
            strikes[strikes < left_threshold]
        ) if np.any(strikes < left_threshold) else 0

        tail_mass_right = np.trapz(
            spd[strikes > right_threshold],
            strikes[strikes > right_threshold]
        ) if np.any(strikes > right_threshold) else 0

        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_mass_left': tail_mass_left,
            'tail_mass_right': tail_mass_right,
            'tail_mass_total': tail_mass_left + tail_mass_right
        }
