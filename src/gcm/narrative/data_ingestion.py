"""
Narrative data ingestion pipeline
Fetches news and social media data from GDELT and other sources
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
from pathlib import Path

from ..utils.logging_config import setup_logger
from ..utils.config import config

logger = setup_logger(__name__)


class NarrativeDataIngestion:
    """
    Ingest narrative data from multiple sources

    Sources:
    - GDELT (Global Database of Events, Language, and Tone)
    - Twitter Academic API (requires bearer token)
    - News APIs
    """

    def __init__(self):
        """Initialize data ingestion"""
        self.gdelt_base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        logger.info("Initialized narrative data ingestion")

    def fetch_gdelt_articles(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_records: int = 250,
        mode: str = "artlist"
    ) -> pd.DataFrame:
        """
        Fetch articles from GDELT

        Args:
            query: Search query
            start_date: Start date (default: 7 days ago)
            end_date: End date (default: today)
            max_records: Maximum number of records
            mode: GDELT mode (artlist, timeline, etc.)

        Returns:
            DataFrame with articles
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()

        # Format dates for GDELT
        start_str = start_date.strftime("%Y%m%d%H%M%S")
        end_str = end_date.strftime("%Y%m%d%H%M%S")

        params = {
            "query": query,
            "mode": mode,
            "maxrecords": max_records,
            "startdatetime": start_str,
            "enddatetime": end_str,
            "format": "json"
        }

        logger.info(f"Fetching GDELT articles: {query} from {start_date} to {end_date}")

        try:
            response = requests.get(self.gdelt_base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'articles' not in data:
                logger.warning("No articles found in GDELT response")
                return pd.DataFrame()

            articles = data['articles']

            # Convert to DataFrame
            df = pd.DataFrame(articles)

            # Parse date
            if 'seendate' in df.columns:
                df['timestamp'] = pd.to_datetime(df['seendate'], format='%Y%m%dT%H%M%SZ')

            logger.info(f"Fetched {len(df)} articles from GDELT")

            return df

        except Exception as e:
            logger.error(f"GDELT fetch failed: {e}")
            return pd.DataFrame()

    def fetch_twitter_data(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Fetch tweets using Twitter Academic API

        Requires TWITTER_BEARER_TOKEN in environment

        Args:
            query: Search query
            start_date: Start date
            end_date: End date
            max_results: Maximum results

        Returns:
            DataFrame with tweets
        """
        if not config.TWITTER_BEARER_TOKEN:
            logger.warning("Twitter bearer token not configured, skipping Twitter data")
            return pd.DataFrame()

        # Twitter API v2 endpoint
        url = "https://api.twitter.com/2/tweets/search/recent"

        headers = {
            "Authorization": f"Bearer {config.TWITTER_BEARER_TOKEN}"
        }

        params = {
            "query": query,
            "max_results": min(max_results, 100),  # API limit
            "tweet.fields": "created_at,public_metrics,entities"
        }

        if start_date:
            params["start_time"] = start_date.isoformat() + "Z"
        if end_date:
            params["end_time"] = end_date.isoformat() + "Z"

        logger.info(f"Fetching Twitter data: {query}")

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data:
                logger.warning("No tweets found")
                return pd.DataFrame()

            tweets = data['data']
            df = pd.DataFrame(tweets)

            # Parse timestamps
            if 'created_at' in df.columns:
                df['timestamp'] = pd.to_datetime(df['created_at'])

            logger.info(f"Fetched {len(df)} tweets")

            return df

        except Exception as e:
            logger.error(f"Twitter fetch failed: {e}")
            return pd.DataFrame()

    def fetch_narrative_corpus(
        self,
        topics: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch comprehensive narrative corpus from multiple sources

        Args:
            topics: List of topic queries (e.g., ["market crash", "recession", "inflation"])
            start_date: Start date
            end_date: End date

        Returns:
            Combined DataFrame with all narratives
        """
        all_narratives = []

        for topic in topics:
            # Fetch from GDELT
            gdelt_df = self.fetch_gdelt_articles(
                query=topic,
                start_date=start_date,
                end_date=end_date
            )

            if not gdelt_df.empty:
                gdelt_df['source'] = 'gdelt'
                gdelt_df['topic_query'] = topic
                all_narratives.append(gdelt_df)

            # Fetch from Twitter (if configured)
            twitter_df = self.fetch_twitter_data(
                query=topic,
                start_date=start_date,
                end_date=end_date
            )

            if not twitter_df.empty:
                twitter_df['source'] = 'twitter'
                twitter_df['topic_query'] = topic
                all_narratives.append(twitter_df)

        if not all_narratives:
            logger.warning("No narratives fetched from any source")
            return pd.DataFrame()

        # Combine all sources
        combined_df = pd.concat(all_narratives, ignore_index=True)

        # Standardize text column
        if 'title' in combined_df.columns:
            combined_df['text'] = combined_df['title']
        elif 'text' not in combined_df.columns:
            combined_df['text'] = ''

        logger.info(f"Total narratives collected: {len(combined_df)}")

        return combined_df

    def create_mock_narrative_data(
        self,
        n_samples: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Create mock narrative data for testing

        Args:
            n_samples: Number of samples
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with mock narratives
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Generate random timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, periods=n_samples)

        # Mock topics
        topics = [
            "market volatility",
            "economic uncertainty",
            "policy change",
            "inflation concerns",
            "growth outlook"
        ]

        # Mock texts
        texts = [
            f"Discussion about {np.random.choice(topics)} on {ts.strftime('%Y-%m-%d')}"
            for ts in timestamps
        ]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'text': texts,
            'source': 'mock',
            'topic_query': np.random.choice(topics, n_samples)
        })

        logger.info(f"Created {n_samples} mock narratives")

        return df

    def save_corpus(self, df: pd.DataFrame, filename: str):
        """Save narrative corpus to disk"""
        path = config.DATA_DIR / filename
        df.to_parquet(path)
        logger.info(f"Saved corpus to {path}")

    def load_corpus(self, filename: str) -> pd.DataFrame:
        """Load narrative corpus from disk"""
        path = config.DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} narratives from {path}")
        return df
