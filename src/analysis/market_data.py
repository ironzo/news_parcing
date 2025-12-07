"""
Market data integration module for fetching and processing historical market data.

This module provides functionality to:
- Fetch OHLCV data from yfinance
- Normalize and validate market data
- Synchronize with economic news events
- Calculate price impact of news events
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from pytz import timezone, UTC

from ..database.models import EconomicNews, MarketData
from ..database.repository import DatabaseRepository


logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetcher for historical market data with validation and persistence.
    """
    
    def __init__(self, repository: DatabaseRepository):
        """
        Initialize market data fetcher.
        
        Args:
            repository: Database repository instance for persistence.
        """
        self.repository = repository
        self.supported_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VOO', 'VTI']
        logger.info("Initialized MarketDataFetcher")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        save_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from yfinance.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY', 'QQQ').
            start_date: Start date for historical data.
            end_date: End date for historical data.
            interval: Data interval ('1d', '1h', '5m', etc.). Default is '1d'.
            save_to_db: Whether to persist data to database. Default is True.
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            
        Raises:
            ValueError: If symbol is not supported or data fetch fails.
        """
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        
        try:
            # Convert datetime to string format for yfinance compatibility
            start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else str(end_date)
            
            # Use yf.download() which is more reliable than Ticker.history()
            df = yf.download(
                tickers=symbol,
                start=start_str,
                end=end_str,
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Handle MultiIndex columns from yfinance (newer versions)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns - take first level (Price type)
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Normalize column names
            df = df.reset_index()
            
            # Map various possible column names to standard names
            column_mapping = {
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'index': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'close',
                'Volume': 'volume'
            }
            
            # Only rename columns that exist
            rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Select relevant columns
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns after normalization: {missing_cols}")
            df = df[columns]
            
            # Validate and clean data
            df = self._validate_and_clean_data(df, symbol)
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            
            # Save to database if requested
            if save_to_db:
                self._save_to_database(symbol, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean market data.
        
        Args:
            df: Raw DataFrame from yfinance.
            symbol: Trading symbol.
            
        Returns:
            Cleaned DataFrame.
        """
        original_len = len(df)
        
        # Ensure timestamp is timezone-aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(UTC)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(UTC)
        
        # Remove rows with missing prices
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate price logic (high >= low, etc.)
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_rows.any():
            logger.warning(f"Removing {invalid_rows.sum()} rows with invalid price relationships")
            df = df[~invalid_rows]
        
        # Remove negative or zero prices
        invalid_prices = (
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0)
        )
        
        if invalid_prices.any():
            logger.warning(f"Removing {invalid_prices.sum()} rows with invalid prices")
            df = df[~invalid_prices]
        
        # Fill missing volume with 0
        df['volume'] = df['volume'].fillna(0)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        cleaned_len = len(df)
        if cleaned_len < original_len:
            logger.info(f"Cleaned {symbol} data: {original_len} -> {cleaned_len} records")
        
        return df
    
    def _save_to_database(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Save market data to database.
        
        Args:
            symbol: Trading symbol.
            df: DataFrame with market data.
            
        Returns:
            Number of records saved.
        """
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Save to database
            count = self.repository.save_market_data(symbol, records)
            logger.info(f"Saved {count} records for {symbol} to database")
            return count
            
        except Exception as e:
            logger.error(f"Failed to save data to database: {e}")
            raise
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d',
        save_to_db: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of trading symbols.
            start_date: Start date for historical data.
            end_date: End date for historical data.
            interval: Data interval. Default is '1d'.
            save_to_db: Whether to persist data. Default is True.
            
        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    save_to_db=save_to_db
                )
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_cached_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve cached market data from database.
        
        Args:
            symbol: Trading symbol.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
            
        Returns:
            DataFrame with cached market data.
        """
        try:
            records = self.repository.get_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not records:
                logger.warning(f"No cached data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = [r.to_dict() for r in records]
            df = pd.DataFrame(data)
            
            # Parse timestamp strings back to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Retrieved {len(df)} cached records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached data: {e}")
            return pd.DataFrame()


def sync_with_news_events(
    news_items: List[EconomicNews],
    market_data: pd.DataFrame,
    time_window_minutes: int = 30
) -> pd.DataFrame:
    """
    Synchronize news events with market data timestamps.
    
    This function creates a merged dataset that aligns economic news events
    with the corresponding market data within a specified time window.
    
    Args:
        news_items: List of EconomicNews objects from database.
        market_data: DataFrame with market data (must have 'timestamp' column).
        time_window_minutes: Time window in minutes to match news with market data.
        
    Returns:
        DataFrame with columns: news_id, event_time, timestamp, open, high, low, 
        close, volume, title, volatility, forecast, previous, actual.
    """
    logger.info(f"Synchronizing {len(news_items)} news events with market data")
    
    if market_data.empty:
        logger.warning("Market data is empty, cannot synchronize")
        return pd.DataFrame()
    
    # Ensure timestamps are timezone-aware
    if market_data['timestamp'].dt.tz is None:
        market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(UTC)
    
    synchronized_data = []
    
    for news in news_items:
        if news.event_time is None:
            continue
        
        # Ensure news event time is timezone-aware
        event_time = news.event_time
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=UTC)
        else:
            event_time = event_time.astimezone(UTC)
        
        # Find market data within time window
        time_delta = timedelta(minutes=time_window_minutes)
        start_window = event_time - time_delta
        end_window = event_time + time_delta
        
        # Filter market data within window
        mask = (
            (market_data['timestamp'] >= start_window) &
            (market_data['timestamp'] <= end_window)
        )
        matching_data = market_data[mask]
        
        if matching_data.empty:
            # If no exact match, find nearest timestamp
            time_diffs = abs(market_data['timestamp'] - event_time)
            nearest_idx = time_diffs.idxmin()
            nearest_data = market_data.loc[nearest_idx:nearest_idx]
        else:
            # Use the closest timestamp within window
            time_diffs = abs(matching_data['timestamp'] - event_time)
            nearest_idx = time_diffs.idxmin()
            nearest_data = matching_data.loc[nearest_idx:nearest_idx]
        
        # Merge news data with market data
        for _, row in nearest_data.iterrows():
            synchronized_record = {
                'news_id': news.id,
                'event_time': event_time,
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'title': news.title,
                'country': news.country,
                'volatility': news.volatility,
                'forecast': news.forecast,
                'previous': news.previous,
                'actual': news.actual,
            }
            synchronized_data.append(synchronized_record)
    
    result_df = pd.DataFrame(synchronized_data)
    
    if not result_df.empty:
        result_df = result_df.sort_values('event_time').reset_index(drop=True)
        logger.info(f"Created synchronized dataset with {len(result_df)} records")
    else:
        logger.warning("No synchronized records created")
    
    return result_df


def calculate_price_impact(
    news_event: EconomicNews,
    market_data: pd.DataFrame,
    window: int = 30
) -> Dict[str, float]:
    """
    Calculate the price impact of a news event on market prices.
    
    This function analyzes price movements before and after a news event
    to quantify its impact on the market.
    
    Args:
        news_event: EconomicNews object representing the event.
        market_data: DataFrame with market data (must have 'timestamp' and 'close').
        window: Number of minutes before/after event to analyze. Default is 30.
        
    Returns:
        Dictionary with impact metrics:
        - pre_event_price: Close price before event
        - post_event_price: Close price after event
        - absolute_change: Absolute price change
        - percentage_change: Percentage price change
        - volatility: Standard deviation of returns in window
        - max_excursion: Maximum price movement (up or down)
        - time_to_peak: Minutes to reach maximum price
    """
    logger.info(f"Calculating price impact for news event: {news_event.title}")
    
    if market_data.empty or news_event.event_time is None:
        logger.warning("Invalid inputs for price impact calculation")
        return {}
    
    # Ensure timestamps are timezone-aware
    event_time = news_event.event_time
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=UTC)
    
    if market_data['timestamp'].dt.tz is None:
        market_data = market_data.copy()
        market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(UTC)
    
    # Define time windows
    time_delta = timedelta(minutes=window)
    pre_event_start = event_time - time_delta
    post_event_end = event_time + time_delta
    
    # Filter data for pre-event and post-event windows
    pre_event_data = market_data[
        (market_data['timestamp'] >= pre_event_start) &
        (market_data['timestamp'] < event_time)
    ]
    
    post_event_data = market_data[
        (market_data['timestamp'] >= event_time) &
        (market_data['timestamp'] <= post_event_end)
    ]
    
    # Calculate impact metrics
    impact = {}
    
    # Get pre-event and post-event prices
    if not pre_event_data.empty:
        # Use last price before event
        pre_event_price = pre_event_data.iloc[-1]['close']
        impact['pre_event_price'] = float(pre_event_price)
    else:
        # Find nearest price before event
        pre_data = market_data[market_data['timestamp'] < event_time]
        if not pre_data.empty:
            pre_event_price = pre_data.iloc[-1]['close']
            impact['pre_event_price'] = float(pre_event_price)
        else:
            logger.warning("No pre-event data available")
            return {}
    
    if not post_event_data.empty:
        # Use last price in post-event window
        post_event_price = post_event_data.iloc[-1]['close']
        impact['post_event_price'] = float(post_event_price)
    else:
        # Find nearest price after event
        post_data = market_data[market_data['timestamp'] > event_time]
        if not post_data.empty:
            post_event_price = post_data.iloc[0]['close']
            impact['post_event_price'] = float(post_event_price)
        else:
            logger.warning("No post-event data available")
            return impact
    
    # Calculate price changes
    impact['absolute_change'] = float(post_event_price - pre_event_price)
    impact['percentage_change'] = float(
        (post_event_price - pre_event_price) / pre_event_price * 100
    )
    
    # Calculate volatility in the combined window
    combined_data = pd.concat([pre_event_data, post_event_data])
    if len(combined_data) > 1:
        returns = combined_data['close'].pct_change().dropna()
        impact['volatility'] = float(returns.std())
    else:
        impact['volatility'] = 0.0
    
    # Calculate maximum excursion (largest move from pre-event price)
    if not combined_data.empty:
        price_deviations = abs(combined_data['close'] - pre_event_price)
        max_deviation_idx = price_deviations.idxmax()
        
        impact['max_excursion'] = float(price_deviations.max())
        impact['max_excursion_pct'] = float(
            price_deviations.max() / pre_event_price * 100
        )
        
        # Calculate time to peak
        peak_time = combined_data.loc[max_deviation_idx, 'timestamp']
        time_diff = (peak_time - event_time).total_seconds() / 60.0
        impact['time_to_peak_minutes'] = float(time_diff)
    
    logger.info(f"Price impact: {impact.get('percentage_change', 0):.2f}%")
    return impact


def calculate_returns(market_data: pd.DataFrame, periods: List[int] = [1, 5, 15, 30]) -> pd.DataFrame:
    """
    Calculate returns over multiple time periods.
    
    Args:
        market_data: DataFrame with market data (must have 'close' column).
        periods: List of periods (in rows) for return calculation.
        
    Returns:
        DataFrame with original data plus return columns.
    """
    df = market_data.copy()
    
    for period in periods:
        col_name = f'return_{period}' if period > 1 else 'return'
        df[col_name] = df['close'].pct_change(periods=period)
    
    return df


def calculate_volatility(
    market_data: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        market_data: DataFrame with market data (must have 'close' column).
        window: Rolling window size for volatility calculation.
        annualize: Whether to annualize volatility (multiply by sqrt(252)).
        
    Returns:
        DataFrame with original data plus volatility column.
    """
    df = market_data.copy()
    
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Calculate rolling volatility
    volatility = returns.rolling(window=window).std()
    
    if annualize:
        volatility = volatility * np.sqrt(252)  # Assuming daily data
    
    df['volatility'] = volatility
    
    return df

