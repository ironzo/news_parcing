"""
Feature engineering module for transforming raw data into RL-compatible observations.

This module provides functionality to:
- Extract news features: forecast deviation, volatility level, event timing
- Calculate market features: price momentum, volatility, volume patterns
- Compute technical indicators: RSI, MACD, Bollinger Bands
- Build news-market correlation features
- Normalize and scale features for RL training
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pytz import UTC

from ..database.models import EconomicNews


logger = logging.getLogger(__name__)


# Feature dimension constants
NEWS_FEATURE_DIM = 5
MARKET_FEATURE_DIM = 15
POSITION_FEATURE_DIM = 5
TOTAL_FEATURE_DIM = NEWS_FEATURE_DIM + MARKET_FEATURE_DIM + POSITION_FEATURE_DIM  # 25


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Momentum periods
    momentum_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Volatility window
    volatility_window: int = 20
    
    # Volume MA window
    volume_ma_window: int = 20
    
    # Feature scaling bounds
    clip_lower: float = -5.0
    clip_upper: float = 5.0


class FeatureExtractor:
    """
    Extract and engineer features from market data and news events.
    
    Creates a 25-dimensional observation vector:
    - News context (5 dims): deviation ratio, hours until event, volatility score, etc.
    - Market state (15 dims): returns, volatility, volume, technical indicators
    - Position state (5 dims): current position, unrealized P&L, holding period
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Feature engineering configuration.
        """
        self.config = config or FeatureConfig()
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        logger.info("Initialized FeatureExtractor")
    
    def extract_all_features(
        self,
        market_data: pd.DataFrame,
        news_events: Optional[List[EconomicNews]] = None,
        current_position: float = 0.0,
        entry_price: float = 0.0,
        holding_periods: int = 0,
        current_price: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract complete 25-dimensional feature vector.
        
        Args:
            market_data: DataFrame with OHLCV data (must be sorted by timestamp).
            news_events: List of upcoming news events.
            current_position: Current position (-1 short, 0 flat, 1 long).
            entry_price: Entry price of current position.
            holding_periods: Number of periods position has been held.
            current_price: Current market price (uses last close if None).
            
        Returns:
            25-dimensional numpy array of features.
        """
        if market_data.empty:
            logger.warning("Empty market data, returning zeros")
            return np.zeros(TOTAL_FEATURE_DIM)
        
        # Get current price
        if current_price is None:
            current_price = market_data['close'].iloc[-1]
        
        # Extract each feature group
        news_features = self.extract_news_features(news_events, market_data)
        market_features = self.extract_market_features(market_data)
        position_features = self.extract_position_features(
            current_position, entry_price, holding_periods, current_price
        )
        
        # Concatenate all features
        features = np.concatenate([
            news_features,      # 5 dims
            market_features,    # 15 dims
            position_features   # 5 dims
        ])
        
        return features
    
    def extract_news_features(
        self,
        news_events: Optional[List[EconomicNews]],
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract news-related features (5 dimensions).
        
        Features:
        1. Forecast deviation ratio (actual vs forecast)
        2. Hours until next major event
        3. Volatility score (0-3 scale)
        4. News density (number of events in lookback)
        5. Surprise factor (deviation from consensus)
        
        Args:
            news_events: List of economic news events.
            market_data: Market data for timing reference.
            
        Returns:
            5-dimensional numpy array.
        """
        features = np.zeros(NEWS_FEATURE_DIM)
        
        if not news_events or market_data.empty:
            return features
        
        # Get current timestamp from market data
        current_time = market_data['timestamp'].iloc[-1]
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        
        # Process news events
        upcoming_events = []
        recent_events = []
        
        for news in news_events:
            if news.event_time is None:
                continue
            
            event_time = news.event_time
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=UTC)
            
            time_diff = (event_time - current_time).total_seconds() / 3600  # hours
            
            if time_diff > 0:  # Future event
                upcoming_events.append((news, time_diff))
            elif time_diff > -24:  # Past 24 hours
                recent_events.append(news)
        
        # Feature 1: Forecast deviation ratio (from most recent event with actual value)
        deviation_ratio = 0.0
        for news in recent_events:
            if news.actual and news.forecast:
                try:
                    actual = self._parse_numeric(news.actual)
                    forecast = self._parse_numeric(news.forecast)
                    if forecast != 0:
                        deviation_ratio = (actual - forecast) / abs(forecast)
                        break
                except (ValueError, TypeError):
                    pass
        features[0] = np.clip(deviation_ratio, -2.0, 2.0)
        
        # Feature 2: Hours until next major event
        if upcoming_events:
            upcoming_events.sort(key=lambda x: x[1])
            hours_to_event = upcoming_events[0][1]
            # Normalize: 0 = imminent, 1 = 24+ hours away
            features[1] = min(hours_to_event / 24.0, 1.0)
        else:
            features[1] = 1.0  # No upcoming events
        
        # Feature 3: Volatility score (0-3 scale based on event importance)
        max_volatility = 0
        for news, _ in upcoming_events[:3]:  # Check next 3 events
            vol_score = self._parse_volatility_score(news.volatility)
            max_volatility = max(max_volatility, vol_score)
        features[2] = max_volatility / 3.0  # Normalize to [0, 1]
        
        # Feature 4: News density (number of events in past/next 24 hours)
        event_count = len(recent_events) + len([e for e in upcoming_events if e[1] <= 24])
        features[3] = min(event_count / 10.0, 1.0)  # Normalize, cap at 10
        
        # Feature 5: Surprise factor (weighted average of recent deviations)
        surprises = []
        for news in recent_events[:5]:
            if news.actual and news.previous:
                try:
                    actual = self._parse_numeric(news.actual)
                    previous = self._parse_numeric(news.previous)
                    if previous != 0:
                        surprises.append(abs(actual - previous) / abs(previous))
                except (ValueError, TypeError):
                    pass
        
        if surprises:
            features[4] = np.clip(np.mean(surprises), 0.0, 1.0)
        
        return features
    
    def extract_market_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract market-related features (15 dimensions).
        
        Features:
        1-4. Returns: 1-day, 5-day, 10-day, 20-day
        5. Current volatility (20-day)
        6. Volatility percentile (vs historical)
        7. Volume ratio (vs 20-day MA)
        8-9. RSI and RSI momentum
        10-12. MACD, Signal, Histogram
        13-14. Bollinger %B and bandwidth
        15. Trend strength (ADX-like)
        
        Args:
            market_data: DataFrame with OHLCV data.
            
        Returns:
            15-dimensional numpy array.
        """
        features = np.zeros(MARKET_FEATURE_DIM)
        
        if market_data.empty or len(market_data) < 30:
            logger.warning(f"Insufficient market data ({len(market_data)} rows), need at least 30")
            return features
        
        df = market_data.copy()
        
        # Calculate returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Features 1-4: Returns (normalized by volatility)
        volatility = df['return_1'].rolling(20).std().iloc[-1]
        if volatility > 0 and not np.isnan(volatility):
            features[0] = df['return_1'].iloc[-1] / volatility
            features[1] = df['return_5'].iloc[-1] / (volatility * np.sqrt(5))
            features[2] = df['return_10'].iloc[-1] / (volatility * np.sqrt(10))
            features[3] = df['return_20'].iloc[-1] / (volatility * np.sqrt(20))
        
        # Feature 5: Current volatility (annualized)
        ann_volatility = volatility * np.sqrt(252) if volatility else 0
        features[4] = np.clip(ann_volatility / 0.5, 0, 2)  # Normalize by typical vol
        
        # Feature 6: Volatility percentile
        if len(df) >= 252:
            hist_vols = df['return_1'].rolling(20).std()
            current_vol_percentile = (hist_vols < volatility).mean()
            features[5] = current_vol_percentile
        
        # Feature 7: Volume ratio
        df['volume_ma'] = df['volume'].rolling(self.config.volume_ma_window).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1]
        if volume_ma > 0 and not np.isnan(volume_ma):
            features[6] = np.clip(current_volume / volume_ma - 1, -1, 3)  # Center at 0
        
        # Features 8-9: RSI and RSI momentum
        rsi = self._calculate_rsi(df['close'], self.config.rsi_period)
        features[7] = (rsi.iloc[-1] - 50) / 50 if not np.isnan(rsi.iloc[-1]) else 0
        rsi_momentum = rsi.diff(5).iloc[-1]
        features[8] = np.clip(rsi_momentum / 20, -1, 1) if not np.isnan(rsi_momentum) else 0
        
        # Features 10-12: MACD
        macd, signal, histogram = self._calculate_macd(
            df['close'],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        price = df['close'].iloc[-1]
        if price > 0:
            features[9] = np.clip(macd.iloc[-1] / price * 100, -2, 2)
            features[10] = np.clip(signal.iloc[-1] / price * 100, -2, 2)
            features[11] = np.clip(histogram.iloc[-1] / price * 100, -1, 1)
        
        # Features 13-14: Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            df['close'],
            self.config.bollinger_period,
            self.config.bollinger_std
        )
        bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        if bb_width > 0:
            # %B: where price is within bands (0 = lower, 1 = upper)
            percent_b = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / bb_width
            features[12] = np.clip(percent_b - 0.5, -1, 1)  # Center at 0
            # Bandwidth normalized
            features[13] = np.clip(bb_width / bb_middle.iloc[-1] / 0.1, 0, 2)
        
        # Feature 15: Trend strength (simplified ADX-like)
        trend_strength = self._calculate_trend_strength(df)
        features[14] = np.clip(trend_strength, -1, 1)
        
        # Clip all features
        features = np.clip(features, self.config.clip_lower, self.config.clip_upper)
        
        return features
    
    def extract_position_features(
        self,
        current_position: float,
        entry_price: float,
        holding_periods: int,
        current_price: float
    ) -> np.ndarray:
        """
        Extract position-related features (5 dimensions).
        
        Features:
        1. Position direction (-1, 0, 1)
        2. Unrealized P&L percentage
        3. Holding period (normalized)
        4. Distance from entry (absolute)
        5. Position heat (risk indicator)
        
        Args:
            current_position: Position direction (-1 short, 0 flat, 1 long).
            entry_price: Entry price of current position.
            holding_periods: Number of periods position has been held.
            current_price: Current market price.
            
        Returns:
            5-dimensional numpy array.
        """
        features = np.zeros(POSITION_FEATURE_DIM)
        
        # Feature 1: Position direction
        features[0] = np.clip(current_position, -1, 1)
        
        # Features 2-5 only matter if we have a position
        if current_position != 0 and entry_price > 0:
            # Feature 2: Unrealized P&L percentage
            pnl_pct = (current_price - entry_price) / entry_price * current_position
            features[1] = np.clip(pnl_pct * 10, -2, 2)  # Scale by 10 for sensitivity
            
            # Feature 3: Holding period (normalized, cap at 20 days)
            features[2] = min(holding_periods / 20.0, 1.0)
            
            # Feature 4: Distance from entry (absolute percentage)
            distance = abs(current_price - entry_price) / entry_price
            features[3] = np.clip(distance * 10, 0, 2)
            
            # Feature 5: Position heat (risk indicator based on P&L and time)
            # Negative P&L + long holding = higher heat
            heat = -pnl_pct * (1 + holding_periods / 10)
            features[4] = np.clip(heat, -1, 1)
        
        return features
    
    def prepare_features_dataframe(
        self,
        market_data: pd.DataFrame,
        news_events: Optional[List[EconomicNews]] = None
    ) -> pd.DataFrame:
        """
        Prepare a full DataFrame of features for backtesting.
        
        Args:
            market_data: DataFrame with OHLCV data.
            news_events: List of economic news events.
            
        Returns:
            DataFrame with features for each timestamp.
        """
        if market_data.empty:
            return pd.DataFrame()
        
        # Pre-calculate all technical indicators
        df = self._add_technical_indicators(market_data.copy())
        
        # Create feature columns
        feature_columns = [
            # News features
            'news_deviation', 'hours_to_event', 'volatility_score',
            'news_density', 'surprise_factor',
            # Market features
            'return_1_norm', 'return_5_norm', 'return_10_norm', 'return_20_norm',
            'volatility', 'volatility_pctl', 'volume_ratio',
            'rsi_norm', 'rsi_momentum',
            'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            'bb_pct_b', 'bb_bandwidth',
            'trend_strength',
            # Position features (initialized to 0)
            'position', 'unrealized_pnl', 'holding_period',
            'distance_from_entry', 'position_heat'
        ]
        
        for col in feature_columns:
            df[col] = 0.0
        
        # Calculate market features for each row (rolling window)
        min_lookback = max(
            self.config.bollinger_period,
            self.config.macd_slow + self.config.macd_signal,
            self.config.rsi_period,
            max(self.config.momentum_periods)
        ) + 10  # Extra buffer
        
        for i in range(min_lookback, len(df)):
            window_data = df.iloc[:i+1].copy()
            market_features = self.extract_market_features(window_data)
            
            # Assign market features
            df.iloc[i, df.columns.get_loc('return_1_norm'):df.columns.get_loc('trend_strength')+1] = market_features
            
            # News features (if events provided)
            if news_events:
                news_features = self.extract_news_features(news_events, window_data)
                df.iloc[i, df.columns.get_loc('news_deviation'):df.columns.get_loc('surprise_factor')+1] = news_features
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        # Returns
        for period in self.config.momentum_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Volatility
        df['volatility_raw'] = df['return_1'].rolling(self.config.volatility_window).std()
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(self.config.volume_ma_window).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.config.rsi_period)
        
        # MACD
        macd, signal, histogram = self._calculate_macd(
            df['close'],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(
            df['close'],
            self.config.bollinger_period,
            self.config.bollinger_std
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength indicator.
        
        Uses a simplified approach based on directional movement.
        """
        if len(df) < 20:
            return 0.0
        
        # Calculate average directional movement
        recent = df.tail(20)
        
        # Price trend
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Consistency of trend (how many days moved in same direction)
        daily_changes = recent['close'].diff().dropna()
        positive_days = (daily_changes > 0).sum()
        consistency = abs(positive_days / len(daily_changes) - 0.5) * 2  # 0 to 1
        
        # Combine price change direction with consistency
        trend_strength = np.sign(price_change) * consistency
        
        return trend_strength
    
    def _parse_numeric(self, value: str) -> float:
        """Parse numeric value from string, handling percentage signs."""
        if isinstance(value, (int, float)):
            return float(value)
        
        value = str(value).strip()
        value = value.replace('%', '').replace(',', '').replace('$', '')
        value = value.replace('K', 'e3').replace('M', 'e6').replace('B', 'e9')
        
        return float(value)
    
    def _parse_volatility_score(self, volatility: Optional[str]) -> int:
        """Convert volatility string to numeric score (0-3)."""
        if not volatility:
            return 0
        
        volatility = str(volatility).lower()
        
        if 'high' in volatility or '3' in volatility:
            return 3
        elif 'medium' in volatility or 'moderate' in volatility or '2' in volatility:
            return 2
        elif 'low' in volatility or '1' in volatility:
            return 1
        
        return 0
    
    def fit_scaler(self, features: np.ndarray) -> None:
        """
        Fit feature scaler on training data.
        
        Args:
            features: 2D array of features (n_samples, n_features).
        """
        self._feature_means = np.mean(features, axis=0)
        self._feature_stds = np.std(features, axis=0)
        # Avoid division by zero
        self._feature_stds = np.where(self._feature_stds == 0, 1.0, self._feature_stds)
        logger.info(f"Fitted scaler on {len(features)} samples")
    
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """
        Scale features using fitted scaler (z-score normalization).
        
        Args:
            features: Feature array (1D or 2D).
            
        Returns:
            Scaled features.
        """
        if self._feature_means is None:
            logger.warning("Scaler not fitted, returning unscaled features")
            return features
        
        scaled = (features - self._feature_means) / self._feature_stds
        return np.clip(scaled, self.config.clip_lower, self.config.clip_upper)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # News features (5)
            'news_deviation_ratio',
            'hours_to_event',
            'volatility_score',
            'news_density',
            'surprise_factor',
            # Market features (15)
            'return_1d_norm',
            'return_5d_norm',
            'return_10d_norm',
            'return_20d_norm',
            'volatility',
            'volatility_percentile',
            'volume_ratio',
            'rsi_normalized',
            'rsi_momentum',
            'macd_normalized',
            'macd_signal_normalized',
            'macd_histogram_normalized',
            'bollinger_pct_b',
            'bollinger_bandwidth',
            'trend_strength',
            # Position features (5)
            'position_direction',
            'unrealized_pnl',
            'holding_period',
            'distance_from_entry',
            'position_heat'
        ]

