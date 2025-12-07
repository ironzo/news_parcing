"""
Unit tests for feature engineering module.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
import numpy as np
import pandas as pd
from pytz import UTC

from src.analysis.features import (
    FeatureExtractor,
    FeatureConfig,
    NEWS_FEATURE_DIM,
    MARKET_FEATURE_DIM,
    POSITION_FEATURE_DIM,
    TOTAL_FEATURE_DIM
)
from src.database.models import EconomicNews


@pytest.fixture
def feature_config():
    """Create feature configuration."""
    return FeatureConfig(
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bollinger_period=20,
        bollinger_std=2.0
    )


@pytest.fixture
def feature_extractor(feature_config):
    """Create feature extractor instance."""
    return FeatureExtractor(feature_config)


@pytest.fixture
def sample_market_data():
    """Create sample market data DataFrame with enough history."""
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D', tz=UTC)
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.015, n_days)
    prices = base_price * np.cumprod(1 + returns)
    
    data = {
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'high': prices * (1 + np.random.uniform(0.005, 0.015, n_days)),
        'low': prices * (1 - np.random.uniform(0.005, 0.015, n_days)),
        'close': prices,
        'volume': np.random.uniform(1e6, 5e6, n_days),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_news_events():
    """Create sample economic news events."""
    events = []
    base_time = datetime(2023, 3, 15, 14, 30, tzinfo=UTC)
    
    for i in range(5):
        news = Mock(spec=EconomicNews)
        news.id = i + 1
        news.title = f"Economic Event {i + 1}"
        news.country = "USD"
        news.volatility = ["Low", "Medium", "High", "High", "Medium"][i]
        news.event_time = base_time + timedelta(days=i-2)  # Some past, some future
        news.forecast = str(5.0 + i * 0.5)
        news.previous = str(4.5 + i * 0.4)
        news.actual = str(5.2 + i * 0.5) if i < 3 else None  # Past events have actual
        events.append(news)
    
    return events


class TestFeatureDimensions:
    """Test feature dimension constants."""
    
    def test_total_dimension(self):
        """Test that total dimension is correct."""
        assert TOTAL_FEATURE_DIM == 25
        assert TOTAL_FEATURE_DIM == NEWS_FEATURE_DIM + MARKET_FEATURE_DIM + POSITION_FEATURE_DIM
    
    def test_individual_dimensions(self):
        """Test individual feature group dimensions."""
        assert NEWS_FEATURE_DIM == 5
        assert MARKET_FEATURE_DIM == 15
        assert POSITION_FEATURE_DIM == 5


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    def test_initialization(self, feature_extractor):
        """Test feature extractor initialization."""
        assert feature_extractor is not None
        assert feature_extractor.config is not None
    
    def test_extract_all_features_shape(self, feature_extractor, sample_market_data):
        """Test that extract_all_features returns correct shape."""
        features = feature_extractor.extract_all_features(sample_market_data)
        
        assert features.shape == (TOTAL_FEATURE_DIM,)
        assert features.dtype == np.float64
    
    def test_extract_all_features_with_news(
        self, feature_extractor, sample_market_data, sample_news_events
    ):
        """Test feature extraction with news events."""
        features = feature_extractor.extract_all_features(
            market_data=sample_market_data,
            news_events=sample_news_events
        )
        
        assert features.shape == (TOTAL_FEATURE_DIM,)
    
    def test_extract_all_features_with_position(
        self, feature_extractor, sample_market_data
    ):
        """Test feature extraction with active position."""
        features = feature_extractor.extract_all_features(
            market_data=sample_market_data,
            current_position=1.0,
            entry_price=100.0,
            holding_periods=5,
            current_price=105.0
        )
        
        assert features.shape == (TOTAL_FEATURE_DIM,)
        # Position feature should be 1.0
        assert features[20] == 1.0  # Position direction
    
    def test_extract_all_features_empty_data(self, feature_extractor):
        """Test feature extraction with empty data."""
        features = feature_extractor.extract_all_features(pd.DataFrame())
        
        assert features.shape == (TOTAL_FEATURE_DIM,)
        assert np.all(features == 0)


class TestNewsFeatures:
    """Tests for news feature extraction."""
    
    def test_news_features_shape(
        self, feature_extractor, sample_news_events, sample_market_data
    ):
        """Test news features have correct shape."""
        features = feature_extractor.extract_news_features(
            sample_news_events, sample_market_data
        )
        
        assert features.shape == (NEWS_FEATURE_DIM,)
    
    def test_news_features_no_events(self, feature_extractor, sample_market_data):
        """Test news features with no events."""
        features = feature_extractor.extract_news_features(None, sample_market_data)
        
        assert features.shape == (NEWS_FEATURE_DIM,)
        assert np.all(features == 0)
    
    def test_news_features_empty_market_data(
        self, feature_extractor, sample_news_events
    ):
        """Test news features with empty market data."""
        features = feature_extractor.extract_news_features(
            sample_news_events, pd.DataFrame()
        )
        
        assert features.shape == (NEWS_FEATURE_DIM,)
        assert np.all(features == 0)
    
    def test_volatility_score_parsing(self, feature_extractor):
        """Test volatility score parsing."""
        assert feature_extractor._parse_volatility_score("High") == 3
        assert feature_extractor._parse_volatility_score("Medium") == 2
        assert feature_extractor._parse_volatility_score("Low") == 1
        assert feature_extractor._parse_volatility_score(None) == 0
        assert feature_extractor._parse_volatility_score("3 bulls") == 3


class TestMarketFeatures:
    """Tests for market feature extraction."""
    
    def test_market_features_shape(self, feature_extractor, sample_market_data):
        """Test market features have correct shape."""
        features = feature_extractor.extract_market_features(sample_market_data)
        
        assert features.shape == (MARKET_FEATURE_DIM,)
    
    def test_market_features_insufficient_data(self, feature_extractor):
        """Test market features with insufficient data."""
        short_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D', tz=UTC),
            'open': [100] * 10,
            'high': [102] * 10,
            'low': [99] * 10,
            'close': [101] * 10,
            'volume': [1e6] * 10,
        })
        
        features = feature_extractor.extract_market_features(short_data)
        
        assert features.shape == (MARKET_FEATURE_DIM,)
        assert np.all(features == 0)  # Should return zeros for insufficient data
    
    def test_market_features_bounded(self, feature_extractor, sample_market_data):
        """Test that market features are properly bounded."""
        features = feature_extractor.extract_market_features(sample_market_data)
        
        # All features should be within clip bounds
        assert np.all(features >= -5.0)
        assert np.all(features <= 5.0)


class TestPositionFeatures:
    """Tests for position feature extraction."""
    
    def test_position_features_shape(self, feature_extractor):
        """Test position features have correct shape."""
        features = feature_extractor.extract_position_features(
            current_position=0.0,
            entry_price=0.0,
            holding_periods=0,
            current_price=100.0
        )
        
        assert features.shape == (POSITION_FEATURE_DIM,)
    
    def test_position_features_flat(self, feature_extractor):
        """Test position features when flat (no position)."""
        features = feature_extractor.extract_position_features(
            current_position=0.0,
            entry_price=0.0,
            holding_periods=0,
            current_price=100.0
        )
        
        assert features[0] == 0.0  # Position direction
        assert features[1] == 0.0  # Unrealized P&L
    
    def test_position_features_long(self, feature_extractor):
        """Test position features with long position."""
        features = feature_extractor.extract_position_features(
            current_position=1.0,
            entry_price=100.0,
            holding_periods=5,
            current_price=110.0
        )
        
        assert features[0] == 1.0  # Position direction
        assert features[1] > 0  # Positive unrealized P&L
        assert features[2] == 0.25  # Holding period (5/20)
    
    def test_position_features_short(self, feature_extractor):
        """Test position features with short position."""
        features = feature_extractor.extract_position_features(
            current_position=-1.0,
            entry_price=100.0,
            holding_periods=3,
            current_price=95.0
        )
        
        assert features[0] == -1.0  # Position direction
        assert features[1] > 0  # Positive unrealized P&L (short profit)


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""
    
    def test_rsi_calculation(self, feature_extractor, sample_market_data):
        """Test RSI calculation."""
        rsi = feature_extractor._calculate_rsi(sample_market_data['close'])
        
        assert len(rsi) == len(sample_market_data)
        # RSI should be between 0 and 100 (where not NaN)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self, feature_extractor, sample_market_data):
        """Test MACD calculation."""
        macd, signal, histogram = feature_extractor._calculate_macd(
            sample_market_data['close']
        )
        
        assert len(macd) == len(sample_market_data)
        assert len(signal) == len(sample_market_data)
        assert len(histogram) == len(sample_market_data)
        
        # Histogram should be macd - signal
        valid_idx = ~(macd.isna() | signal.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx],
            macd[valid_idx] - signal[valid_idx]
        )
    
    def test_bollinger_bands_calculation(self, feature_extractor, sample_market_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = feature_extractor._calculate_bollinger_bands(
            sample_market_data['close']
        )
        
        assert len(upper) == len(sample_market_data)
        assert len(middle) == len(sample_market_data)
        assert len(lower) == len(sample_market_data)
        
        # Upper should be > middle > lower (where not NaN)
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestFeatureScaling:
    """Tests for feature scaling functionality."""
    
    def test_fit_scaler(self, feature_extractor):
        """Test scaler fitting."""
        # Generate random features
        features = np.random.randn(100, TOTAL_FEATURE_DIM)
        
        feature_extractor.fit_scaler(features)
        
        assert feature_extractor._feature_means is not None
        assert feature_extractor._feature_stds is not None
        assert len(feature_extractor._feature_means) == TOTAL_FEATURE_DIM
        assert len(feature_extractor._feature_stds) == TOTAL_FEATURE_DIM
    
    def test_scale_features(self, feature_extractor):
        """Test feature scaling."""
        # Generate and fit on training data
        train_features = np.random.randn(100, TOTAL_FEATURE_DIM)
        feature_extractor.fit_scaler(train_features)
        
        # Scale test features
        test_features = np.random.randn(10, TOTAL_FEATURE_DIM)
        scaled = feature_extractor.scale_features(test_features)
        
        assert scaled.shape == test_features.shape
        # Scaled features should be clipped
        assert np.all(scaled >= -5.0)
        assert np.all(scaled <= 5.0)
    
    def test_scale_features_without_fitting(self, feature_extractor):
        """Test scaling without fitting returns original features."""
        features = np.random.randn(TOTAL_FEATURE_DIM)
        scaled = feature_extractor.scale_features(features)
        
        np.testing.assert_array_equal(scaled, features)


class TestFeatureNames:
    """Tests for feature name retrieval."""
    
    def test_get_feature_names(self, feature_extractor):
        """Test getting feature names."""
        names = feature_extractor.get_feature_names()
        
        assert len(names) == TOTAL_FEATURE_DIM
        assert all(isinstance(name, str) for name in names)
    
    def test_feature_names_unique(self, feature_extractor):
        """Test that all feature names are unique."""
        names = feature_extractor.get_feature_names()
        
        assert len(names) == len(set(names))


class TestNumericParsing:
    """Tests for numeric value parsing."""
    
    def test_parse_numeric_float(self, feature_extractor):
        """Test parsing float values."""
        assert feature_extractor._parse_numeric(5.5) == 5.5
        assert feature_extractor._parse_numeric("5.5") == 5.5
    
    def test_parse_numeric_with_symbols(self, feature_extractor):
        """Test parsing values with symbols."""
        assert feature_extractor._parse_numeric("5.5%") == 5.5
        assert feature_extractor._parse_numeric("$100") == 100.0
        assert feature_extractor._parse_numeric("1,000") == 1000.0
    
    def test_parse_numeric_with_suffixes(self, feature_extractor):
        """Test parsing values with K/M/B suffixes."""
        assert feature_extractor._parse_numeric("5K") == 5000.0
        assert feature_extractor._parse_numeric("2M") == 2000000.0
        assert feature_extractor._parse_numeric("1B") == 1000000000.0

