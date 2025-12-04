"""
Unit tests for market data integration module.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pytz import UTC

from src.analysis.market_data import (
    MarketDataFetcher,
    sync_with_news_events,
    calculate_price_impact,
    calculate_returns,
    calculate_volatility
)
from src.database.models import EconomicNews, MarketData
from src.database.repository import DatabaseRepository


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = Mock(spec=DatabaseRepository)
    repo.save_market_data = Mock(return_value=10)
    repo.get_market_data = Mock(return_value=[])
    return repo


@pytest.fixture
def fetcher(mock_repository):
    """Create MarketDataFetcher instance."""
    return MarketDataFetcher(mock_repository)


@pytest.fixture
def sample_market_data():
    """Create sample market data DataFrame."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D', tz=UTC)
    data = {
        'timestamp': dates,
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
        'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 106.0, 105.0, 104.0, 103.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1400000, 1300000, 1200000, 1100000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_news_events():
    """Create sample economic news events."""
    events = []
    base_time = datetime(2023, 1, 5, 14, 30, tzinfo=UTC)
    
    for i in range(3):
        news = Mock(spec=EconomicNews)
        news.id = i + 1
        news.title = f"Event {i + 1}"
        news.country = "USD"
        news.volatility = "High"
        news.event_time = base_time + timedelta(days=i)
        news.forecast = "5.0"
        news.previous = "4.5"
        news.actual = "5.2" if i > 0 else None
        events.append(news)
    
    return events


class TestMarketDataFetcher:
    """Tests for MarketDataFetcher class."""
    
    @patch('src.analysis.market_data.yf.Ticker')
    def test_fetch_historical_data_success(self, mock_ticker, fetcher):
        """Test successful data fetch from yfinance."""
        # Mock yfinance response
        mock_df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', end='2023-01-05', freq='D'),
            'Open': [100, 101, 102, 103, 104],
            'High': [102, 103, 104, 105, 106],
            'Low': [99, 100, 101, 102, 103],
            'Close': [101, 102, 103, 104, 105],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        }).set_index('Date')
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        result = fetcher.fetch_historical_data(
            symbol='SPY',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            save_to_db=False
        )
        
        # Assertions
        assert len(result) == 5
        assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert all(result['high'] >= result['low'])
        assert all(result['timestamp'].dt.tz == UTC)
    
    @patch('src.analysis.market_data.yf.Ticker')
    def test_fetch_historical_data_empty_response(self, mock_ticker, fetcher):
        """Test handling of empty data response."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No data returned"):
            fetcher.fetch_historical_data(
                symbol='INVALID',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 5)
            )
    
    def test_validate_and_clean_data(self, fetcher):
        """Test data validation and cleaning."""
        # Create data with issues
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', end='2023-01-05', freq='D'),
            'open': [100, 101, np.nan, 103, -5],  # NaN and negative
            'high': [102, 103, 104, 105, 106],
            'low': [99, 104, 101, 102, 103],  # low > high in row 1
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, np.nan, 1200000, 1300000, 1400000],
        })
        
        result = fetcher._validate_and_clean_data(df, 'SPY')
        
        # Should remove rows with NaN, negative, and invalid relationships
        assert len(result) < len(df)
        assert not result['open'].isna().any()
        assert not result['high'].isna().any()
        assert all(result['high'] >= result['low'])
        assert all(result['open'] > 0)
    
    @patch('src.analysis.market_data.yf.Ticker')
    def test_fetch_multiple_symbols(self, mock_ticker, fetcher):
        """Test fetching data for multiple symbols."""
        mock_df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', end='2023-01-05', freq='D'),
            'Open': [100, 101, 102, 103, 104],
            'High': [102, 103, 104, 105, 106],
            'Low': [99, 100, 101, 102, 103],
            'Close': [101, 102, 103, 104, 105],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        }).set_index('Date')
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        symbols = ['SPY', 'QQQ', 'DIA']
        results = fetcher.fetch_multiple_symbols(
            symbols=symbols,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5),
            save_to_db=False
        )
        
        assert len(results) == 3
        assert all(symbol in results for symbol in symbols)
        assert all(isinstance(df, pd.DataFrame) for df in results.values())
    
    def test_get_cached_data(self, fetcher, mock_repository):
        """Test retrieving cached data from database."""
        # Mock database records
        mock_records = []
        for i in range(5):
            record = Mock(spec=MarketData)
            record.to_dict.return_value = {
                'id': i + 1,
                'symbol': 'SPY',
                'timestamp': f'2023-01-0{i+1}T00:00:00+00:00',
                'open': 100.0 + i,
                'high': 102.0 + i,
                'low': 99.0 + i,
                'close': 101.0 + i,
                'volume': 1000000 + i * 100000,
            }
            mock_records.append(record)
        
        mock_repository.get_market_data.return_value = mock_records
        
        result = fetcher.get_cached_data('SPY')
        
        assert len(result) == 5
        assert result['symbol'].iloc[0] == 'SPY'
        mock_repository.get_market_data.assert_called_once()


class TestSyncWithNewsEvents:
    """Tests for sync_with_news_events function."""
    
    def test_sync_basic(self, sample_news_events, sample_market_data):
        """Test basic synchronization of news with market data."""
        result = sync_with_news_events(sample_news_events, sample_market_data)
        
        assert not result.empty
        assert 'news_id' in result.columns
        assert 'event_time' in result.columns
        assert 'timestamp' in result.columns
        assert 'close' in result.columns
    
    def test_sync_empty_market_data(self, sample_news_events):
        """Test synchronization with empty market data."""
        empty_df = pd.DataFrame()
        result = sync_with_news_events(sample_news_events, empty_df)
        
        assert result.empty
    
    def test_sync_with_time_window(self, sample_market_data):
        """Test synchronization with custom time window."""
        # Create news event at specific time
        news = Mock(spec=EconomicNews)
        news.id = 1
        news.title = "Test Event"
        news.country = "USD"
        news.volatility = "High"
        news.event_time = datetime(2023, 1, 5, 14, 30, tzinfo=UTC)
        news.forecast = "5.0"
        news.previous = "4.5"
        news.actual = None
        
        result = sync_with_news_events([news], sample_market_data, time_window_minutes=60)
        
        assert not result.empty
        assert result['news_id'].iloc[0] == 1
    
    def test_sync_timezone_aware(self, sample_market_data):
        """Test that synchronization handles timezones correctly."""
        news = Mock(spec=EconomicNews)
        news.id = 1
        news.title = "Test Event"
        news.country = "USD"
        news.volatility = "High"
        news.event_time = datetime(2023, 1, 5, 14, 30, tzinfo=UTC)
        news.forecast = "5.0"
        news.previous = "4.5"
        news.actual = None
        
        result = sync_with_news_events([news], sample_market_data)
        
        assert not result.empty
        assert result['timestamp'].dtype == 'datetime64[ns, UTC]'


class TestCalculatePriceImpact:
    """Tests for calculate_price_impact function."""
    
    def test_calculate_impact_basic(self, sample_market_data):
        """Test basic price impact calculation."""
        news = Mock(spec=EconomicNews)
        news.id = 1
        news.title = "Test Event"
        news.event_time = datetime(2023, 1, 5, 12, 0, tzinfo=UTC)
        
        impact = calculate_price_impact(news, sample_market_data, window=60)
        
        assert 'pre_event_price' in impact
        assert 'post_event_price' in impact
        assert 'absolute_change' in impact
        assert 'percentage_change' in impact
        assert 'volatility' in impact
    
    def test_calculate_impact_empty_data(self):
        """Test price impact with empty market data."""
        news = Mock(spec=EconomicNews)
        news.id = 1
        news.title = "Test Event"
        news.event_time = datetime(2023, 1, 5, 12, 0, tzinfo=UTC)
        
        impact = calculate_price_impact(news, pd.DataFrame(), window=60)
        
        assert impact == {}
    
    def test_calculate_impact_no_event_time(self, sample_market_data):
        """Test price impact when news has no event time."""
        news = Mock(spec=EconomicNews)
        news.id = 1
        news.title = "Test Event"
        news.event_time = None
        
        impact = calculate_price_impact(news, sample_market_data, window=60)
        
        assert impact == {}
    
    def test_calculate_impact_metrics(self, sample_market_data):
        """Test that all impact metrics are calculated correctly."""
        news = Mock(spec=EconomicNews)
        news.id = 1
        news.title = "Test Event"
        news.event_time = datetime(2023, 1, 5, 12, 0, tzinfo=UTC)
        
        impact = calculate_price_impact(news, sample_market_data, window=120)
        
        # Check that percentage change is calculated correctly
        if 'pre_event_price' in impact and 'post_event_price' in impact:
            expected_pct = (
                (impact['post_event_price'] - impact['pre_event_price']) 
                / impact['pre_event_price'] * 100
            )
            assert abs(impact['percentage_change'] - expected_pct) < 0.01


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_calculate_returns(self, sample_market_data):
        """Test return calculation."""
        result = calculate_returns(sample_market_data, periods=[1, 5])
        
        assert 'return' in result.columns
        assert 'return_5' in result.columns
        
        # Check that returns are calculated correctly
        expected_return = (result['close'].iloc[1] - result['close'].iloc[0]) / result['close'].iloc[0]
        assert abs(result['return'].iloc[1] - expected_return) < 0.0001
    
    def test_calculate_volatility(self, sample_market_data):
        """Test volatility calculation."""
        result = calculate_volatility(sample_market_data, window=5, annualize=False)
        
        assert 'volatility' in result.columns
        
        # Volatility should be NaN for first few rows due to rolling window
        assert result['volatility'].iloc[:4].isna().all()
        
        # Volatility should be non-negative where it exists
        assert (result['volatility'].dropna() >= 0).all()
    
    def test_calculate_volatility_annualized(self, sample_market_data):
        """Test annualized volatility calculation."""
        result_annual = calculate_volatility(sample_market_data, window=5, annualize=True)
        result_non_annual = calculate_volatility(sample_market_data, window=5, annualize=False)
        
        # Annualized should be larger (multiplied by sqrt(252))
        vol_annual = result_annual['volatility'].dropna().iloc[0]
        vol_non_annual = result_non_annual['volatility'].dropna().iloc[0]
        
        assert vol_annual > vol_non_annual


class TestIntegration:
    """Integration tests with actual database."""
    
    @pytest.mark.integration
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from fetch to database."""
        # Create temporary database
        db_path = tmp_path / "test.db"
        repo = DatabaseRepository(f"sqlite:///{db_path}")
        repo.create_tables()
        
        fetcher = MarketDataFetcher(repo)
        
        # Mock yfinance to avoid actual API calls
        with patch('src.analysis.market_data.yf.Ticker') as mock_ticker:
            mock_df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', end='2023-01-05', freq='D'),
                'Open': [100, 101, 102, 103, 104],
                'High': [102, 103, 104, 105, 106],
                'Low': [99, 100, 101, 102, 103],
                'Close': [101, 102, 103, 104, 105],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            }).set_index('Date')
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_df
            mock_ticker.return_value = mock_ticker_instance
            
            # Fetch and save data
            df = fetcher.fetch_historical_data(
                symbol='SPY',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 5),
                save_to_db=True
            )
            
            # Verify data was saved
            cached_data = fetcher.get_cached_data('SPY')
            assert len(cached_data) == 5
            assert cached_data['symbol'].iloc[0] == 'SPY'

