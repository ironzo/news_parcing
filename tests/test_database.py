"""
Tests for database models and repository.
"""
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, EconomicNews, MarketData, TradingSignal
from src.database.repository import DatabaseRepository


@pytest.fixture
def db_repo():
    """Create an in-memory database for testing."""
    repo = DatabaseRepository("sqlite:///:memory:")
    repo.create_tables()
    return repo


@pytest.fixture
def sample_news_data():
    """Sample news data for testing."""
    return [
        {
            'title': 'Initial Jobless Claims',
            'country': 'USD',
            'volatility': 'High Volatility Expected',
            'time': '2024/02/01 08:30:00',
            'forecast': '213K',
            'previous': '214K',
            'link': 'https://www.investing.com/economic-calendar/initial-jobless-claims-294',
        },
        {
            'title': 'NFP',
            'country': 'USD',
            'volatility': 'High Volatility Expected',
            'time': '2024/02/05 08:30:00',
            'forecast': '180K',
            'previous': '175K',
            'link': 'https://www.investing.com/economic-calendar/nfp-227',
        }
    ]


def test_database_initialization(db_repo):
    """Test database and table creation."""
    assert db_repo.engine is not None
    assert db_repo.SessionLocal is not None


def test_save_news(db_repo, sample_news_data):
    """Test saving news items to database."""
    count = db_repo.save_news(sample_news_data)
    
    assert count == 2
    
    # Verify data was saved
    news = db_repo.get_news()
    assert len(news) == 2


def test_get_news_filtered_by_country(db_repo, sample_news_data):
    """Test retrieving news filtered by country."""
    db_repo.save_news(sample_news_data)
    
    news = db_repo.get_news(country="USD")
    
    assert len(news) == 2
    assert all(item.country == "USD" for item in news)


def test_get_news_with_limit(db_repo, sample_news_data):
    """Test retrieving news with limit."""
    db_repo.save_news(sample_news_data)
    
    news = db_repo.get_news(limit=1)
    
    assert len(news) == 1


def test_save_market_data(db_repo):
    """Test saving market data."""
    market_data = [
        {
            'timestamp': datetime(2024, 1, 1, 9, 30),
            'open': 100.0,
            'high': 102.0,
            'low': 99.0,
            'close': 101.0,
            'volume': 1000000,
        },
        {
            'timestamp': datetime(2024, 1, 2, 9, 30),
            'open': 101.0,
            'high': 103.0,
            'low': 100.0,
            'close': 102.0,
            'volume': 1100000,
        }
    ]
    
    count = db_repo.save_market_data("SPY", market_data)
    
    assert count == 2


def test_get_market_data(db_repo):
    """Test retrieving market data."""
    market_data = [
        {
            'timestamp': datetime(2024, 1, 1, 9, 30),
            'open': 100.0,
            'high': 102.0,
            'low': 99.0,
            'close': 101.0,
            'volume': 1000000,
        }
    ]
    
    db_repo.save_market_data("SPY", market_data)
    data = db_repo.get_market_data("SPY")
    
    assert len(data) == 1
    assert data[0].symbol == "SPY"
    assert data[0].close_price == 101.0


def test_save_trading_signal(db_repo):
    """Test saving a trading signal."""
    signal_data = {
        'timestamp': datetime(2024, 1, 1, 10, 0),
        'symbol': 'SPY',
        'signal': 'BUY',
        'confidence': 0.85,
        'model_version': 'v1.0',
    }
    
    signal = db_repo.save_signal(signal_data)
    
    assert signal.id is not None
    assert signal.symbol == "SPY"
    assert signal.signal == "BUY"


def test_get_signals(db_repo):
    """Test retrieving trading signals."""
    signals = [
        {
            'timestamp': datetime(2024, 1, 1, 10, 0),
            'symbol': 'SPY',
            'signal': 'BUY',
            'confidence': 0.85,
        },
        {
            'timestamp': datetime(2024, 1, 2, 10, 0),
            'symbol': 'SPY',
            'signal': 'SELL',
            'confidence': 0.75,
        }
    ]
    
    for signal_data in signals:
        db_repo.save_signal(signal_data)
    
    retrieved_signals = db_repo.get_signals(symbol="SPY")
    
    assert len(retrieved_signals) == 2


def test_economic_news_to_dict():
    """Test converting EconomicNews to dictionary."""
    news = EconomicNews(
        title="Test News",
        country="USD",
        volatility="High",
        event_time=datetime(2024, 1, 1, 10, 0),
        forecast="100",
        previous="95",
    )
    
    news_dict = news.to_dict()
    
    assert news_dict['title'] == "Test News"
    assert news_dict['country'] == "USD"
    assert 'event_time' in news_dict

