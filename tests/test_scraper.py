"""
Tests for the Investing.com scraper.
"""
import pytest
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup

from src.scrapers.investing_scraper import InvestingScraper


@pytest.fixture
def scraper():
    """Create a scraper instance for testing."""
    return InvestingScraper()


@pytest.fixture
def sample_html():
    """Sample HTML for testing."""
    return """
    <div class="js-event-item" data-event-datetime="2024/02/01 08:30:00">
        <td class="left event">Initial Jobless Claims</td>
        <td class="left flagCur noWrap">USD</td>
        <td class="left textNum sentiment noWrap" title="High Volatility Expected"></td>
        <td class="fore">213K</td>
        <td class="prev">214K</td>
        <a href="/economic-calendar/initial-jobless-claims-294"></a>
    </div>
    """


def test_scraper_initialization():
    """Test scraper initialization with default values."""
    scraper = InvestingScraper()
    assert scraper.user_agent is not None
    assert "Mozilla" in scraper.user_agent
    assert scraper.headers["User-Agent"] == scraper.user_agent


def test_scraper_custom_user_agent():
    """Test scraper initialization with custom user agent."""
    custom_ua = "TestBot/1.0"
    scraper = InvestingScraper(user_agent=custom_ua)
    assert scraper.user_agent == custom_ua


def test_parse_news_item(scraper, sample_html):
    """Test parsing a single news item."""
    soup = BeautifulSoup(sample_html, 'html.parser')
    item = soup.find(class_="js-event-item")
    
    result = scraper.parse_news_item(item)
    
    assert result is not None
    assert result['title'] == "Initial Jobless Claims"
    assert result['country'] == "USD"
    assert result['volatility'] == "High Volatility Expected"
    assert result['time'] == "2024/02/01 08:30:00"
    assert result['forecast'] == "213K"
    assert result['previous'] == "214K"
    assert "initial-jobless-claims-294" in result['link']


def test_filter_high_volatility_usa(scraper):
    """Test filtering for USA high-volatility news."""
    news_items = [
        {'country': 'USD', 'volatility': 'High Volatility Expected', 'title': 'News 1'},
        {'country': 'USD', 'volatility': 'Low Volatility Expected', 'title': 'News 2'},
        {'country': 'EUR', 'volatility': 'High Volatility Expected', 'title': 'News 3'},
        {'country': 'USD', 'volatility': 'High Volatility Expected', 'title': 'News 4'},
    ]
    
    filtered = scraper.filter_high_volatility_usa(news_items)
    
    assert len(filtered) == 2
    assert all(item['country'] == 'USD' for item in filtered)
    assert all(item['volatility'] == 'High Volatility Expected' for item in filtered)


@patch('src.scrapers.investing_scraper.requests.get')
def test_fetch_page_success(mock_get, scraper):
    """Test successful page fetch."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Test</body></html>"
    mock_get.return_value = mock_response
    
    result = scraper.fetch_page()
    
    assert result is not None
    assert isinstance(result, BeautifulSoup)
    mock_get.assert_called_once()


@patch('src.scrapers.investing_scraper.requests.get')
def test_fetch_page_failure(mock_get, scraper):
    """Test failed page fetch."""
    import requests
    mock_get.side_effect = requests.RequestException("Connection error")
    
    result = scraper.fetch_page()
    
    assert result is None


def test_parse_news_item_missing_fields(scraper):
    """Test parsing news item with missing fields."""
    html = '<div class="js-event-item"></div>'
    soup = BeautifulSoup(html, 'html.parser')
    item = soup.find(class_="js-event-item")
    
    result = scraper.parse_news_item(item)
    
    # Should return None when title is missing
    assert result is None

