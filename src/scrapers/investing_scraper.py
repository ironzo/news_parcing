"""
Scraper for Investing.com economic calendar data.
Focuses on high-volatility USA news events.
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


class InvestingScraper:
    """Scraper for Investing.com economic calendar."""
    
    BASE_URL = "https://www.investing.com"
    CALENDAR_URL = f"{BASE_URL}/economic-calendar/"
    
    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize the scraper.
        
        Args:
            user_agent: Custom user agent string. If None, uses a default.
        """
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        self.headers = {
            "Accept": "*/*",
            "User-Agent": self.user_agent,
        }
    
    def fetch_page(self) -> Optional[BeautifulSoup]:
        """
        Fetch the economic calendar page.
        
        Returns:
            BeautifulSoup object if successful, None otherwise.
        """
        try:
            response = requests.get(self.CALENDAR_URL, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully fetched page. Status code: {response.status_code}")
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Failed to fetch page: {e}")
            return None
    
    def parse_news_item(self, item) -> Optional[Dict]:
        """
        Parse a single news item from the HTML.
        
        Args:
            item: BeautifulSoup element containing news item.
            
        Returns:
            Dictionary with news data or None if parsing fails.
        """
        try:
            # Extract title
            title_element = item.find(class_="left event")
            if not title_element:
                return None
            title = title_element.text.strip()
            
            # Extract country
            country_element = item.find(class_="left flagCur noWrap")
            country = country_element.get_text(strip=True) if country_element else ''
            
            # Extract volatility
            volatility_element = item.find(class_="left textNum sentiment noWrap")
            volatility = volatility_element.get("title") if volatility_element else None
            
            # Extract time
            time = item.get("data-event-datetime")
            
            # Extract link
            link_element = item.find("a")
            link = f"{self.BASE_URL}{link_element.get('href')}" if link_element else None
            
            # Extract previous value
            previous_element = item.select_one("[class^='prev']")
            previous = previous_element.get_text(strip=True) if previous_element else 'N/A'
            
            # Extract forecast value
            forecast_element = item.select_one("[class^='fore']")
            forecast = forecast_element.get_text(strip=True) if forecast_element else 'N/A'
            
            return {
                'title': title,
                'country': country,
                'volatility': volatility,
                'time': time,
                'forecast': forecast,
                'previous': previous,
                'link': link,
            }
        except Exception as e:
            logger.warning(f"Failed to parse news item: {e}")
            return None
    
    def filter_high_volatility_usa(self, news_items: List[Dict]) -> List[Dict]:
        """
        Filter news items for USA high-volatility news only.
        
        Args:
            news_items: List of parsed news items.
            
        Returns:
            Filtered list of news items.
        """
        filtered = [
            item for item in news_items
            if item['country'] == 'USD' and 
            item['volatility'] == 'High Volatility Expected'
        ]
        logger.info(f"Filtered {len(filtered)} high-volatility USA news items from {len(news_items)} total")
        return filtered
    
    def scrape(self, filter_usa_high_vol: bool = True) -> List[Dict]:
        """
        Scrape economic calendar and return news items.
        
        Args:
            filter_usa_high_vol: If True, only return USA high-volatility news.
            
        Returns:
            List of news item dictionaries.
        """
        soup = self.fetch_page()
        if not soup:
            logger.error("Failed to fetch page")
            return []
        
        news_items = []
        for item in soup.find_all(class_="js-event-item"):
            parsed_item = self.parse_news_item(item)
            if parsed_item:
                news_items.append(parsed_item)
        
        logger.info(f"Scraped {len(news_items)} total news items")
        
        if filter_usa_high_vol:
            news_items = self.filter_high_volatility_usa(news_items)
        
        return news_items
    
    def print_news_summary(self, news_items: List[Dict]) -> None:
        """
        Print a formatted summary of news items.
        
        Args:
            news_items: List of news item dictionaries.
        """
        print(f"\n{'='*80}")
        print(f"Found {len(news_items)} high-volatility USA news items")
        print(f"{'='*80}\n")
        
        for item in news_items:
            print(f"Title: {item['title']}")
            print(f"Time: {item['time']}")
            print(f"Forecast: {item['forecast']}, Previous: {item['previous']}")
            print(f"Link: {item['link']}")
            print(f"{'-'*80}\n")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the scraper
    scraper = InvestingScraper()
    news = scraper.scrape()
    scraper.print_news_summary(news)

