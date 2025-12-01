"""
Main entry point for the news parsing and trading application.
"""
import argparse
import logging
from datetime import datetime

from src.scrapers.investing_scraper import InvestingScraper
from src.database.repository import DatabaseRepository
from src.utils.config import config
from src.utils.logger import setup_logging


def scrape_news(args):
    """Scrape news from Investing.com and save to database."""
    logger = logging.getLogger(__name__)
    logger.info("Starting news scraping...")
    
    # Initialize scraper
    scraper = InvestingScraper(user_agent=config.scraper.user_agent)
    
    # Scrape news
    news_items = scraper.scrape(filter_usa_high_vol=True)
    
    if not news_items:
        logger.warning("No news items found")
        return
    
    # Print summary
    scraper.print_news_summary(news_items)
    
    # Save to database
    if not args.dry_run:
        db = DatabaseRepository(config.database.url)
        db.create_tables()
        count = db.save_news(news_items)
        logger.info(f"Saved {count} news items to database")
    else:
        logger.info("Dry run - not saving to database")


def list_news(args):
    """List news from database."""
    logger = logging.getLogger(__name__)
    
    db = DatabaseRepository(config.database.url)
    news_items = db.get_news(country="USD", limit=args.limit)
    
    if not news_items:
        logger.info("No news items found in database")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(news_items)} news items in database")
    print(f"{'='*80}\n")
    
    for item in news_items:
        print(f"ID: {item.id}")
        print(f"Title: {item.title}")
        print(f"Time: {item.event_time}")
        print(f"Forecast: {item.forecast}, Previous: {item.previous}")
        print(f"Link: {item.link}")
        print(f"Scraped at: {item.scraped_at}")
        print(f"{'-'*80}\n")


def init_database(args):
    """Initialize database tables."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing database...")
    
    db = DatabaseRepository(config.database.url)
    
    if args.drop:
        logger.warning("Dropping existing tables...")
        db.drop_tables()
    
    db.create_tables()
    logger.info("Database initialized successfully")


def show_config(args):
    """Display current configuration."""
    import json
    print("\nCurrent Configuration:")
    print("="*80)
    print(json.dumps(config.to_dict(), indent=2))
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="News parsing and RL trading application"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape news from Investing.com")
    scrape_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database"
    )
    scrape_parser.set_defaults(func=scrape_news)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List news from database")
    list_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of items to display"
    )
    list_parser.set_defaults(func=list_news)
    
    # Init database command
    init_parser = subparsers.add_parser("init-db", help="Initialize database")
    init_parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating"
    )
    init_parser.set_defaults(func=init_database)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.set_defaults(func=show_config)
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

