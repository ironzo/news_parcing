"""
Main entry point for the news parsing and trading application.
"""
import argparse
import logging
from datetime import datetime, timedelta

from src.scrapers.investing_scraper import InvestingScraper
from src.database.repository import DatabaseRepository
from src.analysis.market_data import MarketDataFetcher, sync_with_news_events, calculate_price_impact
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


def fetch_market_data(args):
    """Fetch historical market data."""
    logger = logging.getLogger(__name__)
    logger.info(f"Fetching market data for {args.symbol}...")
    
    # Initialize repository and fetcher
    db = DatabaseRepository(config.database.url)
    db.create_tables()
    fetcher = MarketDataFetcher(db)
    
    # Parse dates
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Fetch data
    try:
        if args.symbol == 'all':
            symbols = fetcher.supported_symbols
            results = fetcher.fetch_multiple_symbols(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                save_to_db=not args.dry_run
            )
            
            for symbol, df in results.items():
                print(f"\n{symbol}: Fetched {len(df)} records")
                if not df.empty:
                    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        else:
            df = fetcher.fetch_historical_data(
                symbol=args.symbol,
                start_date=start_date,
                end_date=end_date,
                save_to_db=not args.dry_run
            )
            
            print(f"\nFetched {len(df)} records for {args.symbol}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            if args.preview:
                print("\nFirst 5 records:")
                print(df.head())
                print("\nLast 5 records:")
                print(df.tail())
        
        if args.dry_run:
            logger.info("Dry run - not saving to database")
        else:
            logger.info("Data saved to database successfully")
            
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return


def list_market_data(args):
    """List market data from database."""
    logger = logging.getLogger(__name__)
    
    db = DatabaseRepository(config.database.url)
    fetcher = MarketDataFetcher(db)
    
    # Get cached data
    df = fetcher.get_cached_data(args.symbol)
    
    if df.empty:
        logger.info(f"No market data found for {args.symbol}")
        return
    
    print(f"\n{'='*80}")
    print(f"Market data for {args.symbol}: {len(df)} records")
    print(f"{'='*80}\n")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Average volume: {df['volume'].mean():,.0f}")
    
    if args.preview:
        print("\nFirst 10 records:")
        print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(10))


def analyze_news_impact(args):
    """Analyze price impact of news events."""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing news impact on market prices...")
    
    # Initialize repository and fetcher
    db = DatabaseRepository(config.database.url)
    fetcher = MarketDataFetcher(db)
    
    # Get news events
    news_items = db.get_news(country="USD", limit=args.limit)
    
    if not news_items:
        logger.warning("No news items found in database")
        return
    
    # Get market data
    df = fetcher.get_cached_data(args.symbol)
    
    if df.empty:
        logger.warning(f"No market data found for {args.symbol}")
        return
    
    print(f"\n{'='*80}")
    print(f"Analyzing price impact for {len(news_items)} news events")
    print(f"{'='*80}\n")
    
    # Analyze each news event
    for news in news_items:
        impact = calculate_price_impact(news, df, window=args.window)
        
        if impact:
            print(f"Event: {news.title}")
            print(f"Time: {news.event_time}")
            print(f"Pre-event price: ${impact.get('pre_event_price', 0):.2f}")
            print(f"Post-event price: ${impact.get('post_event_price', 0):.2f}")
            print(f"Change: {impact.get('percentage_change', 0):.2f}%")
            print(f"Volatility: {impact.get('volatility', 0):.4f}")
            if 'max_excursion_pct' in impact:
                print(f"Max excursion: {impact['max_excursion_pct']:.2f}%")
            print(f"{'-'*80}\n")
    
    # Create synchronized dataset
    if args.save_sync:
        synced_df = sync_with_news_events(news_items, df)
        if not synced_df.empty:
            output_file = f"synced_data_{args.symbol}.csv"
            synced_df.to_csv(output_file, index=False)
            logger.info(f"Synchronized data saved to {output_file}")
            print(f"\nSynchronized dataset saved to: {output_file}")
            print(f"Total records: {len(synced_df)}")


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
    
    # Fetch market data command
    fetch_parser = subparsers.add_parser("fetch-market", help="Fetch historical market data")
    fetch_parser.add_argument(
        "symbol",
        help="Trading symbol (e.g., SPY, QQQ) or 'all' for all supported symbols"
    )
    fetch_parser.add_argument(
        "--days",
        type=int,
        help="Number of days of history to fetch (alternative to start/end dates)"
    )
    fetch_parser.add_argument(
        "--start-date",
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help="Start date (YYYY-MM-DD)"
    )
    fetch_parser.add_argument(
        "--end-date",
        default=datetime.now().strftime('%Y-%m-%d'),
        help="End date (YYYY-MM-DD)"
    )
    fetch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database"
    )
    fetch_parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview of fetched data"
    )
    fetch_parser.set_defaults(func=fetch_market_data)
    
    # List market data command
    list_market_parser = subparsers.add_parser("list-market", help="List market data from database")
    list_market_parser.add_argument(
        "symbol",
        help="Trading symbol (e.g., SPY, QQQ)"
    )
    list_market_parser.add_argument(
        "--preview",
        action="store_true",
        help="Show data preview"
    )
    list_market_parser.set_defaults(func=list_market_data)
    
    # Analyze news impact command
    analyze_parser = subparsers.add_parser("analyze-impact", help="Analyze news impact on prices")
    analyze_parser.add_argument(
        "symbol",
        help="Trading symbol (e.g., SPY, QQQ)"
    )
    analyze_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of news events to analyze"
    )
    analyze_parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Time window in minutes before/after event"
    )
    analyze_parser.add_argument(
        "--save-sync",
        action="store_true",
        help="Save synchronized dataset to CSV"
    )
    analyze_parser.set_defaults(func=analyze_news_impact)
    
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

