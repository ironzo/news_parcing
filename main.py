"""
Main entry point for the news parsing and trading application.
"""
import argparse
import logging
import os
from datetime import datetime, timedelta

from src.scrapers.investing_scraper import InvestingScraper
from src.database.repository import DatabaseRepository
from src.analysis.market_data import MarketDataFetcher, sync_with_news_events, calculate_price_impact
from src.analysis.trading_env import TradingConfig, create_trading_env, Action
from src.analysis.features import FeatureExtractor
from src.analysis.agent import TradingAgent, AgentConfig
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


def train_agent(args):
    """Train the RL trading agent."""
    logger = logging.getLogger(__name__)
    logger.info("Starting agent training...")
    
    # Initialize repository and fetcher
    db = DatabaseRepository(config.database.url)
    fetcher = MarketDataFetcher(db)
    
    # Get market data
    df = fetcher.get_cached_data(args.symbol)
    
    if df.empty:
        logger.error(f"No market data found for {args.symbol}. Run 'fetch-market' first.")
        return
    
    if len(df) < 400:
        logger.error(f"Insufficient data ({len(df)} rows). Need at least 400 rows for training.")
        return
    
    print(f"\n{'='*80}")
    print(f"Training RL Agent on {args.symbol}")
    print(f"{'='*80}")
    print(f"Data points: {len(df)}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Model directory: {args.model_dir}")
    print(f"{'='*80}\n")
    
    # Configure agent
    agent_config = AgentConfig(
        learning_rate=args.learning_rate,
        total_timesteps=args.timesteps,
        model_dir=args.model_dir,
        log_dir=os.path.join(args.model_dir, "logs"),
        tensorboard_log=os.path.join(args.model_dir, "tensorboard"),
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        verbose=1 if args.verbose else 0
    )
    
    trading_config = TradingConfig(
        initial_capital=args.capital,
        episode_length=args.episode_length
    )
    
    # Create and train agent
    agent = TradingAgent(agent_config)
    agent.prepare_data(df, trading_config=trading_config)
    agent.create_model()
    
    print("Training started... (use Ctrl+C to stop early)")
    print(f"TensorBoard: tensorboard --logdir {agent_config.tensorboard_log}\n")
    
    try:
        agent.train(progress_bar=True)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Evaluate
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")
    
    metrics = agent.evaluate(n_episodes=args.eval_episodes)
    
    print(f"Results:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Mean Return: {metrics['mean_return']*100:.2f}%")
    print(f"  Max Drawdown: {metrics['mean_max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics['mean_win_rate']*100:.1f}%")
    print(f"  Avg Trades: {metrics['mean_trades']:.1f}")
    
    print(f"\nModel saved to: {args.model_dir}/")


def evaluate_agent(args):
    """Evaluate a trained agent."""
    logger = logging.getLogger(__name__)
    
    model_path = args.model_path
    if not model_path.endswith(".zip"):
        model_path = model_path + ".zip"
    
    if not os.path.exists(model_path):
        # Try common paths
        for path in [args.model_path, f"models/{args.model_path}", f"models/{args.model_path}.zip"]:
            if os.path.exists(path):
                model_path = path
                break
        else:
            logger.error(f"Model not found: {args.model_path}")
            return
    
    # Get market data
    db = DatabaseRepository(config.database.url)
    fetcher = MarketDataFetcher(db)
    df = fetcher.get_cached_data(args.symbol)
    
    if df.empty:
        logger.error(f"No market data found for {args.symbol}")
        return
    
    print(f"\n{'='*80}")
    print(f"Evaluating Agent: {model_path}")
    print(f"Symbol: {args.symbol}")
    print(f"{'='*80}\n")
    
    # Create environment and load model
    trading_config = TradingConfig(
        initial_capital=args.capital,
        episode_length=min(args.episode_length, len(df) - 60)
    )
    
    env = create_trading_env(
        market_data=df,
        initial_capital=trading_config.initial_capital,
        episode_length=trading_config.episode_length,
        random_start=False
    )
    
    agent = TradingAgent()
    agent.test_env = env
    agent.load(model_path.replace(".zip", ""), env=env)
    
    # Evaluate
    metrics = agent.evaluate(env=env, n_episodes=args.episodes)
    
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Mean Return: {metrics['mean_return']*100:.2f}%")
    print(f"  Max Drawdown: {metrics['mean_max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics['mean_win_rate']*100:.1f}%")
    print(f"  Avg Trades: {metrics['mean_trades']:.1f}")


def demo_trading(args):
    """Interactive demo of the trading environment."""
    logger = logging.getLogger(__name__)
    
    # Get market data
    db = DatabaseRepository(config.database.url)
    fetcher = MarketDataFetcher(db)
    df = fetcher.get_cached_data(args.symbol)
    
    if df.empty:
        logger.error(f"No market data found for {args.symbol}. Run 'fetch-market' first.")
        return
    
    print(f"\n{'='*80}")
    print(f"Trading Environment Demo - {args.symbol}")
    print(f"{'='*80}")
    print(f"Data points: {len(df)}")
    print(f"Initial capital: ${args.capital:,.2f}")
    print(f"\nActions: 0=BUY, 1=HOLD, 2=SELL, q=quit")
    print(f"{'='*80}\n")
    
    # Create environment
    env = create_trading_env(
        market_data=df,
        initial_capital=args.capital,
        episode_length=min(50, len(df) - 60),
        random_start=True,
        render_mode='ansi'
    )
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Feature extractor for display
    feature_names = FeatureExtractor().get_feature_names()
    
    while not done:
        # Display state
        print(env.render())
        print(f"Total Reward: {total_reward:.4f}")
        
        if args.show_features:
            print("\nKey Features:")
            print(f"  Position: {obs[20]:.1f}")
            print(f"  Unrealized PnL: {obs[21]:.3f}")
            print(f"  RSI: {obs[7]*50+50:.1f}")
            print(f"  Volatility: {obs[4]:.3f}")
        
        # Get action
        try:
            action_input = input("\nAction (0/1/2 or q): ").strip()
            
            if action_input.lower() == 'q':
                break
            
            action = int(action_input)
            if action not in [0, 1, 2]:
                print("Invalid action. Use 0 (BUY), 1 (HOLD), or 2 (SELL)")
                continue
                
        except ValueError:
            print("Invalid input. Enter 0, 1, 2, or q")
            continue
        except EOFError:
            break
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        action_name = ['BUY', 'HOLD', 'SELL'][action]
        print(f"\n>>> {action_name} | Reward: {reward:.4f}")
        print("-" * 40)
    
    # Final stats
    stats = env.get_episode_stats()
    print(f"\n{'='*80}")
    print("Episode Complete!")
    print(f"{'='*80}")
    print(f"Total Return: {stats.get('total_return', 0)*100:.2f}%")
    print(f"Max Drawdown: {stats.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
    print(f"Total Trades: {stats.get('num_trades', 0)}")
    print(f"Win Rate: {stats.get('win_rate', 0)*100:.1f}%")


def run_backtest(args):
    """Run backtest with trained agent."""
    logger = logging.getLogger(__name__)
    
    model_path = args.model_path
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        for path in [f"models/{model_path}", f"models/{model_path}.zip", f"models/best_model", f"models/final_model"]:
            if os.path.exists(path) or os.path.exists(path + ".zip"):
                model_path = path
                break
        else:
            logger.error(f"Model not found: {args.model_path}")
            return
    
    # Get market data
    db = DatabaseRepository(config.database.url)
    fetcher = MarketDataFetcher(db)
    df = fetcher.get_cached_data(args.symbol)
    
    if df.empty:
        logger.error(f"No market data found for {args.symbol}")
        return
    
    print(f"\n{'='*80}")
    print(f"Backtest: {model_path}")
    print(f"Symbol: {args.symbol}")
    print(f"{'='*80}\n")
    
    # Create environment
    env = create_trading_env(
        market_data=df,
        initial_capital=args.capital,
        episode_length=min(args.days, len(df) - 60),
        random_start=False,
        render_mode='ansi' if args.verbose else None
    )
    
    # Load model
    agent = TradingAgent()
    agent.load(model_path.replace(".zip", ""), env=env)
    
    # Run backtest
    obs, _ = env.reset()
    done = False
    actions_taken = []
    
    print("Running backtest...")
    
    while not done:
        action, _ = agent.model.predict(obs, deterministic=True)
        action = int(action)
        actions_taken.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if args.verbose:
            print(env.render())
    
    # Results
    stats = env.get_episode_stats()
    
    print(f"\n{'='*80}")
    print("Backtest Results")
    print(f"{'='*80}")
    print(f"Period: {args.days} trading days")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Final Value: ${env.portfolio.portfolio_value:,.2f}")
    print(f"\nPerformance:")
    print(f"  Total Return: {stats.get('total_return', 0)*100:.2f}%")
    print(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: {stats.get('max_drawdown', 0)*100:.2f}%")
    print(f"\nTrading:")
    print(f"  Total Trades: {stats.get('num_trades', 0)}")
    print(f"  Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
    print(f"  Total P&L: ${stats.get('total_pnl', 0):,.2f}")
    
    # Action distribution
    dist = stats.get('action_distribution', {})
    print(f"\nAction Distribution:")
    print(f"  BUY:  {dist.get('BUY', 0)} ({dist.get('BUY', 0)/max(len(actions_taken),1)*100:.1f}%)")
    print(f"  HOLD: {dist.get('HOLD', 0)} ({dist.get('HOLD', 0)/max(len(actions_taken),1)*100:.1f}%)")
    print(f"  SELL: {dist.get('SELL', 0)} ({dist.get('SELL', 0)/max(len(actions_taken),1)*100:.1f}%)")


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
    
    # Train agent command
    train_parser = subparsers.add_parser("train", help="Train RL trading agent")
    train_parser.add_argument(
        "symbol",
        help="Trading symbol to train on (e.g., SPY)"
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 0.0003)"
    )
    train_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial trading capital (default: 10000)"
    )
    train_parser.add_argument(
        "--episode-length",
        type=int,
        default=252,
        help="Episode length in trading days (default: 252)"
    )
    train_parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save models (default: models)"
    )
    train_parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency in timesteps (default: 10000)"
    )
    train_parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25000,
        help="Checkpoint frequency in timesteps (default: 25000)"
    )
    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)"
    )
    train_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    train_parser.set_defaults(func=train_agent)
    
    # Evaluate agent command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument(
        "model_path",
        help="Path to trained model (e.g., models/best_model)"
    )
    eval_parser.add_argument(
        "symbol",
        help="Trading symbol to evaluate on"
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    eval_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial trading capital (default: 10000)"
    )
    eval_parser.add_argument(
        "--episode-length",
        type=int,
        default=252,
        help="Episode length in trading days (default: 252)"
    )
    eval_parser.set_defaults(func=evaluate_agent)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Interactive trading demo")
    demo_parser.add_argument(
        "symbol",
        help="Trading symbol (e.g., SPY)"
    )
    demo_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial trading capital (default: 10000)"
    )
    demo_parser.add_argument(
        "--show-features",
        action="store_true",
        help="Show key features during trading"
    )
    demo_parser.set_defaults(func=demo_trading)
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest with trained agent")
    backtest_parser.add_argument(
        "model_path",
        help="Path to trained model"
    )
    backtest_parser.add_argument(
        "symbol",
        help="Trading symbol to backtest on"
    )
    backtest_parser.add_argument(
        "--days",
        type=int,
        default=252,
        help="Number of trading days to backtest (default: 252)"
    )
    backtest_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial trading capital (default: 10000)"
    )
    backtest_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show step-by-step output"
    )
    backtest_parser.set_defaults(func=run_backtest)
    
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

