"""
Analysis and RL modules for the trading system.

This package contains:
- Market data integration (market_data.py)
- Feature engineering (features.py)
- Trading environment (trading_env.py)
- RL Agent training (agent.py)
"""

from .market_data import (
    MarketDataFetcher,
    sync_with_news_events,
    calculate_price_impact,
    calculate_returns,
    calculate_volatility,
)

from .features import (
    FeatureExtractor,
    FeatureConfig,
    NEWS_FEATURE_DIM,
    MARKET_FEATURE_DIM,
    POSITION_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
)

from .trading_env import (
    TradingEnvironment,
    TradingConfig,
    PortfolioState,
    Action,
    create_trading_env,
)

from .agent import (
    TradingAgent,
    AgentConfig,
    train_trading_agent,
)


__all__ = [
    # Market data
    'MarketDataFetcher',
    'sync_with_news_events',
    'calculate_price_impact',
    'calculate_returns',
    'calculate_volatility',
    # Features
    'FeatureExtractor',
    'FeatureConfig',
    'NEWS_FEATURE_DIM',
    'MARKET_FEATURE_DIM',
    'POSITION_FEATURE_DIM',
    'TOTAL_FEATURE_DIM',
    # Trading environment
    'TradingEnvironment',
    'TradingConfig',
    'PortfolioState',
    'Action',
    'create_trading_env',
    # Agent
    'TradingAgent',
    'AgentConfig',
    'train_trading_agent',
]
