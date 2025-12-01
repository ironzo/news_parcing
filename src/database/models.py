"""
Database models for storing economic news and trading data.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


class EconomicNews(Base):
    """Model for storing economic news events."""
    
    __tablename__ = 'economic_news'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    country = Column(String(10), nullable=False, index=True)
    volatility = Column(String(100), nullable=True)
    event_time = Column(DateTime, nullable=False, index=True)
    forecast = Column(String(100), nullable=True)
    previous = Column(String(100), nullable=True)
    actual = Column(String(100), nullable=True)  # To be filled after event occurs
    link = Column(Text, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<EconomicNews(id={self.id}, title='{self.title}', event_time='{self.event_time}')>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'country': self.country,
            'volatility': self.volatility,
            'event_time': self.event_time.isoformat() if self.event_time else None,
            'forecast': self.forecast,
            'previous': self.previous,
            'actual': self.actual,
            'link': self.link,
            'scraped_at': self.scraped_at.isoformat() if self.scraped_at else None,
        }


class MarketData(Base):
    """Model for storing market price data."""
    
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close_price})>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume,
        }


class TradingSignal(Base):
    """Model for storing RL agent trading signals."""
    
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal = Column(String(10), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    confidence = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    features = Column(Text, nullable=True)  # JSON string of features used
    executed = Column(Boolean, default=False)
    profit_loss = Column(Float, nullable=True)  # Filled after execution
    
    def __repr__(self):
        return f"<TradingSignal(symbol='{self.symbol}', signal='{self.signal}', timestamp='{self.timestamp}')>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'signal': self.signal,
            'confidence': self.confidence,
            'model_version': self.model_version,
            'executed': self.executed,
            'profit_loss': self.profit_loss,
        }


class BacktestResult(Base):
    """Model for storing backtest results."""
    
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    num_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<BacktestResult(run_id='{self.run_id}', return={self.total_return:.2%})>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'model_version': self.model_version,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

