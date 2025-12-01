"""
Database repository for managing data persistence.
"""
import logging
from typing import List, Optional, Dict
from datetime import datetime

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import Base, EconomicNews, MarketData, TradingSignal, BacktestResult


logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Repository for database operations."""
    
    def __init__(self, database_url: str = "sqlite:///news.db"):
        """
        Initialize database connection.
        
        Args:
            database_url: SQLAlchemy database URL.
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.database_url = database_url
        logger.info(f"Initialized database repository: {database_url}")
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all tables from the database."""
        Base.metadata.drop_all(self.engine)
        logger.warning("All database tables dropped")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # Economic News Operations
    
    def save_news(self, news_items: List[Dict]) -> int:
        """
        Save economic news items to database.
        
        Args:
            news_items: List of news item dictionaries.
            
        Returns:
            Number of items saved.
        """
        session = self.get_session()
        count = 0
        
        try:
            for item in news_items:
                # Parse event time
                event_time = None
                if item.get('time'):
                    try:
                        event_time = datetime.strptime(item['time'], '%Y/%m/%d %H:%M:%S')
                    except ValueError:
                        logger.warning(f"Failed to parse time: {item.get('time')}")
                
                news = EconomicNews(
                    title=item.get('title'),
                    country=item.get('country', 'USD'),
                    volatility=item.get('volatility'),
                    event_time=event_time,
                    forecast=item.get('forecast'),
                    previous=item.get('previous'),
                    link=item.get('link'),
                )
                session.add(news)
                count += 1
            
            session.commit()
            logger.info(f"Saved {count} news items to database")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save news items: {e}")
            raise
        finally:
            session.close()
        
        return count
    
    def get_news(
        self, 
        country: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[EconomicNews]:
        """
        Retrieve economic news from database.
        
        Args:
            country: Filter by country code.
            start_date: Filter by start date.
            end_date: Filter by end date.
            limit: Maximum number of records to return.
            
        Returns:
            List of EconomicNews objects.
        """
        session = self.get_session()
        
        try:
            query = session.query(EconomicNews)
            
            if country:
                query = query.filter(EconomicNews.country == country)
            if start_date:
                query = query.filter(EconomicNews.event_time >= start_date)
            if end_date:
                query = query.filter(EconomicNews.event_time <= end_date)
            
            query = query.order_by(desc(EconomicNews.event_time))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        finally:
            session.close()
    
    # Market Data Operations
    
    def save_market_data(self, symbol: str, data: List[Dict]) -> int:
        """
        Save market data to database.
        
        Args:
            symbol: Trading symbol.
            data: List of OHLCV dictionaries.
            
        Returns:
            Number of records saved.
        """
        session = self.get_session()
        count = 0
        
        try:
            for record in data:
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=record['timestamp'],
                    open_price=record['open'],
                    high_price=record['high'],
                    low_price=record['low'],
                    close_price=record['close'],
                    volume=record.get('volume'),
                )
                session.add(market_data)
                count += 1
            
            session.commit()
            logger.info(f"Saved {count} market data records for {symbol}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save market data: {e}")
            raise
        finally:
            session.close()
        
        return count
    
    def get_market_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[MarketData]:
        """
        Retrieve market data from database.
        
        Args:
            symbol: Trading symbol.
            start_date: Filter by start date.
            end_date: Filter by end date.
            
        Returns:
            List of MarketData objects.
        """
        session = self.get_session()
        
        try:
            query = session.query(MarketData).filter(MarketData.symbol == symbol)
            
            if start_date:
                query = query.filter(MarketData.timestamp >= start_date)
            if end_date:
                query = query.filter(MarketData.timestamp <= end_date)
            
            query = query.order_by(MarketData.timestamp)
            return query.all()
        finally:
            session.close()
    
    # Trading Signal Operations
    
    def save_signal(self, signal_data: Dict) -> TradingSignal:
        """
        Save a trading signal to database.
        
        Args:
            signal_data: Dictionary with signal information.
            
        Returns:
            Created TradingSignal object.
        """
        session = self.get_session()
        
        try:
            signal = TradingSignal(**signal_data)
            session.add(signal)
            session.commit()
            session.refresh(signal)
            logger.info(f"Saved trading signal: {signal.symbol} {signal.signal}")
            return signal
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save signal: {e}")
            raise
        finally:
            session.close()
    
    def get_signals(
        self,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[TradingSignal]:
        """
        Retrieve trading signals from database.
        
        Args:
            symbol: Filter by symbol.
            executed: Filter by execution status.
            limit: Maximum number of records.
            
        Returns:
            List of TradingSignal objects.
        """
        session = self.get_session()
        
        try:
            query = session.query(TradingSignal)
            
            if symbol:
                query = query.filter(TradingSignal.symbol == symbol)
            if executed is not None:
                query = query.filter(TradingSignal.executed == executed)
            
            query = query.order_by(desc(TradingSignal.timestamp))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        finally:
            session.close()
    
    # Backtest Operations
    
    def save_backtest_result(self, result_data: Dict) -> BacktestResult:
        """
        Save backtest results to database.
        
        Args:
            result_data: Dictionary with backtest results.
            
        Returns:
            Created BacktestResult object.
        """
        session = self.get_session()
        
        try:
            result = BacktestResult(**result_data)
            session.add(result)
            session.commit()
            session.refresh(result)
            logger.info(f"Saved backtest result: {result.run_id}")
            return result
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save backtest result: {e}")
            raise
        finally:
            session.close()
    
    def get_backtest_results(self, limit: Optional[int] = 10) -> List[BacktestResult]:
        """
        Retrieve backtest results.
        
        Args:
            limit: Maximum number of results to return.
            
        Returns:
            List of BacktestResult objects.
        """
        session = self.get_session()
        
        try:
            query = session.query(BacktestResult).order_by(desc(BacktestResult.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        finally:
            session.close()

