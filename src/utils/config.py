"""
Configuration management for the news parsing and trading application.
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///news.db"
    echo: bool = False


@dataclass
class ScraperConfig:
    """Scraper configuration."""
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    timeout: int = 10
    retry_attempts: int = 3
    retry_delay: int = 5


@dataclass
class TradingConfig:
    """Trading and RL configuration."""
    initial_capital: float = 10000.0
    default_symbol: str = "SPY"  # S&P 500 ETF
    max_position_size: float = 0.3  # Max 30% of capital per position
    stop_loss: float = 0.02  # 2% stop loss
    take_profit: float = 0.05  # 5% take profit
    
    # RL specific
    model_save_path: str = "models/"
    checkpoint_frequency: int = 1000
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    batch_size: int = 64


@dataclass
class AppConfig:
    """Main application configuration."""
    project_root: Path
    database: DatabaseConfig
    scraper: ScraperConfig
    trading: TradingConfig
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            AppConfig instance.
        """
        project_root = Path(__file__).parent.parent.parent
        
        # Database configuration
        db_url = os.getenv("DATABASE_URL", "sqlite:///news.db")
        db_echo = os.getenv("DATABASE_ECHO", "false").lower() == "true"
        database = DatabaseConfig(url=db_url, echo=db_echo)
        
        # Scraper configuration
        scraper = ScraperConfig(
            user_agent=os.getenv("SCRAPER_USER_AGENT", ScraperConfig.user_agent),
            timeout=int(os.getenv("SCRAPER_TIMEOUT", "10")),
            retry_attempts=int(os.getenv("SCRAPER_RETRY_ATTEMPTS", "3")),
            retry_delay=int(os.getenv("SCRAPER_RETRY_DELAY", "5")),
        )
        
        # Trading configuration
        trading = TradingConfig(
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "10000.0")),
            default_symbol=os.getenv("DEFAULT_SYMBOL", "SPY"),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.3")),
            stop_loss=float(os.getenv("STOP_LOSS", "0.02")),
            take_profit=float(os.getenv("TAKE_PROFIT", "0.05")),
            model_save_path=os.getenv("MODEL_SAVE_PATH", "models/"),
            learning_rate=float(os.getenv("LEARNING_RATE", "0.0003")),
        )
        
        log_level = os.getenv("LOG_LEVEL", "INFO")
        
        return cls(
            project_root=project_root,
            database=database,
            scraper=scraper,
            trading=trading,
            log_level=log_level,
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "project_root": str(self.project_root),
            "database": {
                "url": self.database.url,
                "echo": self.database.echo,
            },
            "scraper": {
                "timeout": self.scraper.timeout,
                "retry_attempts": self.scraper.retry_attempts,
            },
            "trading": {
                "initial_capital": self.trading.initial_capital,
                "default_symbol": self.trading.default_symbol,
                "max_position_size": self.trading.max_position_size,
            },
            "log_level": self.log_level,
        }


# Global configuration instance
config = AppConfig.from_env()

