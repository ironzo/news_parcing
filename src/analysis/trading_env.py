"""
Custom Gymnasium trading environment for RL training.

This module implements a trading environment that:
- Follows the Gymnasium Env interface
- Uses 25-dimensional observation space
- Supports discrete actions: BUY, HOLD, SELL
- Implements risk-adjusted reward function
- Tracks portfolio performance metrics
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Dict, Optional, Tuple, List, Any, SupportsFloat
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .features import FeatureExtractor, FeatureConfig, TOTAL_FEATURE_DIM


logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Trading actions."""
    BUY = 0
    HOLD = 1
    SELL = 2


@dataclass
class TradingConfig:
    """Configuration for trading environment."""
    # Initial capital
    initial_capital: float = 10000.0
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    
    # Position sizing
    max_position_size: float = 1.0  # Maximum position as fraction of portfolio
    
    # Risk management
    max_drawdown_threshold: float = 0.20  # 20% max drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    
    # Reward parameters
    risk_penalty_lambda: float = 1.0  # Penalty weight for drawdown
    holding_penalty: float = 0.0001  # Small penalty for holding to encourage action
    
    # Episode settings
    episode_length: int = 252  # 1 trading year
    random_start: bool = True  # Random starting point in data
    
    # Feature configuration
    feature_config: Optional[FeatureConfig] = None


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    cash: float
    position: float  # -1 (short), 0 (flat), 1 (long)
    position_size: float  # Dollar value of position
    entry_price: float
    holding_periods: int
    
    # Performance tracking
    portfolio_value: float
    peak_value: float
    total_return: float
    current_drawdown: float
    max_drawdown: float
    
    # Trade history
    num_trades: int
    winning_trades: int
    total_pnl: float
    
    def reset(self, initial_capital: float) -> None:
        """Reset portfolio to initial state."""
        self.cash = initial_capital
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.holding_periods = 0
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.total_return = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.num_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0


class TradingEnvironment(gym.Env):
    """
    Custom trading environment following Gymnasium interface.
    
    Observation Space: Box(25,) - continuous feature vector
    Action Space: Discrete(3) - BUY (0), HOLD (1), SELL (2)
    
    The environment simulates trading on historical market data,
    with the agent making decisions at each time step.
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        market_data: pd.DataFrame,
        news_events: Optional[List] = None,
        config: Optional[TradingConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize trading environment.
        
        Args:
            market_data: DataFrame with OHLCV data.
            news_events: Optional list of economic news events.
            config: Trading configuration.
            render_mode: Rendering mode ('human' or 'ansi').
        """
        super().__init__()
        
        self.config = config or TradingConfig()
        self.market_data = market_data.copy().reset_index(drop=True)
        self.news_events = news_events or []
        self.render_mode = render_mode
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.config.feature_config)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # BUY, HOLD, SELL
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TOTAL_FEATURE_DIM,),
            dtype=np.float32
        )
        
        # Initialize portfolio state
        self.portfolio = PortfolioState(
            cash=self.config.initial_capital,
            position=0.0,
            position_size=0.0,
            entry_price=0.0,
            holding_periods=0,
            portfolio_value=self.config.initial_capital,
            peak_value=self.config.initial_capital,
            total_return=0.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            num_trades=0,
            winning_trades=0,
            total_pnl=0.0
        )
        
        # Episode tracking
        self.current_step = 0
        self.start_step = 0
        self.end_step = len(self.market_data) - 1
        self.episode_rewards: List[float] = []
        self.episode_actions: List[int] = []
        self.episode_prices: List[float] = []
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Initialized TradingEnvironment with {len(market_data)} data points")
    
    def _validate_data(self) -> None:
        """Validate market data."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in self.market_data.columns]
        
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")
        
        if len(self.market_data) < self.config.episode_length + 50:
            logger.warning(
                f"Market data ({len(self.market_data)} rows) may be insufficient for "
                f"episode length ({self.config.episode_length})"
            )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (e.g., 'start_step' to specify starting point).
            
        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        
        # Reset portfolio
        self.portfolio.reset(self.config.initial_capital)
        
        # Determine starting point
        min_start = 50  # Need some data for feature calculation
        max_start = len(self.market_data) - self.config.episode_length
        
        if options and 'start_step' in options:
            self.start_step = max(min_start, min(options['start_step'], max_start))
        elif self.config.random_start and max_start > min_start:
            self.start_step = self.np_random.integers(min_start, max_start)
        else:
            self.start_step = min_start
        
        self.current_step = self.start_step
        self.end_step = min(self.start_step + self.config.episode_length, len(self.market_data) - 1)
        
        # Reset episode tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_prices = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Reset environment: step {self.current_step} to {self.end_step}")
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Action to take (0=BUY, 1=HOLD, 2=SELL).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Get current and next prices
        current_price = self._get_current_price()
        
        # Execute action
        trade_executed = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Get new price for P&L calculation
        new_price = self._get_current_price()
        
        # Update portfolio value
        self._update_portfolio_value(new_price)
        
        # Calculate reward
        reward = self._calculate_reward(trade_executed)
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.current_step >= self.end_step
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Track episode data
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_prices.append(new_price)
        
        # Update holding period if in position
        if self.portfolio.position != 0:
            self.portfolio.holding_periods += 1
        
        return observation, reward, terminated, truncated, info
    
    def _get_current_price(self) -> float:
        """Get current closing price."""
        return float(self.market_data.iloc[self.current_step]['close'])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (feature vector)."""
        # Get market data up to current step
        window_data = self.market_data.iloc[:self.current_step + 1].copy()
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            market_data=window_data,
            news_events=self.news_events,
            current_position=self.portfolio.position,
            entry_price=self.portfolio.entry_price,
            holding_periods=self.portfolio.holding_periods,
            current_price=self._get_current_price()
        )
        
        return features.astype(np.float32)
    
    def _execute_action(self, action: int, current_price: float) -> bool:
        """
        Execute trading action.
        
        Args:
            action: Action to take.
            current_price: Current market price.
            
        Returns:
            Whether a trade was executed.
        """
        trade_executed = False
        
        if action == Action.BUY:
            if self.portfolio.position <= 0:  # Not long
                # Close short position if exists
                if self.portfolio.position < 0:
                    self._close_position(current_price)
                
                # Open long position
                self._open_position(1.0, current_price)
                trade_executed = True
                
        elif action == Action.SELL:
            if self.portfolio.position >= 0:  # Not short
                # Close long position if exists
                if self.portfolio.position > 0:
                    self._close_position(current_price)
                
                # Open short position
                self._open_position(-1.0, current_price)
                trade_executed = True
        
        # HOLD action: do nothing
        
        return trade_executed
    
    def _open_position(self, direction: float, price: float) -> None:
        """Open a new position."""
        # Apply slippage
        adjusted_price = price * (1 + self.config.slippage * direction)
        
        # Calculate position size (full portfolio by default)
        position_size = self.portfolio.cash * self.config.max_position_size
        
        # Apply commission
        commission = position_size * self.config.commission_rate
        position_size -= commission
        
        # Update portfolio
        self.portfolio.position = direction
        self.portfolio.position_size = position_size
        self.portfolio.entry_price = adjusted_price
        self.portfolio.holding_periods = 0
        self.portfolio.cash -= position_size + commission
        self.portfolio.num_trades += 1
        
        logger.debug(
            f"Opened {'LONG' if direction > 0 else 'SHORT'} position: "
            f"size=${position_size:.2f}, entry=${adjusted_price:.2f}"
        )
    
    def _close_position(self, price: float) -> None:
        """Close current position."""
        if self.portfolio.position == 0:
            return
        
        # Apply slippage (opposite direction)
        adjusted_price = price * (1 - self.config.slippage * self.portfolio.position)
        
        # Calculate P&L
        if self.portfolio.position > 0:  # Long position
            pnl = (adjusted_price - self.portfolio.entry_price) / self.portfolio.entry_price
        else:  # Short position
            pnl = (self.portfolio.entry_price - adjusted_price) / self.portfolio.entry_price
        
        pnl_dollar = pnl * self.portfolio.position_size
        
        # Apply commission
        commission = self.portfolio.position_size * self.config.commission_rate
        pnl_dollar -= commission
        
        # Update portfolio
        self.portfolio.cash += self.portfolio.position_size + pnl_dollar
        self.portfolio.total_pnl += pnl_dollar
        
        if pnl_dollar > 0:
            self.portfolio.winning_trades += 1
        
        logger.debug(
            f"Closed position: PnL=${pnl_dollar:.2f} ({pnl*100:.2f}%)"
        )
        
        # Reset position
        self.portfolio.position = 0.0
        self.portfolio.position_size = 0.0
        self.portfolio.entry_price = 0.0
        self.portfolio.holding_periods = 0
    
    def _update_portfolio_value(self, current_price: float) -> None:
        """Update portfolio value and drawdown metrics."""
        # Calculate mark-to-market value
        if self.portfolio.position != 0:
            if self.portfolio.position > 0:  # Long
                unrealized_pnl = (
                    (current_price - self.portfolio.entry_price) 
                    / self.portfolio.entry_price 
                    * self.portfolio.position_size
                )
            else:  # Short
                unrealized_pnl = (
                    (self.portfolio.entry_price - current_price) 
                    / self.portfolio.entry_price 
                    * self.portfolio.position_size
                )
            
            self.portfolio.portfolio_value = (
                self.portfolio.cash + self.portfolio.position_size + unrealized_pnl
            )
        else:
            self.portfolio.portfolio_value = self.portfolio.cash
        
        # Update peak and drawdown
        if self.portfolio.portfolio_value > self.portfolio.peak_value:
            self.portfolio.peak_value = self.portfolio.portfolio_value
        
        self.portfolio.current_drawdown = (
            (self.portfolio.peak_value - self.portfolio.portfolio_value) 
            / self.portfolio.peak_value
        )
        self.portfolio.max_drawdown = max(
            self.portfolio.max_drawdown, 
            self.portfolio.current_drawdown
        )
        
        # Update total return
        self.portfolio.total_return = (
            (self.portfolio.portfolio_value - self.config.initial_capital) 
            / self.config.initial_capital
        )
        
        # Check stop loss / take profit
        if self.portfolio.position != 0:
            self._check_risk_limits(current_price)
    
    def _check_risk_limits(self, current_price: float) -> None:
        """Check and enforce stop loss / take profit."""
        if self.portfolio.entry_price == 0:
            return
        
        # Calculate unrealized P&L percentage
        if self.portfolio.position > 0:
            pnl_pct = (current_price - self.portfolio.entry_price) / self.portfolio.entry_price
        else:
            pnl_pct = (self.portfolio.entry_price - current_price) / self.portfolio.entry_price
        
        # Check stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            logger.info(f"Stop loss triggered at {pnl_pct*100:.2f}%")
            self._close_position(current_price)
        
        # Check take profit
        elif pnl_pct >= self.config.take_profit_pct:
            logger.info(f"Take profit triggered at {pnl_pct*100:.2f}%")
            self._close_position(current_price)
    
    def _calculate_reward(self, trade_executed: bool) -> float:
        """
        Calculate reward for current step.
        
        Reward = portfolio_return - risk_penalty
        risk_penalty = lambda * max_drawdown
        
        Args:
            trade_executed: Whether a trade was executed this step.
            
        Returns:
            Reward value.
        """
        # Calculate step return
        if len(self.episode_rewards) > 0:
            prev_value = self.config.initial_capital
            for r in self.episode_rewards:
                prev_value *= (1 + r)
        else:
            prev_value = self.config.initial_capital
        
        step_return = (self.portfolio.portfolio_value - prev_value) / prev_value
        
        # Risk penalty based on drawdown
        risk_penalty = self.config.risk_penalty_lambda * self.portfolio.current_drawdown
        
        # Small holding penalty to encourage active trading
        holding_penalty = 0.0
        if not trade_executed and self.portfolio.position == 0:
            holding_penalty = self.config.holding_penalty
        
        # Combine components
        reward = step_return - risk_penalty - holding_penalty
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -1.0, 1.0)
        
        return float(reward)
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate early."""
        # Terminate if max drawdown exceeded
        if self.portfolio.max_drawdown >= self.config.max_drawdown_threshold:
            logger.info(f"Episode terminated: max drawdown {self.portfolio.max_drawdown:.2%}")
            return True
        
        # Terminate if bankrupt
        if self.portfolio.portfolio_value <= 0:
            logger.info("Episode terminated: bankrupt")
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio.portfolio_value,
            'total_return': self.portfolio.total_return,
            'current_drawdown': self.portfolio.current_drawdown,
            'max_drawdown': self.portfolio.max_drawdown,
            'position': self.portfolio.position,
            'num_trades': self.portfolio.num_trades,
            'winning_trades': self.portfolio.winning_trades,
            'win_rate': (
                self.portfolio.winning_trades / max(1, self.portfolio.num_trades)
            ),
            'total_pnl': self.portfolio.total_pnl,
        }
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()
        return None
    
    def _render_human(self) -> None:
        """Print current state to console."""
        print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """Generate ANSI string representation."""
        price = self._get_current_price()
        pos_str = "LONG" if self.portfolio.position > 0 else (
            "SHORT" if self.portfolio.position < 0 else "FLAT"
        )
        
        return (
            f"Step {self.current_step}/{self.end_step} | "
            f"Price: ${price:.2f} | "
            f"Position: {pos_str} | "
            f"Value: ${self.portfolio.portfolio_value:.2f} | "
            f"Return: {self.portfolio.total_return:.2%} | "
            f"Drawdown: {self.portfolio.current_drawdown:.2%}"
        )
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for completed episode."""
        if not self.episode_rewards:
            return {}
        
        total_reward = sum(self.episode_rewards)
        avg_reward = np.mean(self.episode_rewards)
        reward_std = np.std(self.episode_rewards)
        
        # Calculate Sharpe ratio (annualized)
        if reward_std > 0:
            sharpe = avg_reward / reward_std * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Action distribution
        action_counts = {
            'BUY': self.episode_actions.count(0),
            'HOLD': self.episode_actions.count(1),
            'SELL': self.episode_actions.count(2)
        }
        
        return {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'sharpe_ratio': sharpe,
            'total_return': self.portfolio.total_return,
            'max_drawdown': self.portfolio.max_drawdown,
            'num_trades': self.portfolio.num_trades,
            'win_rate': self.portfolio.winning_trades / max(1, self.portfolio.num_trades),
            'total_pnl': self.portfolio.total_pnl,
            'action_distribution': action_counts,
            'episode_length': len(self.episode_rewards)
        }


def create_trading_env(
    market_data: pd.DataFrame,
    news_events: Optional[List] = None,
    initial_capital: float = 10000.0,
    episode_length: int = 252,
    render_mode: Optional[str] = None,
    **kwargs
) -> TradingEnvironment:
    """
    Factory function to create a trading environment.
    
    Args:
        market_data: DataFrame with OHLCV data.
        news_events: Optional list of economic news events.
        initial_capital: Starting capital.
        episode_length: Length of each episode in trading days.
        render_mode: Rendering mode.
        **kwargs: Additional TradingConfig parameters.
        
    Returns:
        Configured TradingEnvironment instance.
    """
    config = TradingConfig(
        initial_capital=initial_capital,
        episode_length=episode_length,
        **kwargs
    )
    
    return TradingEnvironment(
        market_data=market_data,
        news_events=news_events,
        config=config,
        render_mode=render_mode
    )


# Register environment with Gymnasium
def register_trading_env():
    """Register the trading environment with Gymnasium."""
    try:
        gym.envs.registration.register(
            id='NewsTrading-v0',
            entry_point='src.analysis.trading_env:TradingEnvironment',
        )
        logger.info("Registered NewsTrading-v0 environment")
    except Exception as e:
        logger.debug(f"Environment already registered or error: {e}")

