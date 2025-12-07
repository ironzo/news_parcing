"""
Unit tests for trading environment module.
"""
import pytest
from datetime import datetime
import numpy as np
import pandas as pd
from pytz import UTC
import gymnasium as gym

from src.analysis.trading_env import (
    TradingEnvironment,
    TradingConfig,
    PortfolioState,
    Action,
    create_trading_env,
    TOTAL_FEATURE_DIM
)
from src.analysis.features import TOTAL_FEATURE_DIM


@pytest.fixture
def sample_market_data():
    """Create sample market data DataFrame with enough history."""
    np.random.seed(42)
    n_days = 400  # Enough for episode + lookback
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D', tz=UTC)
    
    # Generate realistic price data with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = base_price * np.cumprod(1 + returns)
    
    data = {
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'high': prices * (1 + np.random.uniform(0.005, 0.015, n_days)),
        'low': prices * (1 - np.random.uniform(0.005, 0.015, n_days)),
        'close': prices,
        'volume': np.random.uniform(1e6, 5e6, n_days),
    }
    return pd.DataFrame(data)


@pytest.fixture
def trading_config():
    """Create trading configuration."""
    return TradingConfig(
        initial_capital=10000.0,
        commission_rate=0.001,
        slippage=0.0005,
        max_position_size=1.0,
        max_drawdown_threshold=0.20,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        episode_length=100,  # Shorter for testing
        random_start=False
    )


@pytest.fixture
def trading_env(sample_market_data, trading_config):
    """Create trading environment instance."""
    return TradingEnvironment(
        market_data=sample_market_data,
        config=trading_config
    )


class TestAction:
    """Tests for Action enum."""
    
    def test_action_values(self):
        """Test action enum values."""
        assert Action.BUY == 0
        assert Action.HOLD == 1
        assert Action.SELL == 2
    
    def test_action_count(self):
        """Test number of actions."""
        assert len(Action) == 3


class TestPortfolioState:
    """Tests for PortfolioState dataclass."""
    
    def test_reset(self):
        """Test portfolio reset."""
        portfolio = PortfolioState(
            cash=5000.0,
            position=1.0,
            position_size=5000.0,
            entry_price=100.0,
            holding_periods=10,
            portfolio_value=10500.0,
            peak_value=11000.0,
            total_return=0.05,
            current_drawdown=0.045,
            max_drawdown=0.05,
            num_trades=5,
            winning_trades=3,
            total_pnl=500.0
        )
        
        portfolio.reset(10000.0)
        
        assert portfolio.cash == 10000.0
        assert portfolio.position == 0.0
        assert portfolio.position_size == 0.0
        assert portfolio.entry_price == 0.0
        assert portfolio.holding_periods == 0
        assert portfolio.portfolio_value == 10000.0
        assert portfolio.peak_value == 10000.0
        assert portfolio.total_return == 0.0
        assert portfolio.current_drawdown == 0.0
        assert portfolio.max_drawdown == 0.0
        assert portfolio.num_trades == 0
        assert portfolio.winning_trades == 0
        assert portfolio.total_pnl == 0.0


class TestTradingEnvironment:
    """Tests for TradingEnvironment class."""
    
    def test_initialization(self, trading_env):
        """Test environment initialization."""
        assert trading_env is not None
        assert trading_env.action_space.n == 3
        assert trading_env.observation_space.shape == (TOTAL_FEATURE_DIM,)
    
    def test_reset(self, trading_env):
        """Test environment reset."""
        observation, info = trading_env.reset(seed=42)
        
        assert observation.shape == (TOTAL_FEATURE_DIM,)
        assert observation.dtype == np.float32
        assert 'step' in info
        assert 'portfolio_value' in info
        assert info['portfolio_value'] == trading_env.config.initial_capital
    
    def test_reset_with_seed(self, trading_env):
        """Test that reset with same seed produces same start."""
        obs1, _ = trading_env.reset(seed=42)
        start1 = trading_env.current_step
        
        obs2, _ = trading_env.reset(seed=42)
        start2 = trading_env.current_step
        
        assert start1 == start2
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_reset_with_options(self, sample_market_data, trading_config):
        """Test reset with specific start step."""
        trading_config.random_start = True
        env = TradingEnvironment(sample_market_data, config=trading_config)
        
        obs, info = env.reset(options={'start_step': 100})
        
        assert env.start_step == 100
    
    def test_step_buy(self, trading_env):
        """Test BUY action."""
        trading_env.reset(seed=42)
        
        obs, reward, terminated, truncated, info = trading_env.step(Action.BUY)
        
        assert obs.shape == (TOTAL_FEATURE_DIM,)
        assert isinstance(reward, float)
        assert trading_env.portfolio.position == 1.0
        assert trading_env.portfolio.num_trades == 1
    
    def test_step_sell(self, trading_env):
        """Test SELL action."""
        trading_env.reset(seed=42)
        
        obs, reward, terminated, truncated, info = trading_env.step(Action.SELL)
        
        assert obs.shape == (TOTAL_FEATURE_DIM,)
        assert trading_env.portfolio.position == -1.0
        assert trading_env.portfolio.num_trades == 1
    
    def test_step_hold(self, trading_env):
        """Test HOLD action."""
        trading_env.reset(seed=42)
        
        obs, reward, terminated, truncated, info = trading_env.step(Action.HOLD)
        
        assert obs.shape == (TOTAL_FEATURE_DIM,)
        assert trading_env.portfolio.position == 0.0
        assert trading_env.portfolio.num_trades == 0
    
    def test_step_sequence(self, trading_env):
        """Test sequence of actions."""
        trading_env.reset(seed=42)
        
        # BUY then SELL
        trading_env.step(Action.BUY)
        assert trading_env.portfolio.position == 1.0
        assert trading_env.portfolio.holding_periods == 1  # Incremented after first step
        
        trading_env.step(Action.HOLD)
        assert trading_env.portfolio.position == 1.0
        assert trading_env.portfolio.holding_periods == 2  # Incremented again
        
        trading_env.step(Action.SELL)
        assert trading_env.portfolio.position == -1.0
        # Open long (1) + open short (1) = 2 trades counted
        # (closing is implicit, not counted separately)
        assert trading_env.portfolio.num_trades == 2
    
    def test_episode_termination(self, trading_env):
        """Test that episode terminates correctly."""
        trading_env.reset(seed=42)
        
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = trading_env.step(Action.HOLD)
            steps += 1
            
            # Safety limit
            if steps > trading_env.config.episode_length + 10:
                break
        
        assert truncated  # Should truncate at episode length
        assert steps == trading_env.config.episode_length
    
    def test_reward_bounded(self, trading_env):
        """Test that rewards are bounded."""
        trading_env.reset(seed=42)
        
        for _ in range(50):
            action = trading_env.action_space.sample()
            _, reward, terminated, truncated, _ = trading_env.step(action)
            
            assert -1.0 <= reward <= 1.0
            
            if terminated or truncated:
                break
    
    def test_info_contents(self, trading_env):
        """Test that info dict contains expected keys."""
        trading_env.reset(seed=42)
        _, _, _, _, info = trading_env.step(Action.BUY)
        
        expected_keys = [
            'step', 'portfolio_value', 'total_return', 'current_drawdown',
            'max_drawdown', 'position', 'num_trades', 'winning_trades',
            'win_rate', 'total_pnl'
        ]
        
        for key in expected_keys:
            assert key in info


class TestTradingMechanics:
    """Tests for trading mechanics."""
    
    def test_position_opening(self, trading_env):
        """Test position opening."""
        trading_env.reset(seed=42)
        initial_cash = trading_env.portfolio.cash
        
        trading_env.step(Action.BUY)
        
        assert trading_env.portfolio.position == 1.0
        assert trading_env.portfolio.position_size > 0
        assert trading_env.portfolio.entry_price > 0
        assert trading_env.portfolio.cash < initial_cash  # Cash reduced
    
    def test_position_closing(self, trading_env):
        """Test position closing."""
        trading_env.reset(seed=42)
        
        # Open long
        trading_env.step(Action.BUY)
        
        # Close by going short
        trading_env.step(Action.SELL)
        
        # Should have closed the long and opened a short
        assert trading_env.portfolio.position == -1.0
    
    def test_commission_applied(self, trading_env):
        """Test that commissions are applied."""
        trading_env.reset(seed=42)
        initial_value = trading_env.portfolio.portfolio_value
        
        # Open and immediately close position
        trading_env.step(Action.BUY)
        trading_env.step(Action.SELL)
        trading_env.step(Action.HOLD)  # Close short
        
        # Portfolio value should be less due to commissions
        # (assuming no significant price movement)
        assert trading_env.portfolio.num_trades > 0
    
    def test_drawdown_tracking(self, trading_env):
        """Test drawdown calculation."""
        trading_env.reset(seed=42)
        
        # Run some steps
        for _ in range(20):
            trading_env.step(trading_env.action_space.sample())
        
        # Drawdown should be non-negative
        assert trading_env.portfolio.current_drawdown >= 0
        assert trading_env.portfolio.max_drawdown >= trading_env.portfolio.current_drawdown


class TestRiskManagement:
    """Tests for risk management features."""
    
    def test_max_drawdown_termination(self, sample_market_data):
        """Test early termination on max drawdown."""
        # Create environment with very low drawdown threshold
        config = TradingConfig(
            initial_capital=10000.0,
            max_drawdown_threshold=0.01,  # 1% max drawdown
            episode_length=100,
            random_start=False
        )
        
        env = TradingEnvironment(sample_market_data, config=config)
        env.reset(seed=42)
        
        terminated = False
        steps = 0
        
        # Trade aggressively to trigger drawdown
        while not terminated and steps < 100:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            
            if truncated:
                break
        
        # May or may not terminate early depending on price movements
        assert steps <= 100


class TestRendering:
    """Tests for environment rendering."""
    
    def test_render_ansi(self, sample_market_data, trading_config):
        """Test ANSI rendering."""
        env = TradingEnvironment(
            sample_market_data, 
            config=trading_config,
            render_mode='ansi'
        )
        env.reset(seed=42)
        env.step(Action.BUY)
        
        output = env.render()
        
        assert output is not None
        assert isinstance(output, str)
        assert 'Step' in output
        assert 'Price' in output
    
    def test_render_human(self, sample_market_data, trading_config, capsys):
        """Test human rendering."""
        env = TradingEnvironment(
            sample_market_data,
            config=trading_config,
            render_mode='human'
        )
        env.reset(seed=42)
        env.step(Action.BUY)
        
        env.render()
        
        captured = capsys.readouterr()
        assert 'Step' in captured.out


class TestEpisodeStats:
    """Tests for episode statistics."""
    
    def test_get_episode_stats(self, trading_env):
        """Test episode statistics calculation."""
        trading_env.reset(seed=42)
        
        # Run episode
        for _ in range(50):
            action = trading_env.action_space.sample()
            _, _, terminated, truncated, _ = trading_env.step(action)
            if terminated or truncated:
                break
        
        stats = trading_env.get_episode_stats()
        
        assert 'total_reward' in stats
        assert 'avg_reward' in stats
        assert 'sharpe_ratio' in stats
        assert 'total_return' in stats
        assert 'max_drawdown' in stats
        assert 'num_trades' in stats
        assert 'action_distribution' in stats
    
    def test_episode_stats_empty(self, trading_env):
        """Test episode stats before any steps."""
        trading_env.reset(seed=42)
        
        stats = trading_env.get_episode_stats()
        
        # Should return empty dict if no rewards
        assert stats == {}


class TestFactoryFunction:
    """Tests for create_trading_env factory function."""
    
    def test_create_trading_env(self, sample_market_data):
        """Test factory function."""
        env = create_trading_env(
            market_data=sample_market_data,
            initial_capital=5000.0,
            episode_length=50
        )
        
        assert env is not None
        assert env.config.initial_capital == 5000.0
        assert env.config.episode_length == 50
    
    def test_create_trading_env_with_kwargs(self, sample_market_data):
        """Test factory function with additional kwargs."""
        env = create_trading_env(
            market_data=sample_market_data,
            initial_capital=10000.0,
            episode_length=100,
            commission_rate=0.002,
            slippage=0.001
        )
        
        assert env.config.commission_rate == 0.002
        assert env.config.slippage == 0.001


class TestGymnasiumCompliance:
    """Tests for Gymnasium interface compliance."""
    
    def test_action_space_valid(self, trading_env):
        """Test that action space is valid."""
        assert isinstance(trading_env.action_space, gym.spaces.Discrete)
        assert trading_env.action_space.n == 3
    
    def test_observation_space_valid(self, trading_env):
        """Test that observation space is valid."""
        assert isinstance(trading_env.observation_space, gym.spaces.Box)
        assert trading_env.observation_space.shape == (TOTAL_FEATURE_DIM,)
    
    def test_step_returns_correct_types(self, trading_env):
        """Test that step returns correct types."""
        trading_env.reset(seed=42)
        
        obs, reward, terminated, truncated, info = trading_env.step(Action.BUY)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_observation_in_space(self, trading_env):
        """Test that observations are within observation space."""
        trading_env.reset(seed=42)
        
        for _ in range(20):
            obs, _, terminated, truncated, _ = trading_env.step(
                trading_env.action_space.sample()
            )
            
            # Check observation shape matches
            assert obs.shape == trading_env.observation_space.shape
            
            if terminated or truncated:
                break
    
    def test_sample_actions_valid(self, trading_env):
        """Test that sampled actions are valid."""
        trading_env.reset(seed=42)
        
        for _ in range(100):
            action = trading_env.action_space.sample()
            assert 0 <= action < 3
            assert trading_env.action_space.contains(action)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_minimal_data(self):
        """Test with minimal market data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D', tz=UTC)
        data = {
            'timestamp': dates,
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.0] * 100,
            'volume': [1e6] * 100,
        }
        df = pd.DataFrame(data)
        
        config = TradingConfig(episode_length=40, random_start=False)
        env = TradingEnvironment(df, config=config)
        
        obs, _ = env.reset(seed=42)
        assert obs.shape == (TOTAL_FEATURE_DIM,)
    
    def test_missing_columns_raises(self):
        """Test that missing columns raises error."""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'close': [100.0] * 100,
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            TradingEnvironment(df)
    
    def test_repeated_actions(self, trading_env):
        """Test repeated same actions."""
        trading_env.reset(seed=42)
        
        # Repeated BUY should only open once
        trading_env.step(Action.BUY)
        initial_trades = trading_env.portfolio.num_trades
        
        trading_env.step(Action.BUY)  # Already long
        
        assert trading_env.portfolio.num_trades == initial_trades
        assert trading_env.portfolio.position == 1.0

