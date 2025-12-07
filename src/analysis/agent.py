"""
RL Agent training module using Stable-Baselines3 PPO.

This module provides functionality to:
- Configure and train PPO agents
- Implement training loop with checkpointing
- Support hyperparameter tuning
- Integrate TensorBoard logging
- Evaluate and save best models
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any, Tuple
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from .trading_env import TradingEnvironment, TradingConfig, create_trading_env
from .features import FeatureExtractor


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for PPO agent training."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    n_steps: int = 2048
    ent_coef: float = 0.01
    clip_range: float = 0.2
    n_epochs: int = 10
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    
    # Network architecture
    policy: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    
    # Training settings
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 50_000
    
    # Paths
    model_dir: str = "models"
    log_dir: str = "logs"
    tensorboard_log: str = "tensorboard"
    
    # Data split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Misc
    seed: int = 42
    verbose: int = 1
    device: str = "auto"


class SharpeRatioCallback(BaseCallback):
    """
    Custom callback to track Sharpe ratio during training.
    Saves the best model based on validation Sharpe ratio.
    """
    
    def __init__(
        self,
        eval_env: TradingEnvironment,
        best_model_save_path: str,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        self.sharpe_history: List[float] = []
        self.eval_timesteps: List[int] = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            sharpe = self._evaluate_sharpe()
            self.sharpe_history.append(sharpe)
            self.eval_timesteps.append(self.num_timesteps)
            
            if self.verbose > 0:
                logger.info(
                    f"Timestep {self.num_timesteps}: Sharpe Ratio = {sharpe:.3f} "
                    f"(Best: {self.best_sharpe:.3f})"
                )
            
            # Save best model
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.model.save(self.best_model_save_path)
                if self.verbose > 0:
                    logger.info(f"New best model saved with Sharpe = {sharpe:.3f}")
        
        return True
    
    def _evaluate_sharpe(self) -> float:
        """Evaluate current policy and calculate Sharpe ratio."""
        episode_rewards = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        # Calculate Sharpe ratio (annualized)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        if std_reward > 0:
            sharpe = mean_reward / std_reward * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return sharpe


class TradingAgent:
    """
    PPO-based trading agent with training, evaluation, and prediction capabilities.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize trading agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config or AgentConfig()
        self.model: Optional[PPO] = None
        self.train_env: Optional[TradingEnvironment] = None
        self.val_env: Optional[TradingEnvironment] = None
        self.test_env: Optional[TradingEnvironment] = None
        self.training_history: Dict[str, List] = {
            'timesteps': [],
            'rewards': [],
            'sharpe_ratios': []
        }
        
        # Create directories
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.tensorboard_log, exist_ok=True)
        
        logger.info("Initialized TradingAgent")
    
    def prepare_data(
        self,
        market_data: pd.DataFrame,
        news_events: Optional[List] = None,
        trading_config: Optional[TradingConfig] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            market_data: Full market data DataFrame.
            news_events: Optional news events.
            trading_config: Trading environment configuration.
            
        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        n = len(market_data)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_data = market_data.iloc[:train_end].reset_index(drop=True)
        val_data = market_data.iloc[train_end:val_end].reset_index(drop=True)
        test_data = market_data.iloc[val_end:].reset_index(drop=True)
        
        logger.info(
            f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )
        
        # Create environments
        trading_config = trading_config or TradingConfig()
        
        self.train_env = create_trading_env(
            market_data=train_data,
            news_events=news_events,
            initial_capital=trading_config.initial_capital,
            episode_length=min(trading_config.episode_length, len(train_data) - 60),
            random_start=True
        )
        
        self.val_env = create_trading_env(
            market_data=val_data,
            news_events=news_events,
            initial_capital=trading_config.initial_capital,
            episode_length=min(trading_config.episode_length, len(val_data) - 60),
            random_start=False
        )
        
        self.test_env = create_trading_env(
            market_data=test_data,
            news_events=news_events,
            initial_capital=trading_config.initial_capital,
            episode_length=min(trading_config.episode_length, len(test_data) - 60),
            random_start=False
        )
        
        return train_data, val_data, test_data
    
    def create_model(self, env: Optional[TradingEnvironment] = None) -> PPO:
        """
        Create PPO model with configured hyperparameters.
        
        Args:
            env: Training environment. Uses self.train_env if None.
            
        Returns:
            Configured PPO model.
        """
        env = env or self.train_env
        
        if env is None:
            raise ValueError("No environment provided. Call prepare_data first.")
        
        # Wrap environment in Monitor for logging
        monitored_env = Monitor(env, self.config.log_dir)
        
        # Create PPO model
        policy_kwargs = {
            "net_arch": self.config.net_arch
        }
        
        self.model = PPO(
            policy=self.config.policy,
            env=monitored_env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            batch_size=self.config.batch_size,
            n_steps=self.config.n_steps,
            ent_coef=self.config.ent_coef,
            clip_range=self.config.clip_range,
            n_epochs=self.config.n_epochs,
            gae_lambda=self.config.gae_lambda,
            max_grad_norm=self.config.max_grad_norm,
            vf_coef=self.config.vf_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.tensorboard_log,
            verbose=self.config.verbose,
            seed=self.config.seed,
            device=self.config.device
        )
        
        logger.info(f"Created PPO model with {self.config.policy}")
        return self.model
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        progress_bar: bool = True
    ) -> PPO:
        """
        Train the agent.
        
        Args:
            total_timesteps: Override total timesteps from config.
            callback: Additional callback(s) to use.
            progress_bar: Whether to show progress bar.
            
        Returns:
            Trained model.
        """
        if self.model is None:
            self.create_model()
        
        timesteps = total_timesteps or self.config.total_timesteps
        
        # Create callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.checkpoint_freq,
            save_path=self.config.model_dir,
            name_prefix="ppo_trading"
        )
        callbacks.append(checkpoint_callback)
        
        # Sharpe ratio callback for validation
        if self.val_env is not None:
            sharpe_callback = SharpeRatioCallback(
                eval_env=self.val_env,
                best_model_save_path=os.path.join(self.config.model_dir, "best_model"),
                n_eval_episodes=self.config.n_eval_episodes,
                eval_freq=self.config.eval_freq,
                verbose=self.config.verbose
            )
            callbacks.append(sharpe_callback)
        
        # Add user callback
        if callback is not None:
            callbacks.append(callback)
        
        callback_list = CallbackList(callbacks)
        
        logger.info(f"Starting training for {timesteps} timesteps")
        
        # Train
        self.model.learn(
            total_timesteps=timesteps,
            callback=callback_list,
            progress_bar=progress_bar
        )
        
        # Save final model
        final_path = os.path.join(self.config.model_dir, "final_model")
        self.model.save(final_path)
        logger.info(f"Training complete. Final model saved to {final_path}")
        
        # Store training history from Sharpe callback
        for cb in callbacks:
            if isinstance(cb, SharpeRatioCallback):
                self.training_history['timesteps'] = cb.eval_timesteps
                self.training_history['sharpe_ratios'] = cb.sharpe_history
        
        return self.model
    
    def evaluate(
        self,
        env: Optional[TradingEnvironment] = None,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on. Uses test_env if None.
            n_episodes: Number of evaluation episodes.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Train or load a model first.")
        
        env = env or self.test_env
        if env is None:
            raise ValueError("No environment provided.")
        
        episode_rewards = []
        episode_returns = []
        episode_drawdowns = []
        episode_trades = []
        episode_win_rates = []
        
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            # Collect stats
            stats = env.get_episode_stats()
            episode_rewards.append(episode_reward)
            episode_returns.append(stats.get('total_return', 0))
            episode_drawdowns.append(stats.get('max_drawdown', 0))
            episode_trades.append(stats.get('num_trades', 0))
            episode_win_rates.append(stats.get('win_rate', 0))
        
        # Calculate aggregate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        sharpe = mean_reward / std_reward * np.sqrt(252) if std_reward > 0 else 0
        
        metrics = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'sharpe_ratio': float(sharpe),
            'mean_return': float(np.mean(episode_returns)),
            'mean_max_drawdown': float(np.mean(episode_drawdowns)),
            'mean_trades': float(np.mean(episode_trades)),
            'mean_win_rate': float(np.mean(episode_win_rates)),
            'n_episodes': n_episodes
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, float]:
        """
        Predict action for given observation.
        
        Args:
            observation: Current state observation.
            deterministic: Whether to use deterministic action.
            
        Returns:
            Tuple of (action, confidence).
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Get action probabilities for confidence
        obs_tensor = self.model.policy.obs_to_tensor(observation.reshape(1, -1))[0]
        with self.model.policy.features_extractor.training:
            features = self.model.policy.features_extractor(obs_tensor)
            latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            action_logits = self.model.policy.action_net(latent_pi)
            probs = action_logits.softmax(dim=-1).detach().cpu().numpy()[0]
        
        confidence = float(probs[action])
        
        return int(action), confidence
    
    def save(self, path: str) -> None:
        """Save model to path."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str, env: Optional[TradingEnvironment] = None) -> PPO:
        """
        Load model from path.
        
        Args:
            path: Path to saved model.
            env: Environment to use with model.
            
        Returns:
            Loaded model.
        """
        env = env or self.train_env
        self.model = PPO.load(path, env=env)
        logger.info(f"Model loaded from {path}")
        return self.model


def train_trading_agent(
    market_data: pd.DataFrame,
    news_events: Optional[List] = None,
    agent_config: Optional[AgentConfig] = None,
    trading_config: Optional[TradingConfig] = None,
    total_timesteps: int = 100_000
) -> Tuple[TradingAgent, Dict[str, float]]:
    """
    Convenience function to train a trading agent.
    
    Args:
        market_data: Historical market data.
        news_events: Optional news events.
        agent_config: Agent configuration.
        trading_config: Trading environment configuration.
        total_timesteps: Training timesteps.
        
    Returns:
        Tuple of (trained agent, test metrics).
    """
    agent = TradingAgent(agent_config)
    
    # Prepare data
    agent.prepare_data(market_data, news_events, trading_config)
    
    # Create and train model
    agent.create_model()
    agent.train(total_timesteps=total_timesteps)
    
    # Evaluate on test set
    test_metrics = agent.evaluate()
    
    return agent, test_metrics

