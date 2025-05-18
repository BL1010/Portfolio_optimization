import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv_no_cash_rebalance(gym.Env):
    def __init__(self, prices_df, window_size=20, alpha=1.0, beta=0.1):
        super(PortfolioEnv_no_cash_rebalance, self).__init__()

        self.prices = prices_df
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.current_step = self.window_size

        self.num_stocks = self.prices.shape[1]
        self.max_steps = len(self.prices) - 1

        # Action: portfolio weights (with shorting allowed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_stocks,), dtype=np.float32)

        # Observation: normalized window + previous weights -> flatten (window_size * num_stocks + num_stocks)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.num_stocks + self.num_stocks,),
            dtype=np.float32
        )

        self.done = False
        self.weights = np.zeros(self.num_stocks)

    def reset(self,*,seed = None,options = None):
        super().reset(seed = seed)
        self.current_step = self.window_size
        self.done = False
        self.weights = np.zeros(self.num_stocks)
        return self._get_observation(),{}

    def _get_observation(self):
        window = self.prices.iloc[self.current_step - self.window_size:self.current_step]
        first_day_prices = window.iloc[0].replace(0, 1e-6)
        normalized_window = (window / first_day_prices).values.astype(np.float32)
        flat_prices = normalized_window.flatten()
        return np.concatenate([flat_prices, self.weights.astype(np.float32)])

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, self.done,False, {}

        # Normalize action to sum to 1 (portfolio constraint)
        # action_sum = np.sum(action)
        # if action_sum != 0:
        #     action = action / action_sum
        # else: #in the next training doe
        #     action = np.ones_like(action) / len(action)
        
        self.weights = action

        current_prices = self.prices.iloc[self.current_step]
        previous_prices = self.prices.iloc[self.current_step - 1]
        daily_log_returns = np.log(current_prices / previous_prices).values

        portfolio_return = np.dot(self.weights, daily_log_returns)

        # Portfolio risk: covariance over the window
        window = self.prices.iloc[self.current_step - self.window_size:self.current_step]
        log_returns = np.log(window / window.shift(1)).dropna()
        cov_matrix = log_returns.cov().values
        portfolio_risk = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))

        reward = self.alpha * portfolio_return - self.beta * portfolio_risk

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        obs = self._get_observation()
        truncated = False

        return obs, reward, self.done, truncated,{
            'return': portfolio_return,
            'risk': portfolio_risk
        }

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Portfolio Weights: {self.weights}")



