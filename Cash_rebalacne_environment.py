import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv_initial_balance(gym.Env):
    def __init__(self, prices_df, window_size=20, alpha=1.0, beta=0.1, initial_balance=10000):
        super(PortfolioEnv_initial_balance, self).__init__()

        self.prices = prices_df
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.initial_balance = initial_balance

        self.current_step = self.window_size
        self.num_stocks = self.prices.shape[1]
        self.max_steps = len(self.prices) - 1

        # Action: portfolio weights (with shorting allowed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_stocks,), dtype=np.float32)

        # Observation: normalized window + previous weights + cash balance -> flatten (window_size * num_stocks + num_stocks + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.num_stocks + self.num_stocks + 1,),
            dtype=np.float32
        )

        # Initialize variables
        self.balance = self.initial_balance  # Cash balance
        self.stock_holdings = np.zeros(self.num_stocks)  # Amount of each stock held
        self.done = False
        self.weights = np.zeros(self.num_stocks)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.done = False
        self.balance = self.initial_balance
        self.stock_holdings = np.zeros(self.num_stocks)
        self.weights = np.zeros(self.num_stocks)
        
        obs = self._get_observation()
    # Convertion of observation to float32 explicitly to match the observation_space dtype
        return obs.astype(np.float32), {}
    def _get_observation(self):
        window = self.prices.iloc[self.current_step - self.window_size:self.current_step]
        first_day_prices = window.iloc[0].replace(0, 1e-6)
        normalized_window = (window / first_day_prices).values.astype(np.float32)
        flat_prices = normalized_window.flatten()
        
        # Return the observation (prices, weights, and balance)
        return np.concatenate([flat_prices, self.weights.astype(np.float32), np.array([self.balance])])

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, self.done, False, {}

        # Normalize action to sum to 1 (portfolio constraint)
        self.weights = action / np.sum(np.abs(action))  # Portfolio weights should sum to 1

        # Determine portfolio value and transactions
        current_prices = self.prices.iloc[self.current_step]
        previous_prices = self.prices.iloc[self.current_step - 1]
        daily_log_returns = np.log(current_prices / previous_prices).values

        # Portfolio return calculation
        portfolio_return = np.dot(self.weights, daily_log_returns)

        # Portfolio risk: covariance over the window
        window = self.prices.iloc[self.current_step - self.window_size:self.current_step]
        log_returns = np.log(window / window.shift(1)).dropna()
        cov_matrix = log_returns.cov().values
        portfolio_risk = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))

        reward = self.alpha * portfolio_return - self.beta * portfolio_risk

        # Portfolio update logic: buying/selling stocks
        total_value = self.balance + np.dot(self.stock_holdings, current_prices)
        target_stock_holdings = self.weights * total_value / current_prices  # Desired stock holdings based on weights

        # Calculate transaction cost: buying/selling stocks (no short-selling restrictions)
        transaction_cost = np.abs(target_stock_holdings - self.stock_holdings).sum() * 0.001  # 0.1% transaction fee
        self.balance -= transaction_cost  # Deduct transaction costs from cash balance

        # Update stock holdings and balance
        for i in range(self.num_stocks):
            cost_to_buy = (target_stock_holdings[i] - self.stock_holdings[i]) * current_prices[i]
            if cost_to_buy > 0:  # Buying
                if self.balance >= cost_to_buy:
                    self.balance -= cost_to_buy
                    self.stock_holdings[i] = target_stock_holdings[i]
            elif cost_to_buy < 0:  # Selling
                self.balance += np.abs(cost_to_buy)  # Return funds from selling
                self.stock_holdings[i] = target_stock_holdings[i]

        portfolio_value = self.balance + np.dot(self.stock_holdings, current_prices)
        # The balance is already updated in the buying/selling steps above
        # Don't reset balance, keep it as the updated amount

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Return observation, reward, done, truncated flag
        obs = self._get_observation()
        truncated = False

        return obs.astype(np.float32), reward, self.done, truncated, {'return': portfolio_return, 'risk': portfolio_risk}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Cash Balance: {self.balance}")
        print(f"Portfolio Weights: {self.weights}")
        print(f"Stock Holdings: {self.stock_holdings}")
        print(f"Total Portfolio Value: {self.balance + np.dot(self.stock_holdings, self.prices.iloc[self.current_step])}")
