import os
import numpy as np
import pandas as pd
from .base_env import BasePortfolioEnv

class HistoricalSP500Env(BasePortfolioEnv):
    """
    Real Historical 10+ Year S&P 500 Market Crash Backtesting Environment.
    
    Replays real daily returns of S&P 500 and NASDAQ Composite over 2014-2024.
    Tracks:
        - Real historical asset price trajectories
        - Rolling empirical covariance matrix Sigma_t (60-day window)
        - Running High-Water Mark H_t
        - Instantaneous Drawdown D_t = 1 - W_t / H_t
        - Proportional transaction costs (10 bps)
    """
    def __init__(
        self,
        csv_path="data/historical_sp500_10yr_returns.csv",
        start_year="2014",
        end_year="2024",
        r=0.02,
        dt=1.0/252.0,
        gamma=2.0,
        max_drawdown=0.20,
        initial_wealth=1.0,
        transaction_cost=0.001
    ):
        if not os.path.exists(csv_path):
            from ..data.fetch_historical_data import fetch_and_process_historical_data
            fetch_and_process_historical_data()
            
        returns_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Filter target 10-year date range
        returns_df = returns_df[(returns_df.index >= f"{start_year}-01-01") & (returns_df.index <= f"{end_year}-12-31")]
        
        self.dates = returns_df.index
        self.returns_matrix = returns_df.values  # Shape: (T, num_assets)
        self.num_samples = len(returns_df)
        num_assets = self.returns_matrix.shape[1]
        
        # Calculate rolling 60-day empirical volatility & covariance
        self.rolling_mu = returns_df.rolling(60).mean().bfill().values * 252.0
        self.rolling_cov = []
        for i in range(self.num_samples):
            window = returns_df.iloc[max(0, i-60):i+1]
            cov = window.cov().fillna(1e-4).values * 252.0
            self.rolling_cov.append(cov)
            
        super(HistoricalSP500Env, self).__init__(
            num_assets=num_assets,
            initial_wealth=initial_wealth,
            r=r,
            dt=dt,
            horizon=self.num_samples - 1,
            gamma=gamma,
            max_drawdown=max_drawdown,
            transaction_cost=transaction_cost
        )

    def _get_env_state_dim(self):
        return self.num_assets

    def _reset_env_state(self):
        self.current_step = 0
        mu_0 = self.rolling_mu[0]
        return mu_0.astype(np.float32)

    def step(self, action):
        u_t = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Proportional transaction cost
        turnover = np.sum(np.abs(u_t - self.prev_u))
        tc_fee = self.transaction_cost * turnover * self.W_t
        self.prev_u = u_t.copy()
        
        self.W_t = max(self.W_t - tc_fee, 1e-6)
        
        # Real historical return at current step
        daily_ret = self.returns_matrix[self.current_step]
        portfolio_ret = np.dot(u_t, daily_ret) + (1.0 - np.sum(u_t)) * (self.r * self.dt)
        
        dW_t = self.W_t * portfolio_ret
        self.W_t = max(self.W_t + dW_t, 1e-6)
        
        self.H_t = max(self.H_t, self.W_t)
        D_t = 1.0 - (self.W_t / self.H_t)
        
        if self.gamma == 1.0:
            reward = np.log(self.W_t / (self.W_t - dW_t + 1e-12))
        else:
            reward = (self.W_t**(1.0 - self.gamma) - 1.0) / (1.0 - self.gamma) * self.dt
            
        self.current_step += 1
        terminated = (self.current_step >= self.horizon) or (self.W_t <= 1e-4)
        truncated = False
        
        drawdown_violation = (D_t > self.max_drawdown)
        
        mu_t = self.rolling_mu[min(self.current_step, self.num_samples-1)]
        cov_t = self.rolling_cov[min(self.current_step, self.num_samples-1)]
        sigma_t = np.linalg.cholesky(cov_t + 1e-6 * np.eye(self.num_assets))
        
        obs = self._get_obs(mu_t.astype(np.float32))
        info = {
            "wealth": self.W_t,
            "high_water_mark": self.H_t,
            "drawdown": D_t,
            "drawdown_violated": drawdown_violation,
            "mu": mu_t.astype(np.float32),
            "sigma": sigma_t.astype(np.float32),
            "r": self.r,
            "date": self.dates[min(self.current_step, self.num_samples-1)]
        }
        
        return obs, reward, terminated, truncated, info
