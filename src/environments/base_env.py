import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BasePortfolioEnv(gym.Env):
    """
    Base Continuous-Time Portfolio Selection Environment.
    
    Tracks:
        - Wealth W_t
        - High-Water Mark H_t = max_{s <= t} W_s
        - Instantaneous Drawdown D_t = 1 - W_t / H_t
        - Proportional Transaction Cost transaction_cost (e.g. 10 bps = 0.001)
    """
    def __init__(
        self,
        num_assets=1,
        initial_wealth=1.0,
        r=0.02,
        dt=1.0/252.0,
        horizon=252,
        gamma=2.0,
        max_drawdown=0.20,
        transaction_cost=0.001
    ):
        super(BasePortfolioEnv, self).__init__()
        
        self.num_assets = num_assets
        self.initial_wealth = initial_wealth
        self.r = r
        self.dt = dt
        self.horizon = horizon
        self.gamma = gamma
        self.max_drawdown = max_drawdown
        self.transaction_cost = transaction_cost
        
        self.current_step = 0
        self.W_t = initial_wealth
        self.H_t = initial_wealth
        self.prev_u = np.zeros(num_assets)
        
        self.action_space = spaces.Box(
            low=-0.5 / float(num_assets), high=2.0 / float(num_assets), shape=(num_assets,), dtype=np.float32
        )
        
        obs_dim = 3 + self._get_env_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_env_state_dim(self):
        return self.num_assets

    def _get_obs(self, env_state):
        D_t = 1.0 - (self.W_t / self.H_t)
        obs = np.concatenate([[self.W_t, self.H_t, D_t], env_state], axis=0).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.W_t = self.initial_wealth
        self.H_t = self.initial_wealth
        self.prev_u = np.zeros(self.num_assets)
        self.wealth_history = [float(self.initial_wealth)]
        
        env_state = self._reset_env_state()
        obs = self._get_obs(env_state)
        info = {
            "wealth": self.W_t,
            "high_water_mark": self.H_t,
            "drawdown": 0.0,
            "drawdown_violated": False,
            "wealth_history": np.array(self.wealth_history, dtype=np.float32)
        }
        return obs, info

    def step(self, action):
        u_t = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Calculate proportional transaction cost penalty: tc_fee = lambda * ||u_t - u_{t-1}||_1
        turnover = np.sum(np.abs(u_t - self.prev_u))
        tc_fee = self.transaction_cost * turnover * self.W_t
        self.prev_u = u_t.copy()
        
        # Apply transaction cost deduction to wealth before asset diffusion
        self.W_t = max(self.W_t - tc_fee, 1e-6)
        
        mu_t, sigma_t, next_env_state = self._step_env_state(u_t)
        
        excess_return = mu_t - self.r
        expected_drift = self.r + np.dot(u_t, excess_return)
        
        m = sigma_t.shape[1] if len(sigma_t.shape) > 1 else 1
        dB_t = self.np_random.normal(0, np.sqrt(self.dt), size=(m,))
        
        diffusion = np.dot(u_t, np.dot(sigma_t, dB_t)) if len(sigma_t.shape) > 1 else u_t[0] * sigma_t * dB_t[0]
        
        dW_t = self.W_t * (expected_drift * self.dt + diffusion)
        self.W_t = max(self.W_t + dW_t, 1e-6)
        
        self.H_t = max(self.H_t, self.W_t)
        D_t = 1.0 - (self.W_t / self.H_t)
        self.wealth_history.append(float(self.W_t))
        
        if self.gamma == 1.0:
            reward = np.log(self.W_t / (self.W_t - dW_t + 1e-12))
        else:
            reward = (self.W_t**(1.0 - self.gamma) - 1.0) / (1.0 - self.gamma) * self.dt
            
        self.current_step += 1
        terminated = (self.current_step >= self.horizon) or (self.W_t <= 1e-4)
        truncated = False
        
        drawdown_violation = (D_t > self.max_drawdown)
        
        obs = self._get_obs(next_env_state)
        info = {
            "wealth": self.W_t,
            "high_water_mark": self.H_t,
            "drawdown": D_t,
            "drawdown_violated": drawdown_violation,
            "mu": mu_t,
            "sigma": sigma_t,
            "r": self.r,
            "wealth_history": np.array(self.wealth_history, dtype=np.float32)
        }
        
        return obs, reward, terminated, truncated, info

    def _reset_env_state(self):
        raise NotImplementedError

    def _step_env_state(self, u_t):
        raise NotImplementedError
