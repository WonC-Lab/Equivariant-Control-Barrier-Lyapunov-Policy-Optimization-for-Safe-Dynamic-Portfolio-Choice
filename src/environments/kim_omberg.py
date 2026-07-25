import numpy as np
from .base_env import BasePortfolioEnv

class KimOmbergEnv(BasePortfolioEnv):
    """
    Kim-Omberg Continuous-Time Portfolio Selection Environment.
    
    Excess return theta_t follows a mean-reverting Ornstein-Uhlenbeck (OU) process:
        d theta_t = kappa * (bar_theta - theta_t) * dt + sigma_theta * dB_t^theta
    Expected return:
        mu_t = r + sigma * theta_t
    """
    def __init__(
        self,
        kappa=1.5,
        bar_theta=0.3,
        sigma_theta=0.15,
        sigma_asset=0.20,
        rho=-0.5,
        r=0.02,
        dt=1.0/252.0,
        horizon=252,
        gamma=2.0,
        max_drawdown=0.20,
        initial_wealth=1.0
    ):
        self.kappa = kappa
        self.bar_theta = bar_theta
        self.sigma_theta = sigma_theta
        self.sigma_asset = sigma_asset
        self.rho = rho
        
        self.theta_t = bar_theta
        
        super(KimOmbergEnv, self).__init__(
            num_assets=1,
            initial_wealth=initial_wealth,
            r=r,
            dt=dt,
            horizon=horizon,
            gamma=gamma,
            max_drawdown=max_drawdown
        )

    def _get_env_state_dim(self):
        return 1

    def _reset_env_state(self):
        self.theta_t = self.bar_theta + self.np_random.normal(0, 0.05)
        return np.array([self.theta_t], dtype=np.float32)

    def _step_env_state(self, u_t):
        # Correlated Brownian motions between asset noise and risk premium noise
        dB_asset = self.np_random.normal(0, np.sqrt(self.dt))
        dB_orthogonal = self.np_random.normal(0, np.sqrt(self.dt))
        dB_theta = self.rho * dB_asset + np.sqrt(1 - self.rho**2) * dB_orthogonal
        
        # Update theta_t via Euler-Maruyama
        dtheta = self.kappa * (self.bar_theta - self.theta_t) * self.dt + self.sigma_theta * dB_theta
        self.theta_t += dtheta
        
        mu_t = np.array([self.r + self.sigma_asset * self.theta_t], dtype=np.float32)
        sigma_t = np.array([[self.sigma_asset]], dtype=np.float32)
        
        return mu_t, sigma_t, np.array([self.theta_t], dtype=np.float32)
