import numpy as np
from .base_env import BasePortfolioEnv

class MultiAssetHestonEnv(BasePortfolioEnv):
    """
    Multi-Asset Stochastic Volatility (Heston) Environment.
    
    Models N risky assets with mean-reverting variance v_{i,t}:
        d v_{i,t} = kappa_v * (theta_v - v_{i,t}) * dt + sigma_v * sqrt(v_{i,t}) * dB_{i,t}^v
    Asset volatility matrix Sigma_t = diag(sqrt(v_{1,t}), ..., sqrt(v_{N,t})) * CorrelationMatrix.
    """
    def __init__(
        self,
        num_assets=5,
        base_mu=0.08,
        kappa_v=2.0,
        theta_v=0.04,
        sigma_v=0.1,
        rho_av=-0.7,
        r=0.02,
        dt=1.0/252.0,
        horizon=252,
        gamma=2.0,
        max_drawdown=0.15,
        initial_wealth=1.0
    ):
        self.kappa_v = kappa_v
        self.theta_v = theta_v
        self.sigma_v = sigma_v
        self.rho_av = rho_av
        self.base_mu = np.full(num_assets, base_mu)
        
        # Setup cross-asset correlation matrix
        corr = 0.3 * np.ones((num_assets, num_assets)) + 0.7 * np.eye(num_assets)
        self.L_corr = np.linalg.cholesky(corr)
        
        self.v_t = np.full(num_assets, theta_v)
        
        super(MultiAssetHestonEnv, self).__init__(
            num_assets=num_assets,
            initial_wealth=initial_wealth,
            r=r,
            dt=dt,
            horizon=horizon,
            gamma=gamma,
            max_drawdown=max_drawdown
        )

    def _get_env_state_dim(self):
        return self.num_assets

    def _reset_env_state(self):
        self.v_t = np.full(self.num_assets, self.theta_v)
        return self.v_t.copy().astype(np.float32)

    def _step_env_state(self, u_t):
        # Update stochastic variances via Feller-bounded Heston SDE
        dZv = self.np_random.normal(0, np.sqrt(self.dt), size=self.num_assets)
        
        for i in range(self.num_assets):
            v_curr = max(self.v_t[i], 1e-4)
            dv = self.kappa_v * (self.theta_v - v_curr) * self.dt + self.sigma_v * np.sqrt(v_curr) * dZv[i]
            self.v_t[i] = max(v_curr + dv, 1e-4)
            
        stds = np.sqrt(self.v_t)
        sigma_t = np.diag(stds) @ self.L_corr
        
        mu_t = self.base_mu + 0.1 * (self.v_t - self.theta_v)
        
        return mu_t.astype(np.float32), sigma_t.astype(np.float32), self.v_t.copy().astype(np.float32)
