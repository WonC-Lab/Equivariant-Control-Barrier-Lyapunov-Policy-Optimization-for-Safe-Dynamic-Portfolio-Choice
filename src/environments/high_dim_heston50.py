import numpy as np
from .base_env import BasePortfolioEnv

class HighDimHeston50Env(BasePortfolioEnv):
    """
    High-Dimensional Continuous-Time Portfolio Selection Environment (N = 50 Risky Assets).
    
    Models 50 risky assets under multi-factor Heston stochastic volatility:
        d v_{i,t} = kappa_v * (theta_v - v_{i,t}) * dt + sigma_v * sqrt(v_{i,t}) * dB_{i,t}^v
    Asset covariance matrix Sigma_t in R^{50 x 50} with dynamic cross-asset factor structure.
    """
    def __init__(
        self,
        num_assets=50,
        base_mu=0.09,
        kappa_v=2.5,
        theta_v=0.04,
        sigma_v=0.12,
        r=0.02,
        dt=1.0/252.0,
        horizon=252,
        gamma=2.0,
        max_drawdown=0.20,
        initial_wealth=1.0,
        transaction_cost=0.001
    ):
        self.kappa_v = kappa_v
        self.theta_v = theta_v
        self.sigma_v = sigma_v
        
        # 50 assets with heterogeneous expected returns (0.05 to 0.13)
        self.base_mu = np.linspace(0.05, 0.13, num_assets)
        
        # 5-Factor correlation structure for 50 assets
        factor_loadings = np.random.RandomState(42).normal(0, 0.3, size=(num_assets, 5))
        unfactored_cov = factor_loadings @ factor_loadings.T + 0.5 * np.eye(num_assets)
        stds = np.sqrt(np.diag(unfactored_cov))
        corr_matrix = unfactored_cov / np.outer(stds, stds)
        
        self.L_corr = np.linalg.cholesky(corr_matrix)
        self.v_t = np.full(num_assets, theta_v)
        
        super(HighDimHeston50Env, self).__init__(
            num_assets=num_assets,
            initial_wealth=initial_wealth,
            r=r,
            dt=dt,
            horizon=horizon,
            gamma=gamma,
            max_drawdown=max_drawdown,
            transaction_cost=transaction_cost
        )

    def _get_env_state_dim(self):
        return self.num_assets

    def _reset_env_state(self):
        self.v_t = np.full(self.num_assets, self.theta_v)
        return self.v_t.copy().astype(np.float32)

    def _step_env_state(self, u_t):
        # Update 50 stochastic variances via Feller-bounded Heston SDE
        dZv = self.np_random.normal(0, np.sqrt(self.dt), size=self.num_assets)
        
        for i in range(self.num_assets):
            v_curr = max(self.v_t[i], 1e-4)
            dv = self.kappa_v * (self.theta_v - v_curr) * self.dt + self.sigma_v * np.sqrt(v_curr) * dZv[i]
            self.v_t[i] = max(v_curr + dv, 1e-4)
            
        stds = np.sqrt(self.v_t)
        sigma_t = np.diag(stds) @ self.L_corr
        
        mu_t = self.base_mu + 0.1 * (self.v_t - self.theta_v)
        
        return mu_t.astype(np.float32), sigma_t.astype(np.float32), self.v_t.copy().astype(np.float32)
