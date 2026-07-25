import numpy as np
from .base_env import BasePortfolioEnv

class JumpDiffusionCrashEnv(BasePortfolioEnv):
    """
    Merton Jump-Diffusion Market Crash Stress Environment.
    
    Models discrete Poisson market crashes (Black-Swan events):
        dS_t / S_t = mu dt + sigma dB_t + J_t dN_t
    where:
        dN_t ~ Poisson(lambda_jump * dt)
        J_t ~ N(mu_jump, sigma_jump^2) represents negative price shock (e.g., -15% crash).
    """
    def __init__(
        self,
        num_assets=1,
        mu=0.10,
        sigma=0.25,
        lambda_jump=2.0,  # Expected 2 market crash jumps per year
        mu_jump=-0.15,    # Average -15% crash per jump
        sigma_jump=0.05,
        r=0.02,
        dt=1.0/252.0,
        horizon=252,
        gamma=2.0,
        max_drawdown=0.20,
        initial_wealth=1.0,
        transaction_cost=0.001
    ):
        self.mu_asset = np.full(num_assets, mu)
        self.sigma_asset = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        
        super(JumpDiffusionCrashEnv, self).__init__(
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
        return self.mu_asset.copy().astype(np.float32)

    def _step_env_state(self, u_t):
        # Simulate Poisson jump events
        num_jumps = self.np_random.poisson(self.lambda_jump * self.dt)
        jump_shock = 0.0
        if num_jumps > 0:
            jump_shock = np.sum(self.np_random.normal(self.mu_jump, self.sigma_jump, size=num_jumps))
            
        # Total effective return includes jump component
        effective_mu = self.mu_asset + (jump_shock / self.dt)
        sigma_t = np.eye(self.num_assets) * self.sigma_asset
        
        return effective_mu.astype(np.float32), sigma_t.astype(np.float32), effective_mu.astype(np.float32)
