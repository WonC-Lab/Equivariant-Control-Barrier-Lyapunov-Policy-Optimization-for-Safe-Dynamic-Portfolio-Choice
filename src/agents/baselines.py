import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class MertonAnalyticalAgent:
    """
    Classical Merton Optimal Portfolio Agent (Unconstrained Baseline).
    pi^* = (1 / gamma) * Sigma^{-1} * (mu - r * 1)
    """
    def __init__(self, num_assets=1, gamma=2.0, r=0.02):
        self.num_assets = num_assets
        self.gamma = gamma
        self.r = r

    def select_action(self, obs, info):
        mu = info.get('mu', np.full(self.num_assets, 0.08))
        sigma = info.get('sigma', np.eye(self.num_assets) * 0.20)
        
        excess_return = mu - self.r
        if self.num_assets == 1:
            sigma_val = sigma[0, 0] if isinstance(sigma, np.ndarray) else sigma
            u_opt = (1.0 / self.gamma) * (excess_return[0] / (sigma_val**2 + 1e-8))
            return np.array([u_opt], dtype=np.float32)
        else:
            Sigma = sigma @ sigma.T
            u_opt = (1.0 / self.gamma) * np.linalg.solve(Sigma + 1e-6 * np.eye(self.num_assets), excess_return)
            return u_opt.astype(np.float32)


class StandardActorCritic(nn.Module):
    def __init__(self, obs_dim, num_assets):
        super(StandardActorCritic, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_assets)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(num_assets))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd)
        value = self.critic(obs).squeeze(-1)
        return Normal(mean, std), value


class UnconstrainedPPOAgent:
    """
    Unconstrained PPO Agent.
    """
    def __init__(self, obs_dim, num_assets, lr=3e-4, gamma=0.99, gae_lambda=0.95):
        self.ac = StandardActorCritic(obs_dim, num_assets)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, _ = self.ac(obs_t)
            action = dist.sample()
        return action.numpy()


class ActionClippingAgent:
    """
    Naive Action-Clipping Baseline:
    Clips action to leverage bounds [u_min, u_max] without CBF safety projection.
    """
    def __init__(self, obs_dim, num_assets, u_min=-0.5, u_max=2.0):
        self.ac = StandardActorCritic(obs_dim, num_assets)
        self.u_min = u_min
        self.u_max = u_max

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, _ = self.ac(obs_t)
            action = dist.sample().numpy()
        return np.clip(action, self.u_min, self.u_max)


class PPOLagrangianAgent:
    """
    PPO-Lagrangian Agent for Safe RL.
    Uses an adaptive Lagrange multiplier lambda_lag to penalize drawdown violations:
        L(theta, lambda) = R_t - lambda_lag * max(0, D_t - alpha)
    """
    def __init__(self, obs_dim, num_assets, max_drawdown=0.20, lr=3e-4, lr_lag=1e-2):
        self.ac = StandardActorCritic(obs_dim, num_assets)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.log_lambda_lag = torch.nn.Parameter(torch.tensor(0.0))
        self.lag_optimizer = optim.Adam([self.log_lambda_lag], lr=lr_lag)
        self.max_drawdown = max_drawdown

    @property
    def lambda_lag(self):
        return torch.exp(self.log_lambda_lag).item()

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, _ = self.ac(obs_t)
            action = dist.sample()
        return action.numpy()
