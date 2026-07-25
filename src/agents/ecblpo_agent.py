import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from ..models.equivariant_policy import EquivariantPolicy
from ..models.value_net import ValueNetwork
from ..safety.cbf_clf_qp import CBFCLFQPFilter

class ECBLPOAgent:
    """
    Equivariant Control Barrier-Lyapunov Policy Optimization (E-CBLPO) Agent.
    
    Combines:
        1. S_N Equivariant Actor Network
        2. Permutation Invariant Value Critic
        3. Differentiable CBF-CLF QP Safety Projection Layer
    """
    def __init__(
        self,
        num_assets=5,
        asset_dim=1,
        global_dim=3,
        max_drawdown=0.20,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2
    ):
        self.num_assets = num_assets
        self.max_drawdown = max_drawdown
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        
        obs_dim = global_dim + num_assets * asset_dim
        
        self.actor = EquivariantPolicy(num_assets, asset_dim, global_dim)
        self.critic = ValueNetwork(obs_dim)
        
        self.log_std = nn.Parameter(torch.full((num_assets,), -0.5))
        
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + [self.log_std], lr=lr_actor
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.qp_filter = CBFCLFQPFilter(max_drawdown=max_drawdown)

    def select_action(self, obs, info, eval_mode=False):
        """
        Returns unconstrained action u_tilde and QP-projected safe action u_safe.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            mean = self.actor(obs_t).squeeze(0)
            if eval_mode:
                u_tilde = mean.numpy()
            else:
                std = torch.exp(self.log_std)
                dist = Normal(mean, std)
                u_tilde = dist.sample().numpy()
                
        # State metadata for CBF-CLF QP filter
        state_dict = {
            'W_t': info.get('wealth', obs[0]),
            'H_t': info.get('high_water_mark', obs[1]),
            'mu_t': info.get('mu', np.full(self.num_assets, 0.08)),
            'sigma_t': info.get('sigma', np.eye(self.num_assets) * 0.20),
            'r_t': info.get('r', 0.02)
        }
        
        u_safe, delta_v = self.qp_filter.filter_action(u_tilde, state_dict)
        return u_tilde, u_safe, delta_v

    def update(self, rollouts):
        """
        PPO Policy & Value function update step using rollouts buffer.
        """
        states = torch.tensor(np.array(rollouts['states']), dtype=torch.float32)
        actions = torch.tensor(np.array(rollouts['actions']), dtype=torch.float32)
        rewards = torch.tensor(np.array(rollouts['rewards']), dtype=torch.float32)
        next_states = torch.tensor(np.array(rollouts['next_states']), dtype=torch.float32)
        dones = torch.tensor(np.array(rollouts['dones']), dtype=torch.float32)
        
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            # GAE Advantage estimation
            deltas = rewards + self.gamma * next_values * (1.0 - dones) - values
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy update
        for _ in range(5):
            means = self.actor(states)
            stds = torch.exp(self.log_std)
            dists = Normal(means, stds)
            
            log_probs = dists.log_prob(actions).sum(dim=-1)
            
            with torch.no_grad():
                old_means = means
                old_dists = Normal(old_means, stds)
                old_log_probs = old_dists.log_prob(actions).sum(dim=-1)
                
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        # Value critic update
        for _ in range(5):
            val_preds = self.critic(states)
            critic_loss = nn.MSELoss()(val_preds, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
        return actor_loss.item(), critic_loss.item()
