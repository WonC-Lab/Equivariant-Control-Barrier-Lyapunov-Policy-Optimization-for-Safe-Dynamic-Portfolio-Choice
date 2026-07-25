import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class SACCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(SACCritic, self).__init__()
        # Q1
        self.q1_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2


class SACAgent:
    """
    Soft Actor-Critic (SAC) Baseline Agent.
    """
    def __init__(self, obs_dim, action_dim, lr=3e-4):
        self.actor = SACActor(obs_dim, action_dim)
        self.critic = SACCritic(obs_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, obs, eval_mode=False):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            
        with torch.no_grad():
            mu, log_std = self.actor(obs_t)
            if eval_mode:
                action = torch.tanh(mu)
            else:
                std = torch.exp(log_std)
                dist = Normal(mu, std)
                action = torch.tanh(dist.sample())
                
        return action.squeeze(0).numpy()
