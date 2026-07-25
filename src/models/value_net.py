import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """
    Critic Network for state value estimation V_phi(x_t).
    
    Permutation Invariant: output is invariant to per-asset feature order.
    """
    def __init__(self, obs_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)
