import torch
import torch.nn as nn
import torch.nn.functional as F

class EquivariantLinear(nn.Module):
    """
    S_N Permutation Equivariant Linear Layer.
    
    Transforms input per-asset state X in R^{B x N x D_in} to R^{B x N x D_out} via:
        Y_i = W_self * X_i + W_mean * (1/N sum_j X_j) + W_global * G + bias
    """
    def __init__(self, in_features, out_features, global_dim=3):
        super(EquivariantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W_self = nn.Linear(in_features, out_features, bias=False)
        self.W_mean = nn.Linear(in_features, out_features, bias=False)
        self.W_global = nn.Linear(global_dim, out_features, bias=True)

    def forward(self, x_assets, x_global):
        """
        Args:
            x_assets (Tensor): Shape (Batch, N, in_features)
            x_global (Tensor): Shape (Batch, global_dim)
        """
        batch_size, num_assets, _ = x_assets.shape
        
        self_part = self.W_self(x_assets)  # (Batch, N, out_features)
        mean_part = self.W_mean(x_assets.mean(dim=1, keepdim=True))  # (Batch, 1, out_features)
        global_part = self.W_global(x_global).unsqueeze(1)  # (Batch, 1, out_features)
        
        out = self_part + mean_part + global_part
        return out


class EquivariantPolicy(nn.Module):
    """
    Lie-Group / Permutation Equivariant Neural Policy Network.
    
    Ensures pi_theta(P * x_t) = P * pi_theta(x_t) for any permutation matrix P in S_N.
    """
    def __init__(self, num_assets=5, asset_dim=1, global_dim=3, hidden_dim=64):
        super(EquivariantPolicy, self).__init__()
        self.num_assets = num_assets
        self.asset_dim = asset_dim
        self.global_dim = global_dim
        
        self.eq_layer1 = EquivariantLinear(asset_dim, hidden_dim, global_dim)
        self.eq_layer2 = EquivariantLinear(hidden_dim, hidden_dim, global_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        Args:
            obs (Tensor): Shape (Batch, 3 + num_assets * asset_dim) or (Batch, 3 + num_assets)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        batch_size = obs.shape[0]
        x_global = obs[:, :self.global_dim]  # W_t, H_t, D_t
        x_assets_flat = obs[:, self.global_dim:]
        
        x_assets = x_assets_flat.view(batch_size, self.num_assets, self.asset_dim)
        
        h1 = F.relu(self.eq_layer1(x_assets, x_global))
        h2 = F.relu(self.eq_layer2(h1, x_global))
        
        u_tilde = self.out_layer(h2).squeeze(-1)  # Shape: (Batch, N)
        return u_tilde
