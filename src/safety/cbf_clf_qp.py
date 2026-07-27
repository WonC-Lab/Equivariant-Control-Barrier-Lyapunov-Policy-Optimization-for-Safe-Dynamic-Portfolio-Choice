import numpy as np
import torch
from scipy.optimize import minimize

class CBFCLFQPFilter:
    """
    Control Barrier-Lyapunov Differentiable QP Safety Filter.
    
    Projects candidate action u_tilde onto the safe admissible control space U_safe.
    Guarantees:
        1. Stochastic CBF Drawdown Safety: W_t >= (1 - alpha) * H_t
        2. CLF Wealth Growth tracking with slack delta_v
        3. Leverage bounds [u_min, u_max]
    """
    def __init__(
        self,
        max_drawdown=0.20,
        gamma_cbf=2.0,
        kappa_risk=0.5,
        c_clf=0.5,
        p_slack=100.0,
        u_min=-0.5,
        u_max=2.0
    ):
        self.max_drawdown = max_drawdown
        self.gamma_cbf = gamma_cbf
        self.kappa_risk = kappa_risk
        self.c_clf = c_clf
        self.p_slack = p_slack
        self.u_min = u_min
        self.u_max = u_max

    def filter_action(self, u_tilde, state_dict):
        """
        Solves QP safety projection for a single sample.
        
        Args:
            u_tilde (np.ndarray): Unconstrained candidate action (num_assets,)
            state_dict (dict): Contains 'W_t', 'H_t', 'mu_t', 'sigma_t', 'r_t'
            
        Returns:
            u_safe (np.ndarray): Projected safe action (num_assets,)
            delta_v (float): CLF growth relaxation slack
        """
        W_t = state_dict['W_t']
        H_t = state_dict['H_t']
        mu_t = state_dict['mu_t']
        sigma_t = state_dict['sigma_t']
        r_t = state_dict['r_t']
        
        num_assets = len(u_tilde)
        excess_return = mu_t - r_t
        
        # Use a small safety buffer (0.8%) to account for discrete 1-day step sampling gap
        effective_alpha = self.max_drawdown - 0.008
        h_val = W_t - (1.0 - effective_alpha) * H_t
        
        # Define QP Objective: min_{u, delta_v} 0.5 * ||u - u_tilde||^2 + p * delta_v^2
        def objective(x):
            u = x[:num_assets]
            delta_v = x[num_assets]
            return 0.5 * np.sum((u - u_tilde)**2) + self.p_slack * (delta_v**2)

        def objective_grad(x):
            u = x[:num_assets]
            delta_v = x[num_assets]
            grad = np.zeros(num_assets + 1)
            grad[:num_assets] = u - u_tilde
            grad[num_assets] = 2.0 * self.p_slack * delta_v
            return grad

        # CBF Drawdown Inequality Constraint: cbf_ineq(x) >= 0
        # W_t * (r_t + u^\top excess_return) - 0.5 * kappa * W_t^2 * u^\top (sigma sigma^\top) u + gamma * h_val >= 0
        Sigma = sigma_t @ sigma_t.T if len(sigma_t.shape) > 1 else np.array([[sigma_t**2]])
        
        def cbf_constraint(x):
            u = x[:num_assets]
            drift = W_t * (r_t + np.dot(u, excess_return))
            diffusion_risk = 0.5 * self.kappa_risk * (W_t**2) * np.dot(u, Sigma @ u)
            return drift - diffusion_risk + self.gamma_cbf * h_val

        # CLF Growth Constraint: clf_ineq(x) >= 0
        # - ( - (u^\top excess_return + r_t - 0.5 * u^\top Sigma u) + c * ln(W_t) ) + delta_v >= 0
        def clf_constraint(x):
            u = x[:num_assets]
            delta_v = x[num_assets]
            growth_rate = r_t + np.dot(u, excess_return) - 0.5 * np.dot(u, Sigma @ u)
            return growth_rate + self.c_clf * np.log(max(W_t, 1e-6)) + delta_v

        # Discrete-Time 3.5-Sigma Stochastic Safety Boundary Constraint:
        # Prevents discrete step sampling overshoot under extreme 3.5-sigma daily market drops
        def discrete_safety_constraint(x):
            u = x[:num_assets]
            port_vol = np.sqrt(max(np.dot(u, Sigma @ u), 1e-12))
            max_allowed_vol = max(0.0, h_val) / (W_t * 3.5 * np.sqrt(1.0/252.0) + 1e-6)
            return max_allowed_vol - port_vol

        constraints = [
            {'type': 'ineq', 'fun': cbf_constraint},
            {'type': 'ineq', 'fun': clf_constraint},
            {'type': 'ineq', 'fun': discrete_safety_constraint}
        ]

        # Normalize per-asset bounds by num_assets so total portfolio leverage remains in [u_min, u_max]
        asset_u_min = self.u_min / float(num_assets)
        asset_u_max = self.u_max / float(num_assets)
        bounds = [(asset_u_min, asset_u_max) for _ in range(num_assets)] + [(0.0, None)]
        
        # Initial guess
        x0 = np.zeros(num_assets + 1)
        x0[:num_assets] = np.clip(u_tilde / float(num_assets), asset_u_min, asset_u_max)
        x0[num_assets] = 0.0

        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6, 'maxiter': 100}
        )

        if res.success:
            u_safe = res.x[:num_assets]
            delta_v = res.x[num_assets]
        else:
            # Fallback to zero allocation (cash position is provably safe when W_t > (1-alpha)H_t)
            u_safe = np.zeros(num_assets)
            delta_v = 0.0

        return u_safe, delta_v

    def filter_batch_torch(self, u_tilde_tensor, state_dicts):
        """
        PyTorch wrapper for batch safety projection.
        """
        device = u_tilde_tensor.device
        u_tilde_np = u_tilde_tensor.detach().cpu().numpy()
        batch_size = u_tilde_np.shape[0]
        
        u_safe_list = []
        for i in range(batch_size):
            u_safe, _ = self.filter_action(u_tilde_np[i], state_dicts[i])
            u_safe_list.append(u_safe)
            
        u_safe_tensor = torch.tensor(np.array(u_safe_list), dtype=torch.float32, device=device)
        # Straight-Through Estimator or residual connection for backprop
        return u_tilde_tensor + (u_safe_tensor - u_tilde_tensor).detach()
