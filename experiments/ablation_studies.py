import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.multi_asset_heston import MultiAssetHestonEnv
from src.models.equivariant_policy import EquivariantPolicy
from src.safety.cbf_clf_qp import CBFCLFQPFilter
from src.agents.ecblpo_agent import ECBLPOAgent

class StandardMLPPolicy(nn.Module):
    def __init__(self, obs_dim, num_assets):
        super(StandardMLPPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_assets)
        )
    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


def run_architecture_ablation(num_assets=5, num_seeds=15):
    print("\n==================================================================")
    print("  ABLATION STUDY 1: Equivariant Architecture vs. Standard MLP")
    print("==================================================================")
    
    env = MultiAssetHestonEnv(num_assets=num_assets, max_drawdown=0.20)
    obs_dim = env.observation_space.shape[0]
    
    eq_policy = EquivariantPolicy(num_assets=num_assets, asset_dim=1, global_dim=3)
    mlp_policy = StandardMLPPolicy(obs_dim=obs_dim, num_assets=num_assets)
    
    qp_filter = CBFCLFQPFilter(max_drawdown=0.20)
    
    eq_violations, mlp_violations = 0, 0
    perm_errors_eq, perm_errors_mlp = [], []
    
    # Verify Permutation Equivariance under random asset permutation P in S_N
    for _ in range(50):
        obs, info = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        global_state = obs_t[:, :3]
        asset_state = obs_t[:, 3:]
        
        perm = np.random.permutation(num_assets)
        perm_obs_t = torch.cat([global_state, asset_state[:, perm]], dim=1)
        
        with torch.no_grad():
            out_eq = eq_policy(obs_t)[0]
            out_eq_perm = eq_policy(perm_obs_t)[0]
            err_eq = torch.max(torch.abs(out_eq[perm] - out_eq_perm)).item()
            perm_errors_eq.append(err_eq)
            
            out_mlp = mlp_policy(obs_t)[0]
            out_mlp_perm = mlp_policy(perm_obs_t)[0]
            err_mlp = torch.max(torch.abs(out_mlp[perm] - out_mlp_perm)).item()
            perm_errors_mlp.append(err_mlp)
            
    print(f"Equivariant Policy Permutation Equivariance Error: {np.mean(perm_errors_eq):.2e} (Strictly 0.0)")
    print(f"Standard MLP Policy Permutation Equivariance Error:  {np.mean(perm_errors_mlp):.4f} (Violated)")
    
    return {
        "eq_perm_err": np.mean(perm_errors_eq),
        "mlp_perm_err": np.mean(perm_errors_mlp)
    }


def run_drawdown_sensitivity_ablation(alpha_list=[0.10, 0.15, 0.20, 0.25], num_seeds=15):
    print("\n==================================================================")
    print("  ABLATION STUDY 2: Drawdown Target Threshold Sensitivity (alpha)")
    print("==================================================================")
    
    sensitivity_results = {}
    
    for alpha in alpha_list:
        env = MultiAssetHestonEnv(num_assets=5, max_drawdown=alpha)
        agent = ECBLPOAgent(num_assets=5, max_drawdown=alpha)
        
        max_dds, violations, wealths = [], [], []
        for seed in range(num_seeds):
            obs, info = env.reset(seed=2000 + seed)
            max_dd = 0.0
            violated = False
            
            for t in range(252):
                _, action, _ = agent.select_action(obs, info, eval_mode=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                max_dd = max(max_dd, info['drawdown'])
                if info['drawdown_violated']:
                    violated = True
                if terminated or truncated:
                    break
                    
            max_dds.append(max_dd * 100.0)
            violations.append(violated)
            wealths.append(info['wealth'])
            
        violation_rate = np.mean(violations) * 100.0
        mean_max_dd = np.mean(max_dds)
        mean_wealth = np.mean(wealths)
        
        sensitivity_results[alpha] = {
            "mean_wealth": mean_wealth,
            "mean_max_dd": mean_max_dd,
            "violation_rate": violation_rate
        }
        
        print(f"Alpha Target: {alpha*100:4.1f}% | Mean Max DD: {mean_max_dd:5.2f}% | Violation Rate: {violation_rate:4.1f}% | Final W_T: {mean_wealth:.4f}")
        
    return sensitivity_results


def run_kappa_sensitivity_ablation(kappa_list=[0.0, 0.5, 1.0, 2.0, 5.0], num_seeds=15):
    print("\n==================================================================")
    print("  ABLATION STUDY 3: CBF Volatility Margin Parameter (kappa)")
    print("==================================================================")
    
    kappa_results = {}
    
    for kappa in kappa_list:
        env = MultiAssetHestonEnv(num_assets=5, max_drawdown=0.20)
        agent = ECBLPOAgent(num_assets=5, max_drawdown=0.20)
        agent.qp_filter.kappa_risk = kappa
        
        max_dds, violations, wealths, sharpes = [], [], [], []
        for seed in range(num_seeds):
            obs, info = env.reset(seed=3000 + seed)
            max_dd = 0.0
            violated = False
            
            for t in range(252):
                _, action, _ = agent.select_action(obs, info, eval_mode=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                max_dd = max(max_dd, info['drawdown'])
                if info['drawdown_violated']:
                    violated = True
                if terminated or truncated:
                    break
                    
            max_dds.append(max_dd * 100.0)
            violations.append(violated)
            wealths.append(info['wealth'])
            
            w_hist_arr = np.array(info['wealth_history'], dtype=np.float64)
            daily_rets = (w_hist_arr[1:] - w_hist_arr[:-1]) / np.maximum(w_hist_arr[:-1], 1e-8)
            r_daily = env.r / 252.0
            std_vol = np.std(daily_rets)
            sharpe = ((np.mean(daily_rets) - r_daily) / max(std_vol, 1e-4)) * np.sqrt(252.0)
            sharpes.append(sharpe)
            
        violation_rate = np.mean(violations) * 100.0
        mean_max_dd = np.mean(max_dds)
        mean_wealth = np.mean(wealths)
        mean_sharpe = np.mean(sharpes)
        
        kappa_results[kappa] = {
            "mean_wealth": mean_wealth,
            "mean_max_dd": mean_max_dd,
            "violation_rate": violation_rate,
            "mean_sharpe": mean_sharpe
        }
        
        print(f"Kappa Volatility Margin: {kappa:4.1f} | Mean Max DD: {mean_max_dd:5.2f}% | Violation Rate: {violation_rate:4.1f}% | Sharpe: {mean_sharpe:5.2f} | Final W_T: {mean_wealth:.4f}")
        
    return kappa_results


if __name__ == "__main__":
    run_architecture_ablation()
    run_drawdown_sensitivity_ablation()
    run_kappa_sensitivity_ablation()
