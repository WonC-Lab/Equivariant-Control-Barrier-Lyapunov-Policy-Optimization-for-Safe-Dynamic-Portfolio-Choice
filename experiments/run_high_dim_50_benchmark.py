import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.high_dim_heston50 import HighDimHeston50Env
from src.agents.ecblpo_agent import ECBLPOAgent
from src.agents.baselines import MertonAnalyticalAgent, UnconstrainedPPOAgent, ActionClippingAgent, PPOLagrangianAgent
from src.agents.sac_agent import SACAgent

def run_50asset_benchmark(num_seeds=15, max_drawdown=0.20, horizon=252):
    print("==================================================================")
    print(f"  RUNNING HIGH-DIMENSIONAL BENCHMARK (N=50 RISKY ASSETS, {num_seeds} SEEDS)")
    print(f"  Target Hard Drawdown Limit alpha = {max_drawdown*100:.1f}%")
    print("==================================================================")
    
    env = HighDimHeston50Env(num_assets=50, horizon=horizon, max_drawdown=max_drawdown)
    num_assets = env.num_assets
    obs_dim = env.observation_space.shape[0]
    
    agents = {
        "Merton (Analytical)": MertonAnalyticalAgent(num_assets=num_assets),
        "Unconstrained PPO": UnconstrainedPPOAgent(obs_dim=obs_dim, num_assets=num_assets),
        "Soft Actor-Critic (SAC)": SACAgent(obs_dim=obs_dim, action_dim=num_assets),
        "Action Clipping": ActionClippingAgent(obs_dim=obs_dim, num_assets=num_assets),
        "PPO-Lagrangian": PPOLagrangianAgent(obs_dim=obs_dim, num_assets=num_assets, max_drawdown=max_drawdown),
        "E-CBLPO (Ours)": ECBLPOAgent(num_assets=num_assets, max_drawdown=max_drawdown)
    }
    
    results = {}
    
    for agent_name, agent in agents.items():
        wealths, max_dds, violations, sharpes = [], [], [], []
        
        for seed in range(num_seeds):
            obs, info = env.reset(seed=3000 + seed)
            max_ep_dd = 0.0
            violated = False
            returns = []
            
            for t in range(horizon):
                if agent_name == "Merton (Analytical)":
                    action = agent.select_action(obs, info)
                elif agent_name == "E-CBLPO (Ours)":
                    _, action, _ = agent.select_action(obs, info, eval_mode=True)
                elif agent_name == "Soft Actor-Critic (SAC)":
                    action = agent.select_action(obs, eval_mode=True)
                else:
                    action = agent.select_action(obs)
                    
                next_obs, reward, terminated, truncated, info = env.step(action)
                returns.append(reward)
                
                max_ep_dd = max(max_ep_dd, info['drawdown'])
                if info['drawdown_violated']:
                    violated = True
                    
                obs = next_obs
                if terminated or truncated:
                    break
                    
            wealths.append(info['wealth'])
            max_dds.append(max_ep_dd * 100.0)
            violations.append(violated)
            
            arr_ret = np.array(returns)
            std_ret = np.std(arr_ret) + 1e-8
            sharpe = (np.mean(arr_ret) - env.r * env.dt) / std_ret * np.sqrt(252)
            sharpes.append(sharpe)
            
        mean_w = np.mean(wealths)
        mean_dd = np.mean(max_dds)
        violation_rate = np.mean(violations) * 100.0
        mean_sharpe = np.mean(sharpes)
        
        results[agent_name] = {
            "mean_wealth": mean_w,
            "mean_max_dd": mean_dd,
            "violation_rate": violation_rate,
            "mean_sharpe": mean_sharpe
        }
        
        print(f"\n--- {agent_name} (N=50 Assets) ---")
        print(f"  Final Wealth W_T:    {mean_w:.4f}")
        print(f"  Mean Max Drawdown:  {mean_dd:.2f}%")
        print(f"  Violation Rate (%): {violation_rate:.1f}%")
        print(f"  Sharpe Ratio:       {mean_sharpe:.4f}")
        
    return results

if __name__ == "__main__":
    run_50asset_benchmark(num_seeds=10)
