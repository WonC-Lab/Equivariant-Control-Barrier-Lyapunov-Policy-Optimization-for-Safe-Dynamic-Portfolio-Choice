import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.kim_omberg import KimOmbergEnv
from src.environments.multi_asset_heston import MultiAssetHestonEnv
from src.agents.ecblpo_agent import ECBLPOAgent
from src.agents.baselines import MertonAnalyticalAgent, UnconstrainedPPOAgent, PPOLagrangianAgent

def run_evaluation(env_name="kim_omberg", episodes=20, horizon=252, max_drawdown=0.20):
    print(f"==================================================")
    print(f" Running Benchmark Evaluation on [{env_name.upper()}]")
    print(f" Target Max Drawdown Limit: {max_drawdown*100:.1f}%")
    print(f"==================================================")
    
    if env_name == "kim_omberg":
        env = KimOmbergEnv(horizon=horizon, max_drawdown=max_drawdown)
    else:
        env = MultiAssetHestonEnv(num_assets=5, horizon=horizon, max_drawdown=max_drawdown)
        
    num_assets = env.num_assets
    obs_dim = env.observation_space.shape[0]
    
    agents = {
        "Merton (Analytical)": MertonAnalyticalAgent(num_assets=num_assets),
        "Unconstrained PPO": UnconstrainedPPOAgent(obs_dim=obs_dim, num_assets=num_assets),
        "PPO-Lagrangian": PPOLagrangianAgent(obs_dim=obs_dim, num_assets=num_assets, max_drawdown=max_drawdown),
        "E-CBLPO (Ours)": ECBLPOAgent(num_assets=num_assets, max_drawdown=max_drawdown)
    }
    
    results = {}
    
    for agent_name, agent in agents.items():
        wealth_trajectories = []
        drawdown_trajectories = []
        violations = []
        sharpe_ratios = []
        
        for ep in range(episodes):
            obs, info = env.reset(seed=100 + ep)
            wealth_hist = [info['wealth']]
            drawdown_hist = [info['drawdown']]
            violation_count = 0
            returns = []
            
            for t in range(horizon):
                if agent_name == "Merton (Analytical)":
                    action = agent.select_action(obs, info)
                elif agent_name == "E-CBLPO (Ours)":
                    _, action, _ = agent.select_action(obs, info, eval_mode=True)
                else:
                    action = agent.select_action(obs)
                    
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                wealth_hist.append(info['wealth'])
                drawdown_hist.append(info['drawdown'])
                returns.append(reward)
                if info['drawdown_violated']:
                    violation_count += 1
                    
                obs = next_obs
                if terminated or truncated:
                    break
                    
            wealth_trajectories.append(wealth_hist)
            drawdown_trajectories.append(drawdown_hist)
            violations.append(violation_count > 0)
            
            arr_ret = np.array(returns)
            std_ret = np.std(arr_ret) + 1e-8
            sharpe = (np.mean(arr_ret) - env.r * env.dt) / std_ret * np.sqrt(252)
            sharpe_ratios.append(sharpe)
            
        violation_rate = np.mean(violations) * 100.0
        avg_final_wealth = np.mean([w[-1] for w in wealth_trajectories])
        avg_max_dd = np.mean([np.max(dd) for dd in drawdown_trajectories]) * 100.0
        avg_sharpe = np.mean(sharpe_ratios)
        
        results[agent_name] = {
            "final_wealth": avg_final_wealth,
            "max_drawdown": avg_max_dd,
            "violation_rate": violation_rate,
            "sharpe_ratio": avg_sharpe,
            "wealth_trajectories": wealth_trajectories,
            "drawdown_trajectories": drawdown_trajectories
        }
        
        print(f"\n--- {agent_name} ---")
        print(f"  Final Wealth:       {avg_final_wealth:.4f}")
        print(f"  Max Drawdown:       {avg_max_dd:.2f}%")
        print(f"  DD Violation Rate:  {violation_rate:.1f}%")
        print(f"  Sharpe Ratio:       {avg_sharpe:.4f}")
        
    return results

if __name__ == "__main__":
    run_evaluation(env_name="kim_omberg", episodes=10)
