import os
import sys
import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.kim_omberg import KimOmbergEnv
from src.environments.multi_asset_heston import MultiAssetHestonEnv
from src.environments.jump_diffusion_crash import JumpDiffusionCrashEnv

from src.agents.ecblpo_agent import ECBLPOAgent
from src.agents.baselines import MertonAnalyticalAgent, UnconstrainedPPOAgent, ActionClippingAgent, PPOLagrangianAgent
from src.agents.sac_agent import SACAgent

def run_multi_seed_evaluation(env_type="kim_omberg", num_seeds=30, max_drawdown=0.20, horizon=252):
    print(f"\n==================================================================")
    print(f"  RUNNING RIGOROUS MULTI-SEED EXPERIMENTS: [{env_type.upper()}] (Seeds: {num_seeds})")
    print(f"  Target Max Drawdown Threshold alpha = {max_drawdown*100:.1f}%")
    print(f"==================================================================")
    
    if env_type == "kim_omberg":
        env = KimOmbergEnv(horizon=horizon, max_drawdown=max_drawdown)
    elif env_type == "heston":
        env = MultiAssetHestonEnv(num_assets=5, horizon=horizon, max_drawdown=max_drawdown)
    else:
        env = JumpDiffusionCrashEnv(horizon=horizon, max_drawdown=max_drawdown)
        
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
        wealth_final_list = []
        max_dd_list = []
        violation_flags = []
        sharpe_list = []
        calmar_list = []
        
        for seed in range(num_seeds):
            obs, info = env.reset(seed=1000 + seed)
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
                    
            w_final = info['wealth']
            wealth_final_list.append(w_final)
            max_dd_list.append(max_ep_dd)
            violation_flags.append(violated)
            
            # Standard financial Sharpe ratio using exact daily wealth returns
            w_hist_arr = np.array(info['wealth_history'], dtype=np.float64)
            daily_rets = (w_hist_arr[1:] - w_hist_arr[:-1]) / np.maximum(w_hist_arr[:-1], 1e-8)
            
            r_daily = env.r / 252.0
            excess_daily_rets = daily_rets - r_daily
            
            mean_excess = np.mean(excess_daily_rets)
            std_vol = np.std(daily_rets)
            
            if std_vol > 1e-6:
                sharpe = (mean_excess / std_vol) * np.sqrt(252.0)
            else:
                sharpe = 0.0
            sharpe_list.append(sharpe)
            
            annual_net_return = (w_final - 1.0) / (horizon / 252.0)
            max_dd_ratio = max(max_ep_dd, 0.001)
            calmar = annual_net_return / max_dd_ratio
            calmar_list.append(calmar)
            
        w_arr = np.array(wealth_final_list)
        dd_arr = np.array(max_dd_list) * 100.0
        
        mean_w = np.mean(w_arr)
        std_w = np.std(w_arr)
        sem_w = stats.sem(w_arr)
        ci_95_w = sem_w * stats.t.ppf((1 + 0.95) / 2.0, num_seeds - 1) if num_seeds > 1 else 0.0
        
        cvar_05 = np.mean(np.sort(w_arr)[:max(1, int(0.05 * num_seeds))])
        
        mean_dd = np.mean(dd_arr)
        violation_rate = np.mean(violation_flags) * 100.0
        
        mean_sharpe = np.mean(sharpe_list)
        mean_calmar = np.mean(calmar_list)
        
        results[agent_name] = {
            "mean_wealth": mean_w,
            "std_wealth": std_w,
            "ci_95_wealth": ci_95_w,
            "cvar_05": cvar_05,
            "mean_max_dd": mean_dd,
            "violation_rate": violation_rate,
            "mean_sharpe": mean_sharpe,
            "mean_calmar": mean_calmar
        }
        
        print(f"\n--- {agent_name} ---")
        print(f"  Final Wealth W_T:    {mean_w:.4f} ± {ci_95_w:.4f} (Std: {std_w:.4f})")
        print(f"  CVaR (5% Tail W_T): {cvar_05:.4f}")
        print(f"  Mean Max Drawdown:  {mean_dd:.2f}%")
        print(f"  Violation Rate (%): {violation_rate:.1f}%")
        print(f"  Sharpe Ratio:       {mean_sharpe:.4f}")
        print(f"  Calmar Ratio:       {mean_calmar:.4f}")
        
    return results

if __name__ == "__main__":
    for env in ["kim_omberg", "heston", "jump_crash"]:
        run_multi_seed_evaluation(env_type=env, num_seeds=20)
