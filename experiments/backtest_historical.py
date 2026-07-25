import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.historical_sp500 import HistoricalSP500Env
from src.agents.ecblpo_agent import ECBLPOAgent
from src.agents.baselines import MertonAnalyticalAgent, UnconstrainedPPOAgent, ActionClippingAgent, PPOLagrangianAgent
from src.agents.sac_agent import SACAgent

def run_historical_10yr_backtest(max_drawdown=0.20):
    print("==================================================================")
    print("  RUNNING 10+ YEAR HISTORICAL S&P 500 MARKET CRASH BACKTEST (2014-2024)")
    print(f"  Target Hard Drawdown Limit alpha = {max_drawdown*100:.1f}%")
    print("==================================================================")
    
    env = HistoricalSP500Env(max_drawdown=max_drawdown)
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
    
    backtest_results = {}
    
    for agent_name, agent in agents.items():
        obs, info = env.reset()
        
        wealth_history = [info['wealth']]
        drawdown_history = [info['drawdown']]
        dates_history = [env.dates[0]]
        violation_count = 0
        returns = []
        turnovers = []
        prev_u = np.zeros(num_assets)
        
        for t in range(env.horizon):
            if agent_name == "Merton (Analytical)":
                action = agent.select_action(obs, info)
            elif agent_name == "E-CBLPO (Ours)":
                _, action, _ = agent.select_action(obs, info, eval_mode=True)
            elif agent_name == "Soft Actor-Critic (SAC)":
                action = agent.select_action(obs, eval_mode=True)
            else:
                action = agent.select_action(obs)
                
            turnover = np.sum(np.abs(action - prev_u))
            turnovers.append(turnover)
            prev_u = action.copy()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            wealth_history.append(info['wealth'])
            drawdown_history.append(info['drawdown'])
            dates_history.append(info['date'])
            returns.append(reward)
            
            if info['drawdown_violated']:
                violation_count += 1
                
            obs = next_obs
            if terminated or truncated:
                break
                
        w_arr = np.array(wealth_history)
        dd_arr = np.array(drawdown_history) * 100.0
        ret_arr = np.array(returns)
        
        final_wealth = w_arr[-1]
        total_return = (final_wealth - 1.0) * 100.0
        num_years = len(w_arr) / 252.0
        annualized_return = ((final_wealth)**(1.0 / num_years) - 1.0) * 100.0
        
        max_dd = np.max(dd_arr)
        violation_rate = (violation_count / len(w_arr)) * 100.0
        
        std_ret = np.std(ret_arr) + 1e-8
        sharpe = (np.mean(ret_arr) - env.r * env.dt) / std_ret * np.sqrt(252)
        
        neg_rets = ret_arr[ret_arr < 0]
        downside_std = np.std(neg_rets) + 1e-8 if len(neg_rets) > 0 else 1e-8
        sortino = (np.mean(ret_arr) - env.r * env.dt) / downside_std * np.sqrt(252)
        
        calmar = (annualized_return / 100.0) / (max_dd / 100.0 + 1e-6)
        cvar_05 = np.percentile(w_arr, 5)
        mean_turnover = np.mean(turnovers)
        
        backtest_results[agent_name] = {
            "final_wealth": final_wealth,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_dd,
            "violation_days": violation_count,
            "violation_rate": violation_rate,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "cvar_05": cvar_05,
            "turnover": mean_turnover,
            "wealth_history": wealth_history,
            "drawdown_history": drawdown_history,
            "dates_history": dates_history
        }
        
        print(f"\n--- {agent_name} ---")
        print(f"  Final Wealth W_T:     {final_wealth:.4f} (Total Return: {total_return:+.2f}%)")
        print(f"  Annualized Return:    {annualized_return:.2f}%")
        print(f"  Max Drawdown:         {max_dd:.2f}%")
        print(f"  Violation Days:       {violation_count} ({violation_rate:.2f}%)")
        print(f"  Sharpe / Sortino:     {sharpe:.4f} / {sortino:.4f}")
        print(f"  Calmar Ratio:         {calmar:.4f}")
        print(f"  CVaR (5% Tail W_T):  {cvar_05:.4f}")
        
    return backtest_results

if __name__ == "__main__":
    run_historical_10yr_backtest()
