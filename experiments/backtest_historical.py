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
        pct_returns = []
        turnovers = []
        prev_u = np.zeros(num_assets)
        
        for t in range(env.horizon):
            w_prev = info['wealth']
            
            if agent_name == "E-CBLPO (Ours)":
                mu_t = env.rolling_mu[t]
                cov_t = env.rolling_cov[t]
                excess = mu_t - env.r
                u_merton = np.linalg.solve(cov_t + 1e-4*np.eye(num_assets), excess) / 2.0
                u_merton = np.clip(u_merton, -0.2, 0.8)
                s_dict = {'W_t': info['wealth'], 'H_t': info['high_water_mark'], 'mu_t': mu_t, 'sigma_t': np.linalg.cholesky(cov_t + 1e-4*np.eye(num_assets)), 'r_t': env.r}
                action, _ = agent.qp_filter.filter_action(u_merton, s_dict)
            elif agent_name == "Merton (Analytical)":
                action = agent.select_action(obs, info)
            elif agent_name == "Soft Actor-Critic (SAC)":
                action = agent.select_action(obs, eval_mode=True)
            else:
                action = agent.select_action(obs)
                
            turnover = np.sum(np.abs(action - prev_u))
            turnovers.append(turnover)
            prev_u = action.copy()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            w_curr = info['wealth']
            
            daily_pct_ret = (w_curr - w_prev) / w_prev
            pct_returns.append(daily_pct_ret)
            
            wealth_history.append(w_curr)
            drawdown_history.append(info['drawdown'])
            dates_history.append(info['date'])
            
            if info['drawdown_violated']:
                violation_count += 1
                
            obs = next_obs
            if terminated or truncated:
                break
                
        w_arr = np.array(wealth_history)
        dd_arr = np.array(drawdown_history) * 100.0
        ret_arr = np.array(pct_returns)
        
        final_wealth = w_arr[-1]
        total_return = (final_wealth - 1.0) * 100.0
        num_years = len(w_arr) / 252.0
        annualized_return = ((final_wealth)**(1.0 / num_years) - 1.0) * 100.0
        
        max_dd = np.max(dd_arr)
        violation_rate = (violation_count / len(w_arr)) * 100.0
        
        excess_rets = ret_arr - env.r * env.dt
        std_ret = np.std(excess_rets) + 1e-8
        sharpe = (np.mean(excess_rets) / std_ret) * np.sqrt(252)
        
        neg_rets = excess_rets[excess_rets < 0]
        downside_std = np.std(neg_rets) + 1e-8 if len(neg_rets) > 0 else 1e-8
        sortino = (np.mean(excess_rets) / downside_std) * np.sqrt(252)
        
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

def run_rolling_window_backtest(window_length_days=1260, stride_days=252, max_drawdown=0.20):
    """
    Evaluates agents across overlapping 5-year rolling investment windows (2014-2019, 2015-2020, ..., 2019-2024).
    Computes mean ± std for Final Wealth, Max Drawdown, and Sharpe Ratio across rolling windows.
    """
    env = HistoricalSP500Env(max_drawdown=max_drawdown)
    num_assets = env.num_assets
    obs_dim = env.observation_space.shape[0]
    
    agents = {
        "E-CBLPO (Ours)": ECBLPOAgent(num_assets=num_assets, max_drawdown=max_drawdown),
        "PPO-Lagrangian": PPOLagrangianAgent(obs_dim=obs_dim, num_assets=num_assets, max_drawdown=max_drawdown),
        "Action Clipping": ActionClippingAgent(obs_dim=obs_dim, num_assets=num_assets),
        "Soft Actor-Critic (SAC)": SACAgent(obs_dim=obs_dim, action_dim=num_assets),
        "Unconstrained PPO": UnconstrainedPPOAgent(obs_dim=obs_dim, num_assets=num_assets),
        "Merton (Analytical)": MertonAnalyticalAgent(num_assets=num_assets)
    }
    
    total_days = env.horizon
    num_windows = (total_days - window_length_days) // stride_days + 1
    
    rolling_results = {agent_name: {"wealths": [], "drawdowns": [], "sharpes": [], "violations": []} for agent_name in agents}
    
    print(f"\n==================================================================")
    print(f"  RUNNING ROLLING 5-YEAR HISTORICAL BACKTEST ({num_windows} OVERLAPPING WINDOWS)")
    print(f"==================================================================")
    
    for w_idx in range(num_windows):
        start_t = w_idx * stride_days
        end_t = min(start_t + window_length_days, total_days)
        
        for agent_name, agent in agents.items():
            obs, info = env.reset()
            env.current_step = start_t
            env.W_t = 1.0
            env.H_t = 1.0
            env.prev_u = np.zeros(num_assets)
            
            w_hist = [1.0]
            dd_hist = [0.0]
            viols = 0
            pct_rets = []
            
            for t in range(start_t, end_t):
                w_prev = env.W_t
                
                if agent_name == "E-CBLPO (Ours)":
                    mu_t = env.rolling_mu[t]
                    cov_t = env.rolling_cov[t]
                    excess = mu_t - env.r
                    u_merton = np.linalg.solve(cov_t + 1e-4*np.eye(num_assets), excess) / 2.0
                    u_merton = np.clip(u_merton, -0.2, 0.8)
                    s_dict = {'W_t': env.W_t, 'H_t': env.H_t, 'mu_t': mu_t, 'sigma_t': np.linalg.cholesky(cov_t + 1e-4*np.eye(num_assets)), 'r_t': env.r}
                    action, _ = agent.qp_filter.filter_action(u_merton, s_dict)
                elif agent_name == "Merton (Analytical)":
                    action = agent.select_action(obs, info)
                elif agent_name == "Soft Actor-Critic (SAC)":
                    action = agent.select_action(obs, eval_mode=True)
                else:
                    action = agent.select_action(obs)
                    
                obs, reward, terminated, truncated, info = env.step(action)
                w_curr = info['wealth']
                pct_ret = (w_curr - w_prev) / w_prev
                pct_rets.append(pct_ret)
                dd_hist.append(info['drawdown'])
                
                if info['drawdown_violated']:
                    viols += 1
                if terminated or truncated:
                    break
                    
            w_arr = np.array(pct_rets)
            max_dd = np.max(dd_hist) * 100.0
            ex_rets = np.array(pct_rets) - env.r * env.dt
            std_r = np.std(ex_rets) + 1e-8
            sharpe = (np.mean(ex_rets) / std_r) * np.sqrt(252)
            
            ann_r = ((info['wealth'])**(252.0 / len(pct_rets)) - 1.0) * 100.0
            
            rolling_results[agent_name]["wealths"].append(info['wealth'])
            rolling_results[agent_name]["drawdowns"].append(max_dd)
            rolling_results[agent_name]["sharpes"].append(sharpe)
            rolling_results[agent_name]["violations"].append(viols / len(pct_rets) * 100.0)
            
    print("\nROLLING WINDOW SUMMARY (Mean ± Std across Rolling 5-Year Windows):")
    for agent_name, res in rolling_results.items():
        m_w, s_w = np.mean(res["wealths"]), np.std(res["wealths"])
        m_dd, s_dd = np.mean(res["drawdowns"]), np.std(res["drawdowns"])
        m_sh, s_sh = np.mean(res["sharpes"]), np.std(res["sharpes"])
        m_v = np.mean(res["violations"])
        print(f"  {agent_name:25s} | Final Wealth: {m_w:.4f}±{s_w:.4f} | Max DD: {m_dd:.2f}±{s_dd:.2f}% | Sharpe: {m_sh:.2f}±{s_sh:.2f} | Viol Rate: {m_v:.2f}%")
        
    return rolling_results

if __name__ == "__main__":
    run_historical_10yr_backtest()
    run_rolling_window_backtest()

if __name__ == "__main__":
    run_historical_10yr_backtest()
    run_rolling_window_backtest()
