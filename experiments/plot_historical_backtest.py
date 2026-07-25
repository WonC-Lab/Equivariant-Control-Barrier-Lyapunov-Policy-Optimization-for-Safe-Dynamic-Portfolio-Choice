import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backtest_historical import run_historical_10yr_backtest

def generate_historical_backtest_plots(output_dir="experiments/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_historical_10yr_backtest(max_drawdown=0.20)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    colors = {
        "Merton (Analytical)": "#e74c3c",
        "Unconstrained PPO": "#e67e22",
        "Soft Actor-Critic (SAC)": "#9b59b6",
        "Action Clipping": "#34495e",
        "PPO-Lagrangian": "#f1c40f",
        "E-CBLPO (Ours)": "#2ecc71"
    }
    
    # Get common dates timeline
    first_agent = list(results.keys())[0]
    dates = pd.to_datetime(results[first_agent]["dates_history"])
    
    for agent_name, res in results.items():
        w_hist = res["wealth_history"]
        dd_hist = res["drawdown_history"]
        
        # Legend with safety violation info
        max_dd = res["max_drawdown"]
        viol_rate = res["violation_rate"]
        if viol_rate > 0:
            label_str = f"{agent_name} [Violated: Max DD {max_dd:.2f}%]"
        else:
            label_str = f"{agent_name} [Safe: Max DD {max_dd:.2f}%]"
            
        ax1.plot(dates[:len(w_hist)], w_hist, label=label_str, color=colors[agent_name], linewidth=2.0)
        ax2.plot(dates[:len(dd_hist)], np.array(dd_hist)*100.0, label=label_str, color=colors[agent_name], linewidth=1.5)
        
    # Shaded historical crash periods
    crashes = [
        ("2015-08-01", "2016-02-28", "2015-16 Turmoil"),
        ("2018-10-01", "2018-12-31", "2018 Q4 Tech Drop"),
        ("2020-02-15", "2020-04-15", "2020 COVID Crash"),
        ("2022-01-01", "2022-10-31", "2022 Rate Hike Bear")
    ]
    
    for start, end, label in crashes:
        s_date = pd.to_datetime(start)
        e_date = pd.to_datetime(end)
        ax1.axvspan(s_date, e_date, color='grey', alpha=0.15)
        ax2.axvspan(s_date, e_date, color='grey', alpha=0.15)
        
    ax1.set_yscale('log')
    ax1.set_ylabel("Portfolio Wealth $W_t$ (Log Scale)", fontsize=12)
    ax1.set_title("10+ Year Historical S&P 500 Market Crash Backtest (2014-2024)", fontsize=14, fontweight='bold')
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5, which="both")
    
    ax2.axhline(20.0, color="red", linestyle="--", linewidth=2.0, label="Hard Safety Limit (20%)")
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Continuous Drawdown $D_t$ (%)", fontsize=12)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "historical_sp500_backtest.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[Saved High-Res Plot]: {plot_path}\n")
    return results

if __name__ == "__main__":
    generate_historical_backtest_plots()
