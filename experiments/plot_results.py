import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from evaluate import run_evaluation

def generate_paper_plots(output_dir="experiments/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_evaluation(env_name="kim_omberg", episodes=15, max_drawdown=0.20)
    
    # Plot 1: Cumulative Wealth Trajectories
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = {
        "Merton (Analytical)": "#e74c3c",
        "Unconstrained PPO": "#e67e22",
        "PPO-Lagrangian": "#f1c40f",
        "E-CBLPO (Ours)": "#2ecc71"
    }
    
    for agent_name, res in results.items():
        w_mean = np.mean(res["wealth_trajectories"], axis=0)
        dd_mean = np.mean(res["drawdown_trajectories"], axis=0) * 100.0
        steps = np.arange(len(w_mean))
        
        ax1.plot(steps, w_mean, label=agent_name, color=colors[agent_name], linewidth=2.0)
        ax2.plot(steps, dd_mean, label=agent_name, color=colors[agent_name], linewidth=2.0)
        
    ax1.set_ylabel("Portfolio Wealth $W_t$", fontsize=12)
    ax1.set_title("E-CBLPO: Cumulative Wealth Trajectories under Stochastic Risk Premium", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Red dashed line for hard 20% drawdown limit
    ax2.axhline(20.0, color="red", linestyle="--", linewidth=1.5, label="Hard Safety Boundary (20%)")
    ax2.set_xlabel("Time Steps (Days)", fontsize=12)
    ax2.set_ylabel("Drawdown $D_t$ (%)", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ecblpo_wealth_drawdown_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[Saved Plot]: {plot_path}")

if __name__ == "__main__":
    generate_paper_plots()
