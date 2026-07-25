import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from run_robust_experiments import run_multi_seed_evaluation

def generate_latex_tables(output_path="experiments/paper_results_table.tex"):
    print("\n==================================================================")
    print(" GENERATING PUBLICATION-GRADE LATEX TABLES FOR PAPER SUBMISSION")
    print("==================================================================")
    
    envs = ["kim_omberg", "heston", "jump_crash"]
    all_results = {}
    
    for env in envs:
        all_results[env] = run_multi_seed_evaluation(env_type=env, num_seeds=25, max_drawdown=0.20)
        
    latex_str = r"""\begin{table*}[t!]
\centering
\caption{\textbf{Multi-Seed Empirical Benchmark Performance across Continuous-Time Portfolio Environments.} Performance metrics reported over $M=25$ independent market trajectory seeds ($\alpha = 20.0\%$ Max Drawdown Limit). $W_T$ reports Mean $\pm 95\%$ Confidence Interval. $CVaR_{0.05}$ measures the bottom 5\% tail wealth. Drawdown Violation Rate (\%) measures the percentage of episodes violating hard safety bounds.}
\label{tab:empirical_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Environment \& Algorithm} & \textbf{Final Wealth $W_T$} & \textbf{CVaR$_{0.05}$ ($W_T$)} & \textbf{Max Drawdown (\%)} & \textbf{Violation Rate (\%)} & \textbf{Sharpe Ratio} & \textbf{Calmar Ratio} \\
\midrule
"""

    env_names = {
        "kim_omberg": "Kim-Omberg Stochastic Risk Premium Model",
        "heston": "Multi-Asset Stochastic Volatility (Heston 5-Asset)",
        "jump_crash": "Merton Jump-Diffusion Market Crash Stress Model"
    }

    for env_key, env_label in env_names.items():
        latex_str += f"\\multicolumn{{7}}{{l}}{{\\textbf{{{env_label}}}}} \\\\\n\\midrule\n"
        res_dict = all_results[env_key]
        for agent_name, m in res_dict.items():
            w_str = f"${m['mean_wealth']:.4f} \\pm {m['ci_95_wealth']:.4f}$"
            cvar_str = f"{m['cvar_05']:.4f}"
            dd_str = f"{m['mean_max_dd']:.2f}\\%"
            viol_str = f"\\mathbf{{{m['violation_rate']:.1f}\\%}}" if m['violation_rate'] == 0.0 else f"{m['violation_rate']:.1f}\\%"
            sharpe_str = f"{m['mean_sharpe']:.4f}"
            calmar_str = f"{m['mean_calmar']:.4f}"
            
            if "Ours" in agent_name:
                agent_formatted = f"\\textbf{{{agent_name}}}"
                w_str = f"$\\mathbf{{{m['mean_wealth']:.4f} \\pm {m['ci_95_wealth']:.4f}}}$"
            else:
                agent_formatted = agent_name
                
            latex_str += f"{agent_formatted} & {w_str} & {cvar_str} & {dd_str} & {viol_str} & {sharpe_str} & {calmar_str} \\\\\n"
        latex_str += "\\midrule\n"
        
    latex_str += r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
        
    print(f"\n[Success] LaTeX table generated at: {output_path}\n")
    print(latex_str)
    return latex_str

if __name__ == "__main__":
    generate_latex_tables()
