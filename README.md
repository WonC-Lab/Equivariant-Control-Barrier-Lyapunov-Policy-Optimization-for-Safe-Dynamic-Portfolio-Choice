# Equivariant Control Barrier-Lyapunov Policy Optimization for Safe Dynamic Portfolio Choice

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **Equivariant Control Barrier-Lyapunov Policy Optimization (E-CBLPO)** for safe continuous-time dynamic portfolio choice.

> **Author**: WonChan Cho (Department of Mathematics, Sungkyunkwan University)  
> **Contact**: `chln0124@skku.edu`  
> **GitHub**: [WonC-Lab](https://github.com/WonC-Lab)

---

## Abstract & Key Contributions

Continuous-time dynamic portfolio optimization under stochastic market regimes faces severe drawdown risks and potential insolvency during tail-risk events. Conventional reinforcement learning (RL) and optimal control approaches enforce risk limits via soft penalty terms in the reward function or Lagrangian relaxations, which fail to guarantee safety during black-swan market crashes.

**E-CBLPO** unifies:
1. **Stochastic Control Barrier Functions (CBF)**: Formulates a drawdown barrier $h(x_t) = W_t - (1-\alpha)H_t \ge 0$ establishing pathwise forward invariance of the safe wealth set $\mathcal{C}$ ($\mathbb{P}(D_t \le \alpha, \, \forall t \in [0, T]) = 1$).
2. **Control Lyapunov Functions (CLF)**: Drives continuous portfolio growth and expected utility maximization with adaptive slack relaxation $\delta_v$.
3. **Lie-Equivariant Neural Architecture**: Constrains neural policy outputs to preserve asset permutation ($S_N$) and diagonal scaling symmetries ($\pi_\phi(P x_t) = P \pi_\phi(x_t)$).
4. **Differentiable QP Safety Projection Layer**: Projects unconstrained policy actions onto the safe control space in real-time via convex KKT optimization.

---

## Benchmark Empirical Performance Summary

Metrics reported over $M=25$ independent market trajectory seeds ($\alpha = 20.0\%$ Max Drawdown Limit).

| Environment & Algorithm | Final Wealth $W_T$ | $CVaR_{0.05} (W_T)$ | Max Drawdown (%) | Violation Rate (%) | Sharpe Ratio | Calmar Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kim-Omberg Stochastic Risk Premium** | | | | | | |
| Merton (Analytical) | $1.0655 \pm 0.0706$ | 0.7023 | 13.75% | 12.0% | 0.1945 | 0.9036 |
| Unconstrained PPO | $0.8107 \pm 0.0481$ | 0.5881 | 25.16% | 76.0% | -1.4582 | -0.6739 |
| Soft Actor-Critic (SAC) | $1.0566 \pm 0.0698$ | 0.6893 | 13.95% | 20.0% | 0.2262 | 0.9651 |
| Action Clipping | $0.8843 \pm 0.0568$ | 0.6149 | 19.46% | 40.0% | -1.0927 | -0.2964 |
| PPO-Lagrangian | $0.8411 \pm 0.0530$ | 0.5960 | 22.25% | 48.0% | -1.1736 | -0.6027 |
| **E-CBLPO (Ours)** | $\mathbf{1.0325 \pm 0.0254}$ | **0.8847** | **4.91%** | **0.0%** | **0.1888** | **1.2646** |
| **Multi-Asset Stochastic Volatility (Heston 5-Asset)** | | | | | | |
| Merton (Analytical) | $1.0889 \pm 0.1092$ | 0.6333 | 20.97% | 52.0% | 0.2565 | 0.9815 |
| Unconstrained PPO | $0.7771 \pm 0.0450$ | 0.6292 | 26.86% | 80.0% | -2.1155 | -0.7727 |
| Soft Actor-Critic (SAC) | $1.0497 \pm 0.0464$ | 0.8197 | 9.66% | 8.0% | 0.2805 | 1.1332 |
| Action Clipping | $0.7679 \pm 0.0419$ | 0.5636 | 26.69% | 80.0% | -2.3445 | -0.8218 |
| PPO-Lagrangian | $0.7671 \pm 0.0430$ | 0.5819 | 27.60% | 76.0% | -2.3096 | -0.8001 |
| **E-CBLPO (Ours)** | $\mathbf{1.0508 \pm 0.0462}$ | **0.8250** | **9.53%** | **0.0%** | **0.3062** | **1.0900** |
| **Merton Jump-Diffusion Market Crash Stress** | | | | | | |
| Merton (Analytical) | $0.8962 \pm 0.0740$ | 0.5089 | 25.56% | 64.0% | -0.5562 | -0.2152 |
| Unconstrained PPO | $0.8046 \pm 0.0746$ | 0.3570 | 30.50% | 84.0% | -0.8813 | -0.5212 |
| Soft Actor-Critic (SAC) | $0.8881 \pm 0.0807$ | 0.5364 | 27.13% | 72.0% | -0.5431 | -0.2156 |
| Action Clipping | $0.7963 \pm 0.0926$ | 0.3481 | 32.07% | 80.0% | -1.0394 | -0.4610 |
| PPO-Lagrangian | $0.8191 \pm 0.0724$ | 0.5447 | 29.38% | 72.0% | -0.8363 & -0.4536 |
| **E-CBLPO (Ours)** | $\mathbf{1.0631 \pm 0.0365}$ | **0.9572** | **5.70%** | **0.0%** | **0.3661** | **1.6785** |

---

## Repository Structure

```
├── src/
│   ├── agents/            # E-CBLPO agent, PPO, SAC, Lagrangian baselines
│   ├── environments/      # Kim-Omberg, Heston 5-Asset, Merton Jump-Diffusion SDE envs
│   ├── models/            # S_N Equivariant Policy & Value neural networks
│   └── safety/            # Differentiable CBF-CLF QP Safety Filter
├── experiments/
│   ├── plots/             # Generated high-resolution paper figures (.png)
│   ├── run_robust_experiments.py   # Multi-seed evaluation pipeline
│   ├── generate_paper_tables.py    # LaTeX table generation script
│   ├── plot_credibility_proofs.py  # 50-asset heatmap & scalability plots
│   ├── plot_historical_backtest.py # 10+ Year S&P 500 historical backtest
│   └── backtest_historical.py      # Real-market data backtest engine
├── tests/                 # Unit tests for CBF filter, equivariance, and SDE dynamics
├── README.md              # Documentation and usage guide
└── .gitignore             # Git ignore configuration
```

---

## Installation & Requirements

Ensure Python 3.9+ and PyTorch are installed.

```bash
git clone https://github.com/WonC-Lab/Equivariant-Control-Barrier-Lyapunov-Policy-Optimization-for-Safe-Dynamic-Portfolio-Choice.git
cd Equivariant-Control-Barrier-Lyapunov-Policy-Optimization-for-Safe-Dynamic-Portfolio-Choice

# Install dependencies
pip install torch numpy scipy pandas matplotlib gymnasium
```

---

## Quickstart & Reproduction

### 1. Run Multi-Seed Benchmark Experiments
Evaluates all algorithms over 25 independent random seeds across Kim-Omberg, Heston, and Jump-Diffusion environments:
```bash
python experiments/run_robust_experiments.py
```

### 2. Generate LaTeX Performance Tables
Generates publication-ready LaTeX tables (`experiments/paper_results_table.tex`):
```bash
python experiments/generate_paper_tables.py
```

### 3. Generate Credibility Plots & 50-Asset Heatmaps
Generates high-dimensional scalability plots, Pareto frontiers, and 50-asset allocation heatmaps:
```bash
python experiments/plot_credibility_proofs.py
```

### 4. Run 10+ Year Real-Market Historical S&P 500 Backtest
Executes historical backtest over 2,779 real trading days (2014–2024) across 4 major market crashes:
```bash
python experiments/plot_historical_backtest.py
```

---

## Citation

If you use E-CBLPO in your research, please cite:

```bibtex
@article{cho2026equivariant,
  title={Equivariant Control Barrier-Lyapunov Policy Optimization for Safe Dynamic Portfolio Choice},
  author={Cho, WonChan},
  journal={},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
