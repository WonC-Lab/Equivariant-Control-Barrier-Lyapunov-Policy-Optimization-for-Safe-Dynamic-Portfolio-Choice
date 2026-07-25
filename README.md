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
| Merton (Analytical) | $1.0655 \pm 0.0706$ | 0.7023 | 13.75% | 12.0% | 0.2715 | 0.4765 |
| Unconstrained PPO | $0.8472 \pm 0.0662$ | 0.5572 | 25.29% | 68.0% | -1.1001 | -0.6041 |
| Soft Actor-Critic (SAC) | $1.0113 \pm 0.0173$ | 0.9469 | 3.87% | **0.0%** | -0.2125 | 0.2916 |
| Action Clipping | $0.8732 \pm 0.0531$ | 0.6034 | 19.99% | 48.0% | -1.1658 | -0.6345 |
| PPO-Lagrangian | $0.8492 \pm 0.0435$ | 0.6746 | 22.50% | 64.0% | -1.6539 | -0.6704 |
| **E-CBLPO (Ours)** | $\mathbf{1.0278 \pm 0.0169}$ | **0.9296** | **3.12%** | **0.0%** | **0.1934** | **0.8910** |
| **Multi-Asset Stochastic Volatility (Heston 5-Asset)** | | | | | | |
| Merton (Analytical) | $1.0856 \pm 0.1166$ | 0.5997 | 23.21% | 52.0% | 0.2372 | 0.3689 |
| Unconstrained PPO | $0.3489 \pm 0.0531$ | 0.1195 | 68.32% | 100.0% | -5.3268 | -0.9530 |
| Soft Actor-Critic (SAC) | $1.0058 \pm 0.0151$ | 0.9372 | 4.33% | **0.0%** | -0.3952 | 0.1347 |
| Action Clipping | $0.4101 \pm 0.0507$ | 0.1680 | 62.73% | 100.0% | -5.0673 | -0.9404 |
| PPO-Lagrangian | $0.3861 \pm 0.0513$ | 0.1469 | 65.20% | 100.0% | -5.2015 | -0.9415 |
| **E-CBLPO (Ours)** | $\mathbf{1.0050 \pm 0.0308}$ | **0.9424** | **6.84%** | **0.0%** | **-0.2056** | **0.0723** |
| **Merton Jump-Diffusion Market Crash Stress** | | | | | | |
| Merton (Analytical) | $0.8962 \pm 0.0740$ | 0.5089 | 25.56% | 64.0% | -0.7050 | -0.4062 |
| Unconstrained PPO | $0.7677 \pm 0.0951$ | 0.3755 | 34.14% | 80.0% | -1.1174 | -0.6804 |
| Soft Actor-Critic (SAC) | $0.9850 \pm 0.0238$ | 0.8340 | 7.52% | **0.0%** | -0.6204 | -0.1994 |
| Action Clipping | $0.7861 \pm 0.0976$ | 0.5016 | 30.80% | 84.0% | -0.8377 | -0.5655 |
| PPO-Lagrangian | $0.9257 \pm 0.0944$ | 0.4863 | 24.48% | 60.0% | -0.4207 | -0.3034 |
| **E-CBLPO (Ours)** | $\mathbf{1.0352 \pm 0.0127}$ | **0.9772** | **1.91%** | **0.0%** | **0.5057** | **1.8425** |

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
