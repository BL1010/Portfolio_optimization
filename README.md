This repository implements Portfolio Optimization using Reinforcement Learning (RL) techniques and presents a comparative study across different environments and model-free RL algorithms. The aim is to train intelligent agents that can dynamically allocate assets over time to maximize cumulative return and minimize risk.

Key Contributions
1.Dynamic portfolio allocation using RL
2.Comparative study of performance across:
Different environments (e.g., discrete vs continuous action space, risk-adjusted rewards)
3.Various model-free RL algorithms
4.Visualization of portfolio weights, rewards, and drawdowns
odular design for easy extension

The Following models have been used in the project. 
1. DDPG Model
2. PPO Model

Three types of environments have been tested: 

1. Environment Without Cash Rebalance(infinite capital)
2. Environment With Cash Rebalance(dynamically adjusted capital based on profits and losses)
3. Environment With sharpe ratio reward. 
