import os
import argparse
import numpy as np
import torch

from src.environments.kim_omberg import KimOmbergEnv
from src.environments.multi_asset_heston import MultiAssetHestonEnv
from src.agents.ecblpo_agent import ECBLPOAgent
from src.agents.baselines import MertonAnalyticalAgent, UnconstrainedPPOAgent, PPOLagrangianAgent

def train_ecblpo(env_type="kim_omberg", episodes=100, horizon=252, max_drawdown=0.20):
    print(f"=== Starting E-CBLPO Training on [{env_type.upper()}] ===")
    
    if env_type == "kim_omberg":
        env = KimOmbergEnv(horizon=horizon, max_drawdown=max_drawdown)
    else:
        env = MultiAssetHestonEnv(num_assets=5, horizon=horizon, max_drawdown=max_drawdown)
        
    num_assets = env.num_assets
    agent = ECBLPOAgent(num_assets=num_assets, max_drawdown=max_drawdown)
    
    history = {
        "final_wealth": [],
        "max_drawdown": [],
        "violations": []
    }
    
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        max_ep_drawdown = 0.0
        violation_count = 0
        
        rollout = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        for t in range(horizon):
            u_tilde, u_safe, _ = agent.select_action(obs, info)
            
            next_obs, reward, terminated, truncated, info = env.step(u_safe)
            
            rollout['states'].append(obs)
            rollout['actions'].append(u_tilde)
            rollout['rewards'].append(reward)
            rollout['next_states'].append(next_obs)
            rollout['dones'].append(terminated or truncated)
            
            ep_reward += reward
            max_ep_drawdown = max(max_ep_drawdown, info['drawdown'])
            if info['drawdown_violated']:
                violation_count += 1
                
            obs = next_obs
            if terminated or truncated:
                break
                
        agent.update(rollout)
        
        history["final_wealth"].append(info['wealth'])
        history["max_drawdown"].append(max_ep_drawdown)
        history["violations"].append(violation_count)
        
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep:3d}/{episodes} | Final Wealth: {info['wealth']:.4f} | Max DD: {max_ep_drawdown*100:.2f}% | Violations: {violation_count}")

    print("=== Training Complete ===")
    return history, agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train E-CBLPO")
    parser.add_argument("--env", type=str, default="kim_omberg", choices=["kim_omberg", "heston"])
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    train_ecblpo(env_type=args.env, episodes=args.episodes)
