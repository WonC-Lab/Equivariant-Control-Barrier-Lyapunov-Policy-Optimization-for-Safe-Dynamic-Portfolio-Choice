import unittest
import numpy as np
import torch

from src.environments.kim_omberg import KimOmbergEnv
from src.environments.multi_asset_heston import MultiAssetHestonEnv
from src.models.equivariant_policy import EquivariantPolicy
from src.models.value_net import ValueNetwork
from src.safety.cbf_clf_qp import CBFCLFQPFilter
from src.agents.ecblpo_agent import ECBLPOAgent
from src.agents.baselines import MertonAnalyticalAgent

class TestECBLPOComponents(unittest.TestCase):
    
    def test_kim_omberg_env(self):
        env = KimOmbergEnv(horizon=20, max_drawdown=0.20)
        obs, info = env.reset()
        self.assertEqual(obs.shape[0], 4)  # W_t, H_t, D_t, theta_t
        self.assertAlmostEqual(info['wealth'], 1.0)
        
        obs, reward, terminated, truncated, info = env.step(np.array([0.5]))
        self.assertGreater(info['wealth'], 0.0)
        self.assertLessEqual(info['drawdown'], 1.0)

    def test_multi_asset_heston_env(self):
        env = MultiAssetHestonEnv(num_assets=5, horizon=20, max_drawdown=0.15)
        obs, info = env.reset()
        self.assertEqual(obs.shape[0], 3 + 5)  # W_t, H_t, D_t, v_{1..5}
        
        obs, reward, terminated, truncated, info = env.step(np.ones(5) * 0.2)
        self.assertGreater(info['wealth'], 0.0)

    def test_equivariant_policy_symmetry(self):
        """
        Tests S_N permutation equivariance: pi(P * x) == P * pi(x).
        """
        num_assets = 5
        policy = EquivariantPolicy(num_assets=num_assets, asset_dim=1, global_dim=3)
        policy.eval()
        
        global_state = torch.tensor([[1.2, 1.5, 0.2]])  # W, H, D
        asset_state = torch.tensor([[0.05, 0.02, 0.08, -0.01, 0.04]])
        obs = torch.cat([global_state, asset_state], dim=1)
        
        with torch.no_grad():
            output_orig = policy(obs)[0]
            
        # Apply permutation P = (1, 2, 0, 4, 3)
        perm = [1, 2, 0, 4, 3]
        perm_asset_state = asset_state[:, perm]
        perm_obs = torch.cat([global_state, perm_asset_state], dim=1)
        
        with torch.no_grad():
            output_perm = policy(perm_obs)[0]
            
        # Check if output is permuted by the same P
        expected_output_perm = output_orig[perm]
        
        max_diff = torch.max(torch.abs(output_perm - expected_output_perm)).item()
        self.assertLess(max_diff, 1e-5, f"Permutation Equivariance failed! Max diff: {max_diff}")

    def test_cbf_clf_qp_filter_safety(self):
        """
        Tests that CBF QP filter strictly enforces drawdown safety when near boundary.
        """
        qp = CBFCLFQPFilter(max_drawdown=0.20, gamma_cbf=2.0)
        
        # State at 19.99% drawdown (very close to 20% limit) under adverse market drift
        state_dict = {
            'W_t': 0.8001,
            'H_t': 1.0,
            'mu_t': np.array([-0.10, -0.10]),
            'sigma_t': np.diag([0.30, 0.30]),
            'r_t': 0.01
        }
        
        # Aggressive risky allocation u_tilde = [2.0, 2.0]
        u_tilde = np.array([2.0, 2.0])
        u_safe, delta_v = qp.filter_action(u_tilde, state_dict)
        
        # Verify u_safe reduces leverage to prevent crossing 20% drawdown limit
        self.assertLess(np.sum(u_safe), np.sum(u_tilde))
        self.assertTrue(np.all(u_safe >= -0.5) and np.all(u_safe <= 2.0))

    def test_ecblpo_agent_rollout(self):
        env = KimOmbergEnv(horizon=10, max_drawdown=0.20)
        agent = ECBLPOAgent(num_assets=1, max_drawdown=0.20)
        
        obs, info = env.reset()
        for _ in range(5):
            u_tilde, u_safe, _ = agent.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(u_safe)
            if terminated:
                break
        self.assertGreater(info['wealth'], 0.0)

if __name__ == "__main__":
    unittest.main()
