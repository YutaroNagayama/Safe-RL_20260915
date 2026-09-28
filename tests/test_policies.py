import numpy as np
import torch

from uav_safe_marl.policies.sac_actor import SACActor
from uav_safe_marl.critics.cost_ensemble import CostCriticEnsemble


def test_sac_gaussian_reparameterization_is_finite_and_mean_is_deterministic():
    torch.manual_seed(1)
    actor = SACActor(5, 8)
    observation = torch.zeros((4, 5))
    mean_one, _ = actor.sample(observation, deterministic=True)
    mean_two, _ = actor.sample(observation, deterministic=True)
    stochastic, log_probability = actor.sample(observation, deterministic=False)
    assert torch.equal(mean_one, mean_two)
    assert torch.isfinite(stochastic).all() and torch.isfinite(log_probability).all()
    assert np.all(np.abs(stochastic.detach().numpy()) <= 1.0)


def test_cost_critic_ensemble_members_are_independently_initialized():
    torch.manual_seed(9)
    ensemble = CostCriticEnsemble(3, state_dim=4, joint_action_dim=3, hidden_dim=8)
    first_weights = [next(member.parameters()).detach() for member in ensemble.members]
    assert not torch.equal(first_weights[0], first_weights[1])
    assert not torch.equal(first_weights[1], first_weights[2])
