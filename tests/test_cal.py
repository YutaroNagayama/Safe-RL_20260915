import numpy as np

from uav_safe_marl.algorithms.cal import CALObjective, TorchCAL, cost_ucb
from uav_safe_marl.algorithms.primal_dual import ProjectedDual


def test_cost_ucb_uses_population_standard_deviation():
    estimates = np.array([[1.0, 4.0], [3.0, 4.0]])
    np.testing.assert_allclose(cost_ucb(estimates, beta=2.0), [4.0, 4.0])


def test_cal_piecewise_objective_and_dual_projection():
    objective = CALObjective(cost_limit=2.0, coefficient=2.0)
    assert objective.evaluate(10.0, 1.0, dual=0.0) == 10.0
    assert objective.evaluate(10.0, 3.0, dual=1.0) == 8.0
    dual = ProjectedDual(0.0, 0.5)
    assert dual.update(-2.0) == 0.0 and dual.update(2.0) == 1.0


def test_torch_cal_modes_are_representation_independent():
    import torch

    members = torch.tensor([[[1.0]], [[3.0]]])
    actor_members = torch.tensor([[[2.0]], [[4.0]]], requires_grad=True)
    pdf = TorchCAL(2.0, 1.0, 1.0, "pdf_variant")
    paper = TorchCAL(2.0, 1.0, 1.0, "paper")
    replay_ucb = pdf.ucb(members)
    actor_ucb = pdf.ucb(actor_members)
    assert replay_ucb.item() == 3.0
    assert torch.isfinite(pdf.actor_penalty(actor_ucb, replay_ucb, 0.5)).all()
    assert torch.isfinite(paper.actor_penalty(actor_ucb, replay_ucb, 0.5)).all()


def test_episodic_cal_uses_one_start_state_violation_for_replay_gradient():
    import torch

    cal = TorchCAL(6.0, 2.0, 1.0, "pdf_variant")
    actor_ucb = torch.tensor([[3.0], [9.0]], requires_grad=True)
    replay_ucb = torch.tensor([[100.0], [100.0]])
    penalty = cal.actor_penalty(actor_ucb, replay_ucb, dual=1.0, constraint_violation=0.5)
    # lambda_tilde = 1 + 2 * 0.5 = 2 for every replay state.
    np.testing.assert_allclose(penalty.detach().numpy(), [[6.0], [18.0]])
    assert cal.effective_multiplier(1.0, 0.5) == 2.0
