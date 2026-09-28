import numpy as np
import torch
from copy import deepcopy

from uav_safe_marl.core.types import Transition
from uav_safe_marl.critics.graph_critic import GraphCostCriticEnsemble, GraphScalarCritic
from uav_safe_marl.graph.observations import stack_graph_observations
from uav_safe_marl.policies.graph_actor import GraphSACActor
from uav_safe_marl import build_env, load_config
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.replay.buffer import NumpyReplayBuffer


def graph(count, embedding_dim=4):
    rng = np.random.default_rng(count)
    adjacency = np.ones((count, count), dtype=bool)
    np.fill_diagonal(adjacency, False)
    return {
        "node_features": rng.normal(size=(count, 14)),
        "adjacency": adjacency,
        "edge_features": rng.normal(size=(count, count, 7 + embedding_dim)),
        "node_mask": np.ones(count, dtype=bool),
    }


def torch_graph(value):
    batch = stack_graph_observations([value])
    return {key: torch.as_tensor(item, dtype=torch.bool if key in {"adjacency", "node_mask"} else torch.float32) for key, item in batch.items()}


def test_graph_actor_is_permutation_equivariant_and_emits_configured_z():
    torch.manual_seed(3)
    actor = GraphSACActor(11, 8, 4, 1, 2, 8)
    original = graph(3)
    permutation = np.array([2, 0, 1])
    permuted = {
        "node_features": original["node_features"][permutation],
        "adjacency": original["adjacency"][permutation][:, permutation],
        "edge_features": original["edge_features"][permutation][:, permutation],
        "node_mask": original["node_mask"][permutation],
    }
    action, _, embedding = actor.sample(torch_graph(original), deterministic=True)
    permuted_action, _, permuted_embedding = actor.sample(torch_graph(permuted), deterministic=True)
    assert embedding.shape == (1, 3, 4)
    torch.testing.assert_close(permuted_action[0], action[0, permutation])
    torch.testing.assert_close(permuted_embedding[0], embedding[0, permutation])


def test_graph_critic_is_permutation_invariant():
    torch.manual_seed(4)
    critic = GraphScalarCritic(8, 8, 1, 2)
    original = graph(3)
    action = torch.randn(1, 3, 3)
    value = critic(torch_graph(original), action, torch.ones((1, 3, 3), dtype=torch.bool))
    permutation = np.array([2, 0, 1])
    permuted = {
        "node_features": original["node_features"][permutation],
        "adjacency": original["adjacency"][permutation][:, permutation],
        "edge_features": original["edge_features"][permutation][:, permutation],
        "node_mask": original["node_mask"][permutation],
    }
    permuted_value = critic(torch_graph(permuted), action[:, permutation], torch.ones((1, 3, 3), dtype=torch.bool))
    torch.testing.assert_close(value, permuted_value)


def test_cost_graph_ensemble_members_heads_targets_and_updates_are_independent():
    ensemble = GraphCostCriticEnsemble(3, 8, 8, 1, 2)
    encoder_pointers = [next(member.encoder.parameters()).data_ptr() for member in ensemble.members]
    head_pointers = [next(member.head.parameters()).data_ptr() for member in ensemble.members]
    assert len(set(encoder_pointers)) == 3
    assert len(set(head_pointers)) == 3

    targets = deepcopy(ensemble)
    target_pointers = [next(member.encoder.parameters()).data_ptr() for member in targets.members]
    assert len(set(target_pointers)) == 3
    assert set(target_pointers).isdisjoint(encoder_pointers)

    updated_before = [parameter.detach().clone() for parameter in ensemble.members[0].parameters()]
    untouched = [parameter.detach().clone() for parameter in ensemble.members[1].parameters()]
    optimizer = torch.optim.SGD(ensemble.members[0].parameters(), lr=0.1)
    value = graph(2)
    loss = ensemble.members[0](torch_graph(value), torch.zeros((1, 2, 3)), torch.ones((1, 2, 2), dtype=torch.bool)).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert any(not torch.equal(before, after) for before, after in zip(updated_before, ensemble.members[0].parameters(), strict=True))
    for before, after in zip(untouched, ensemble.members[1].parameters(), strict=True):
        torch.testing.assert_close(before, after)


def test_graph_replay_pads_variable_agent_counts():
    replay = NumpyReplayBuffer(4, seed=0)
    for count in (2, 3):
        value = graph(count)
        replay.add(Transition(value, np.zeros((count, 3)), np.zeros((count, 3)), 0.0, 0.0, value, False))
    sample = replay.sample(2)
    assert sample["state"]["node_features"].shape == (2, 3, 14)
    assert sorted(sample["state"]["node_mask"].sum(axis=1).tolist()) == [2, 3]
    assert sample["nominal_action"].shape == (2, 3, 3)


def test_graph_replay_freezes_received_embedding_and_information_age_snapshot():
    replay = NumpyReplayBuffer(2, seed=0)
    observed = graph(2)
    observed["edge_features"][0, 1, 6] = 9.0  # information age
    observed["edge_features"][0, 1, 7:] = [4.0, 3.0, 2.0, 1.0]
    expected = observed["edge_features"].copy()
    replay.add(Transition(observed, np.zeros((2, 3)), np.zeros((2, 3)), 0.0, 0.0, observed, False))
    observed["edge_features"].fill(-999.0)
    sampled = replay.sample(1)["state"]["edge_features"][0, :2, :2]
    np.testing.assert_array_equal(sampled, expected)


def test_graph_backend_emits_message_embedding_with_configured_width(tmp_path):
    config = load_config(overrides={
        "graph": {"actor_embedding_dim": 7},
        "environment": {"max_steps": 1},
        "learning": {"batch_size": 8},
    })
    trainer = SafeRLTrainer(config, build_env(config), output_dir=tmp_path)
    trainer.train(episodes=1)
    assert trainer.backend.last_actor_embedding.shape == (config.environment.agent_count, 7)


def test_actor_graph_masks_exited_agents_and_their_edges():
    from uav_safe_marl.communication.message import P2PMessage
    from uav_safe_marl.core.types import AgentState
    from uav_safe_marl.graph.observations import build_actor_graph

    states = [AgentState(np.zeros(3), np.zeros(3), np.ones(3), np.ones(3), np.ones(3)), AgentState(np.ones(3), np.zeros(3), np.ones(3) * 2, np.ones(3), np.ones(3))]
    cache = {1: {0: P2PMessage(0, 1, np.zeros(3), np.zeros(3), np.ones(4), 1, 0)}}
    value = build_actor_graph(states, cache, 4, np.array([False, True]))
    np.testing.assert_array_equal(value["node_mask"], [False, True])
    assert not value["adjacency"].any()
    np.testing.assert_array_equal(value["node_features"][0], np.zeros(14))
