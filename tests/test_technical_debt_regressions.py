from __future__ import annotations

import random

import numpy as np
import torch

from uav_safe_marl import build_env, load_config
from uav_safe_marl.communication.message import P2PMessage
from uav_safe_marl.evaluation.scenarios import interaction_descriptor
from uav_safe_marl.runners.trainer import SafeRLTrainer


def _trainer(tmp_path, **overrides):
    base = {
        "environment": {"agent_count": 2, "max_steps": 2},
        "learning": {"batch_size": 100, "buffer_size": 100},
        "monitoring": {"enabled": False},
    }
    for section, values in overrides.items():
        base.setdefault(section, {}).update(values)
    config = load_config(overrides=base)
    return SafeRLTrainer(config, build_env(config), tmp_path)


def test_replay_next_graph_is_exact_next_tick_actor_graph(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.train(1, reset_options={"predicted_los_edges": [(0, 1)]})
    assert len(trainer.buffer) == 2  # terminal pending transition is flushed
    first, second = trainer.buffer._at(0), trainer.buffer._at(1)
    for key in first.next_state:
        np.testing.assert_array_equal(first.next_state[key], second.state[key])
    assert first.next_state["adjacency"].any()


def test_multirate_phase_and_held_residual_are_frozen_in_replay(tmp_path):
    trainer = _trainer(tmp_path, communication={"actor_period_k": 3}, environment={"max_steps": 4})
    calls = 0

    def act(_observation, deterministic=False):
        nonlocal calls
        calls += 1
        return np.full(trainer.env.action_space.shape, 0.1 * calls)

    trainer.backend.act = act
    trainer.train(1)
    transitions = [trainer.buffer._at(index) for index in range(len(trainer.buffer))]
    assert [item.actor_phase for item in transitions] == [0, 1, 2, 0]
    np.testing.assert_allclose(transitions[0].held_residual_action, transitions[1].held_residual_action)
    np.testing.assert_allclose(transitions[1].held_residual_action, transitions[2].held_residual_action)
    assert transitions[0].next_actor_phase == 1
    assert not transitions[0].next_is_actor_decision_tick
    assert transitions[2].next_is_actor_decision_tick


def test_reactive_cache_drops_out_of_range_and_does_not_revive_stale_message(tmp_path):
    trainer = _trainer(tmp_path, communication={"information_mode": "reactive_share"})
    trainer.env.reset()
    message = P2PMessage(1, 0, np.zeros(3), np.zeros(3), np.ones(2), 1, 0)
    cache = {0: {1: message}}
    trainer.env.states[0].position[:] = 0.0
    trainer.env.states[1].position[:] = trainer.config.safety.d_eng + 1.0
    trainer._purge_invalid_cache(cache)
    assert cache[0] == {}
    trainer.env.states[1].position[:] = 0.0
    trainer._purge_invalid_cache(cache)
    assert cache[0] == {}  # a fresh receive is required after re-entry


def test_checkpoint_restores_replay_cursor_and_rng(tmp_path):
    trainer = _trainer(tmp_path / "first", environment={"max_steps": 1})
    trainer.train(1)
    checkpoint = tmp_path / "latest.pt"
    trainer.save_checkpoint(checkpoint)
    expected = (random.random(), np.random.random(), float(torch.rand(())))
    restored = _trainer(tmp_path / "second", environment={"max_steps": 1})
    restored.load_checkpoint(checkpoint)
    assert len(restored.buffer) == len(trainer.buffer)
    assert len(restored.initial_state_buffer) == len(trainer.initial_state_buffer)
    assert restored.scenario_bank_cursor == trainer.scenario_bank_cursor
    assert restored.total_environment_steps == trainer.total_environment_steps
    assert (random.random(), np.random.random(), float(torch.rand(()))) == expected


def test_descriptor_uses_supplied_distance_thresholds():
    pair = [{"first": 0, "second": 1, "predicted_cpa_distance": 42.0, "predicted_cpa_time": 1.0, "candidate": True}]
    result = interaction_descriptor(2, pair, 1.0, np.ones(2), np.ones(2), 12.0, 35.0, 45.0)
    assert result["has_candidate_cpa_le_d_eng"]
    assert not result["has_candidate_cpa_le_d_warn"]
    assert result["descriptor_d_safe"] == 12.0
