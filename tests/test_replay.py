import numpy as np

from uav_safe_marl.core.types import Transition
from uav_safe_marl.replay.buffer import NumpyReplayBuffer
from uav_safe_marl.replay.initial_state_buffer import InitialStateBuffer


def test_replay_preserves_nominal_and_safe_actions():
    buffer = NumpyReplayBuffer(2, seed=7)
    transition = Transition(np.zeros((2, 3)), np.ones((2, 3)), np.ones((2, 3)) * 2, 1.0, 3.0, np.ones((2, 3)), False)
    buffer.add(transition)
    sample = buffer.sample(1)
    np.testing.assert_array_equal(sample["nominal_action"][0], transition.nominal_action)
    np.testing.assert_array_equal(sample["safe_action"][0], transition.safe_action)


def test_replay_keeps_terminated_and_truncated_separate_and_bootstraps_timeout():
    buffer = NumpyReplayBuffer(2, seed=0)
    buffer.add(Transition(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), 0.0, 0.0, np.zeros((1, 3)), False, True))
    sample = buffer.sample(1)
    assert sample["terminated"][0, 0] == 0.0
    assert sample["truncated"][0, 0] == 1.0
    assert sample["bootstrap_mask"][0, 0] == 1.0

    terminal = NumpyReplayBuffer(1, seed=0)
    terminal.add(Transition(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), 0.0, 0.0, np.zeros((1, 3)), True, False))
    terminal_sample = terminal.sample(1)
    assert terminal_sample["bootstrap_mask"][0, 0] == 0.0


def test_ring_buffer_overwrites_oldest_transition_and_samples_logical_contents():
    buffer = NumpyReplayBuffer(2, seed=4)
    for reward in (1.0, 2.0, 3.0):
        buffer.add(Transition(np.array([reward]), np.zeros((1, 3)), np.zeros((1, 3)), reward, 0.0, np.array([reward + 1]), False))
    batch = buffer.sample(2)
    assert sorted(batch["reward"][:, 0].tolist()) == [2.0, 3.0]


def test_sampling_rng_can_resume_without_copying_replay():
    buffer = NumpyReplayBuffer(5, seed=9)
    for reward in range(5):
        buffer.add(Transition(np.array([reward]), np.zeros((1, 3)), np.zeros((1, 3)), reward, 0.0, np.array([reward + 1]), False))
    state = buffer.sampling_state_dict()
    expected = buffer.sample(3)["reward"]
    buffer.load_sampling_state_dict(state)
    np.testing.assert_array_equal(buffer.sample(3)["reward"], expected)


def test_n_step_sampling_respects_episode_edges_and_timeout_bootstrap():
    buffer = NumpyReplayBuffer(8, seed=0)

    def add(index, cost, *, terminated=False, truncated=False):
        value = np.full((1, 3), float(index))
        buffer.add(Transition(
            state=value,
            nominal_action=np.zeros((1, 3)),
            safe_action=np.zeros((1, 3)),
            reward=0.0,
            cost=cost,
            next_state=np.full((1, 3), float(index + 1)),
            terminated=terminated,
            truncated=truncated,
            actor_phase=0,
            is_actor_decision_tick=True,
            held_residual_action=np.zeros((1, 3)),
            next_actor_phase=0,
            next_is_actor_decision_tick=True,
            next_held_residual_action=np.zeros((1, 3)),
        ))

    add(0, 1.0)
    add(1, 2.0, terminated=True)
    add(10, 4.0)
    add(11, 8.0, truncated=True)
    batch = buffer.sample_n_step(4, n_step=3, gamma=0.5)
    rows = {
        int(state[0, 0]): (
            float(cost), int(horizon), float(mask), int(next_state[0, 0])
        )
        for state, cost, horizon, mask, next_state in zip(
            batch["state"],
            batch["n_step_cost"][:, 0],
            batch["n_step_horizon"][:, 0],
            batch["n_step_bootstrap_mask"][:, 0],
            batch["n_step_next_state"],
            strict=True,
        )
    }
    assert rows[0] == (2.0, 2, 0.0, 2)
    assert rows[1] == (2.0, 1, 0.0, 2)
    assert rows[10] == (8.0, 2, 1.0, 12)
    assert rows[11] == (8.0, 1, 1.0, 12)


def test_initial_state_buffer_samples_the_configured_mu_and_restores_state():
    buffer = InitialStateBuffer(8, seed=3)
    buffer.add(np.full((4, 2), 2.0), 2, "n2")
    buffer.add(np.full((4, 2), 4.0), 4, "n4")
    batch = buffer.sample(2000, {2: 0.25, 4: 0.75})
    fraction_n4 = np.mean(batch["actual_agent_count"] == 4)
    assert 0.70 < fraction_n4 < 0.80
    snapshot = buffer.state_dict()
    restored = InitialStateBuffer(8, seed=99)
    restored.load_state_dict(snapshot)
    assert restored.occupancy_by_agent_count() == {2: 1, 4: 1}
