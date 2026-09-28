import numpy as np

from uav_safe_marl.communication.message import P2PMessage
from uav_safe_marl.communication.p2p import P2PCommunication
from uav_safe_marl.core.types import AgentState
from uav_safe_marl.graph.unimp import UniMPEncoder
from uav_safe_marl.graph.graph_builder import message_features
from uav_safe_marl.runners.trainer import SafeRLTrainer


def state():
    return AgentState(np.zeros(3), np.zeros(3), np.ones(3), np.ones(3), np.ones(3))


def message(sender, position, embedding):
    return P2PMessage(sender, 0, np.asarray(position, float), np.zeros(3), np.asarray(embedding, float), 1, 0)


def test_unimp_has_fixed_dimension_and_is_neighbour_permutation_invariant():
    encoder = UniMPEncoder(enabled=True, embedding_dim=2)
    messages = [message(1, [1, 0, 0], [2, 3]), message(2, [0, 1, 0], [4, 5])]
    forward = encoder.encode(state(), messages)
    reverse = encoder.encode(state(), list(reversed(messages)))
    assert forward.shape == reverse.shape == (encoder.output_dim,)
    np.testing.assert_allclose(forward, reverse)


def test_message_matrix_width_is_fixed_even_without_neighbours():
    assert message_features([], np.zeros(3), embedding_dim=5).shape == (0, 12)
    matrix = message_features([message(1, [1, 0, 0], [2, 3])], np.zeros(3), embedding_dim=5)
    assert matrix.shape == (1, 12)


def test_lagged_neighbour_embedding_changes_two_hop_representation():
    encoder = UniMPEncoder(enabled=True, embedding_dim=2)
    first = encoder.encode(state(), [message(1, [1, 0, 0], [0, 0])])
    second = encoder.encode(state(), [message(1, [1, 0, 0], [7, 0])])
    assert not np.array_equal(first, second)


def test_embedding_age_increases_across_heartbeat_only_updates():
    rich = message(1, [1, 0, 0], [2, 3])
    cache = {0: {1: rich}}
    heartbeat = P2PMessage(1, 0, np.ones(3), np.zeros(3), np.empty(0), 1, 1)
    SafeRLTrainer._merge_messages(cache, {0: [heartbeat]})
    assert cache[0][1].age_steps == 2
    heartbeat2 = P2PMessage(1, 0, np.ones(3), np.zeros(3), np.empty(0), 1, 2)
    SafeRLTrainer._merge_messages(cache, {0: [heartbeat2]})
    assert cache[0][1].age_steps == 3


def test_prior_share_uses_injected_los_candidates_not_all_to_all():
    states = [state(), state(), state()]
    communication = P2PCommunication("prior_share", 0, 25.0, 1)
    communication.reset([(0, 1)])
    communication.send(states, np.zeros((3, 2)), 0)
    delivered = communication.receive(1)
    assert communication.message_count == 2
    assert {message.sender_id for message in delivered[0]} == {1}
    assert {message.sender_id for message in delivered[1]} == {0}
    assert 2 not in delivered


def test_new_embedding_is_never_visible_to_a_peer_in_the_same_tick():
    states = [state(), state()]
    communication = P2PCommunication("prior_share", 0, 25.0, 1)
    communication.reset([(0, 1)])
    z_t = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert communication.receive(0) == {}
    communication.send(states, z_t, 0)
    assert communication.receive(0) == {}
    delivered = communication.receive(1)
    np.testing.assert_array_equal(delivered[1][0].embedding, z_t[0])
    np.testing.assert_array_equal(delivered[0][0].embedding, z_t[1])
    assert all(message.sent_step == 0 and message.age_steps == 1 for messages in delivered.values() for message in messages)


def test_queued_messages_from_exited_agents_are_discarded():
    states = [state(), state()]
    communication = P2PCommunication("prior_share", 1, 25.0, 1)
    communication.reset([(0, 1)])
    communication.send(states, np.zeros((2, 2)), 0, np.array([True, True]))
    assert communication.receive(1, np.array([False, True])) == {}


def test_preflight_candidate_is_retained_while_both_agents_are_active():
    states = [
        AgentState(np.array([-100.0, 0, 0]), np.zeros(3), np.ones(3), np.ones(3), np.ones(3)),
        AgentState(np.array([100.0, 0, 0]), np.zeros(3), np.ones(3), np.ones(3), np.ones(3)),
    ]
    communication = P2PCommunication("prior_share", 1, 50.0, 1)
    communication.reset([(0, 1)])
    communication.send(states, np.zeros((2, 2)), 10, np.array([True, True]))
    delivered = communication.receive(11, np.array([True, True]))
    assert communication.message_count == 2
    assert set(delivered) == {0, 1}
    communication.send(states, np.zeros((2, 2)), 12, np.array([False, True]))
    assert communication.message_count == 2


def test_none_mode_disables_all_p2p_messages_even_for_candidates():
    states = [state(), state()]
    communication = P2PCommunication("none", 1, 50.0, 1)
    communication.reset([(0, 1)])
    communication.send(states, np.zeros((2, 2)), 0, np.array([True, True]))
    assert communication.message_count == 0
    assert communication.transmitted_bytes == 0
    assert communication.receive(1) == {}
