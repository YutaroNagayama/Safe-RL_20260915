"""Independent centralized state-action graph critics."""

from __future__ import annotations

from uav_safe_marl.graph.neural import CriticGraphEncoder
from uav_safe_marl.graph.feature_layout import NODE_FEATURES
from uav_safe_marl.policies.base import require_torch

torch = require_torch()
nn = torch.nn


class GraphScalarCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.encoder = CriticGraphEncoder(
            NODE_FEATURES.graph_width + NODE_FEATURES.critic_timing_width + NODE_FEATURES.action_width,
            hidden_dim, embedding_dim, layers, heads,
        )
        self.head = nn.Sequential(nn.Linear(2 * embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, graph, nominal_action, adjacency):
        node_features = graph["node_features"]
        expected_state_width = NODE_FEATURES.graph_width + NODE_FEATURES.critic_timing_width
        if node_features.shape[-1] == NODE_FEATURES.graph_width:
            padding = node_features.new_zeros((*node_features.shape[:-1], NODE_FEATURES.critic_timing_width))
            node_features = torch.cat((node_features, padding), dim=-1)
        elif node_features.shape[-1] != expected_state_width:
            raise ValueError(f"unexpected critic node width: {node_features.shape[-1]}")
        nodes = torch.cat((node_features, nominal_action), dim=-1)
        return self.head(self.encoder(nodes, adjacency, graph["node_mask"]))


class GraphCostCriticEnsemble(nn.Module):
    """Cost ensemble whose graph encoders and heads are fully independent."""

    def __init__(self, size: int, hidden_dim: int, embedding_dim: int, layers: int, heads: int) -> None:
        super().__init__()
        if size < 2:
            raise ValueError("cost ensemble size must be at least two")
        self.members = nn.ModuleList([GraphScalarCritic(hidden_dim, embedding_dim, layers, heads) for _ in range(size)])

    def forward(self, graph, action, adjacency):
        return torch.stack([member(graph, action, adjacency) for member in self.members], dim=0)

    def ucb(self, graph, action, adjacency, beta: float):
        values = self(graph, action, adjacency)
        return values.mean(dim=0) + beta * values.std(dim=0, unbiased=False)
