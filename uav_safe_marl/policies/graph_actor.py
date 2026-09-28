"""Graph-encoded shared SAC and TD3 actor heads."""

from __future__ import annotations

from uav_safe_marl.graph.neural import ActorGraphEncoder
from uav_safe_marl.graph.feature_layout import NODE_FEATURES
from .base import require_torch

torch = require_torch()
nn = torch.nn


class GraphSACActor(nn.Module):
    def __init__(self, edge_dim: int, graph_hidden: int, embedding_dim: int, layers: int, heads: int, policy_hidden: int) -> None:
        super().__init__()
        self.encoder = ActorGraphEncoder(NODE_FEATURES.graph_width, edge_dim, graph_hidden, embedding_dim, layers, heads)
        self.body = nn.Sequential(nn.Linear(NODE_FEATURES.graph_width + embedding_dim, policy_hidden), nn.ReLU(), nn.Linear(policy_hidden, policy_hidden), nn.ReLU())
        self.mean = nn.Linear(policy_hidden, 3)
        self.log_std = nn.Linear(policy_hidden, 3)

    def sample(self, graph, deterministic: bool = False):
        z = self.encoder(graph["node_features"], graph["adjacency"], graph["edge_features"], graph["node_mask"])
        features = self.body(torch.cat((graph["node_features"], z), dim=-1))
        mean, log_std = self.mean(features), self.log_std(features).clamp(-20.0, 2.0)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw) * graph["node_mask"].unsqueeze(-1)
        log_probability = distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        log_probability = log_probability.sum(dim=-1, keepdim=True) * graph["node_mask"].unsqueeze(-1)
        return action, log_probability, z


class GraphTD3Actor(nn.Module):
    def __init__(self, edge_dim: int, graph_hidden: int, embedding_dim: int, layers: int, heads: int, policy_hidden: int) -> None:
        super().__init__()
        self.encoder = ActorGraphEncoder(NODE_FEATURES.graph_width, edge_dim, graph_hidden, embedding_dim, layers, heads)
        self.head = nn.Sequential(nn.Linear(NODE_FEATURES.graph_width + embedding_dim, policy_hidden), nn.ReLU(), nn.Linear(policy_hidden, policy_hidden), nn.ReLU(), nn.Linear(policy_hidden, 3), nn.Tanh())

    def forward(self, graph):
        z = self.encoder(graph["node_features"], graph["adjacency"], graph["edge_features"], graph["node_mask"])
        action = self.head(torch.cat((graph["node_features"], z), dim=-1)) * graph["node_mask"].unsqueeze(-1)
        return action, z
