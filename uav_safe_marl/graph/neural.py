"""Dependency-light trainable attention encoders for actor and centralized critics."""

from __future__ import annotations

import math

from uav_safe_marl.policies.base import require_torch
from uav_safe_marl.graph.feature_layout import NODE_FEATURES

torch = require_torch()
nn = torch.nn


class LocalP2PAttentionLayer(nn.Module):
    """Aggregate received edge messages into each receiver's local state."""

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(edge_dim, hidden_dim)
        self.value = nn.Linear(edge_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, adjacency, edge_features, node_mask):
        batch, agents, _ = hidden.shape
        query = self.query(hidden).view(batch, agents, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(edge_features).view(batch, agents, agents, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        value = self.value(edge_features).view(batch, agents, agents, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        scores = (query.unsqueeze(3) * key).sum(dim=-1) / math.sqrt(self.head_dim)
        valid = adjacency.unsqueeze(1) & node_mask[:, None, :, None] & node_mask[:, None, None, :]
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1) * valid
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        aggregated = (weights.unsqueeze(-1) * value).sum(dim=3).permute(0, 2, 1, 3).reshape(batch, agents, -1)
        updated = self.norm(hidden + self.output(aggregated))
        return torch.relu(updated) * node_mask.unsqueeze(-1)


class ActorGraphEncoder(nn.Module):
    """Small local P2P encoder producing the transmitted self representation z_i."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, embedding_dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([LocalP2PAttentionLayer(hidden_dim, edge_dim, heads) for _ in range(layers)])
        self.embedding = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, node_features, adjacency, edge_features, node_mask):
        hidden = torch.relu(self.node_projection(node_features)) * node_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, adjacency, edge_features, node_mask)
        return self.embedding(hidden) * node_mask.unsqueeze(-1)


class NodeAttentionLayer(nn.Module):
    """Permutation-equivariant centralized node-to-node attention layer."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, adjacency, node_mask):
        batch, agents, _ = hidden.shape
        query = self.query(hidden).view(batch, agents, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(hidden).view(batch, agents, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(hidden).view(batch, agents, self.heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.einsum("bhid,bhjd->bhij", query, key) / math.sqrt(self.head_dim)
        identity = torch.eye(agents, dtype=torch.bool, device=hidden.device).unsqueeze(0)
        valid_edges = (adjacency | identity) & node_mask[:, :, None] & node_mask[:, None, :]
        scores = scores.masked_fill(~valid_edges.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1) * valid_edges.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        aggregate = torch.einsum("bhij,bhjd->bhid", weights, value).permute(0, 2, 1, 3).reshape(batch, agents, -1)
        return torch.relu(self.norm(hidden + self.output(aggregate))) * node_mask.unsqueeze(-1)


class CriticGraphEncoder(nn.Module):
    """Larger centralized state-action graph encoder with invariant pooling."""

    def __init__(self, node_dim: int, hidden_dim: int, embedding_dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.input = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([NodeAttentionLayer(hidden_dim, heads) for _ in range(layers)])
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, node_features, adjacency, node_mask):
        hidden = torch.relu(self.input(node_features)) * node_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, adjacency, node_mask)
        encoded = self.output(hidden) * node_mask.unsqueeze(-1)
        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        mean = encoded.sum(dim=1) / denominator
        masked = encoded.masked_fill(~node_mask.unsqueeze(-1), -torch.inf)
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)


def centralized_adjacency(graph, mode: str, interaction_radius: float):
    """Build CTDE-only critic edges separately from Actor P2P edges."""
    mask = graph["node_mask"]
    valid = mask[:, :, None] & mask[:, None, :]
    if mode == "fully_connected":
        return valid
    positions = graph["node_features"][..., NODE_FEATURES.position]
    distances = torch.linalg.vector_norm(positions[:, :, None, :] - positions[:, None, :, :], dim=-1)
    safety_relation = distances <= interaction_radius
    actor_relation = graph["adjacency"] | graph["adjacency"].transpose(1, 2)
    return valid & (safety_relation | actor_relation)
