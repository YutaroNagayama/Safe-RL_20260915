"""Torch helpers shared by SAC and TD3 actors."""

from __future__ import annotations


def require_torch():
    """Import Torch with an actionable optional-dependency error."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install uav-safe-marl[learn] to use neural policies") from exc
    return torch


def resolve_device(requested: str):
    """Resolve auto/cpu/cuda/mps without hard-coding the execution device."""
    torch = require_torch()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def graph_to_torch(graph, device):
    """Convert an unbatched or batched dense graph dictionary to Torch."""
    torch = require_torch()
    result = {}
    unbatched_dimensions = {"node_features": 2, "adjacency": 2, "edge_features": 3, "node_mask": 1}
    for key, value in graph.items():
        dtype = torch.bool if key in {"adjacency", "node_mask"} else torch.float32
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.ndim == unbatched_dimensions[key]:
            tensor = tensor.unsqueeze(0)
        result[key] = tensor
    return result
