"""Single source of truth for actor/critic observation feature layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NodeFeatureLayout:
    position: slice = slice(0, 3)
    velocity: slice = slice(3, 6)
    goal_displacement: slice = slice(6, 9)
    acceleration_limit: slice = slice(9, 10)
    velocity_limit: slice = slice(10, 11)
    normalized_reference: slice = slice(11, 14)
    active_mask: slice = slice(14, 15)
    graph_width: int = 14
    pooled_direct_width: int = 15
    action_width: int = 3
    critic_timing_width: int = 4  # held normalized residual (3) + normalized phase (1)


NODE_FEATURES = NodeFeatureLayout()
EDGE_DIRECT_WIDTH = 7  # relative position, relative velocity, information age
EDGE_INFORMATION_AGE_INDEX = 6
