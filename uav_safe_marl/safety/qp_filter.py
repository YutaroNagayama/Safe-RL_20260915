"""Decentralized HOCBF runtime filter and unconstrained learning correction."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uav_safe_marl.core.types import AgentState, HOCBFConstraint, SafetyResult
from .conflict_resolver import PriorityConflictResolver
from .hocbf import build_local_constraint, hocbf_terms
from .qp_backend import ActiveSetQPSolver
from .responsibility import CapabilityResponsibilityAllocator, effective_bounds


class HOCBFSafetyFilter:
    """Apply all engaged pair constraints in one local QP per UAV."""

    def __init__(self, d_safe: float, d_eng: float, k1: float, k2: float, dt: float, allocator: CapabilityResponsibilityAllocator, conflict_resolver_enabled: bool = True, speed_constraint_mode: str = "box") -> None:
        self.d_safe, self.d_eng = d_safe, d_eng
        self.k1, self.k2, self.dt = k1, k2, dt
        self.allocator = allocator
        self.conflict_resolver_enabled = conflict_resolver_enabled
        self.speed_constraint_mode = speed_constraint_mode
        self.solver = ActiveSetQPSolver()
        self.resolver = PriorityConflictResolver(self.solver)

    def constraints_for(self, agent_id: int, states: Sequence[AgentState], nominal: np.ndarray, active_mask: np.ndarray | None = None) -> list[HOCBFConstraint]:
        """Build local constraints for neighbours within ``d_eng``."""
        result: list[HOCBFConstraint] = []
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        if not active[agent_id]:
            return result
        own = states[agent_id]
        for neighbour_id, neighbour in enumerate(states):
            if neighbour_id == agent_id or not active[neighbour_id] or np.linalg.norm(own.position - neighbour.position) > self.d_eng:
                continue
            relative, _, _, q_ij = hocbf_terms(own, neighbour, self.d_safe, self.k1, self.k2)
            allocation = self.allocator.allocate_pair_details(own, neighbour, q_ij)
            result.append(build_local_constraint(neighbour_id, relative, q_ij, allocation.rho_first, nominal, allocation.feasible))
        return result

    def constraints_for_all(self, states: Sequence[AgentState], nominal: np.ndarray, active_mask: np.ndarray | None = None) -> list[list[HOCBFConstraint]]:
        """Build both directed constraints from every engaged unordered pair once."""
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        constraints: list[list[HOCBFConstraint]] = [[] for _ in states]
        for first_id in range(len(states)):
            if not active[first_id]:
                continue
            first = states[first_id]
            for second_id in range(first_id + 1, len(states)):
                if not active[second_id]:
                    continue
                second = states[second_id]
                relative, _, _, q_ij = hocbf_terms(first, second, self.d_safe, self.k1, self.k2)
                if np.linalg.norm(relative) > self.d_eng:
                    continue
                allocation = self.allocator.allocate_pair_details(first, second, q_ij)
                constraints[first_id].append(build_local_constraint(
                    second_id, relative, q_ij, allocation.rho_first,
                    nominal[first_id], allocation.feasible,
                ))
                constraints[second_id].append(build_local_constraint(
                    first_id, -relative, q_ij, allocation.rho_second,
                    nominal[second_id], allocation.feasible,
                ))
        return constraints

    def filter(self, agent_id: int, states: Sequence[AgentState], nominal: np.ndarray, active_mask: np.ndarray | None = None) -> SafetyResult:
        """Return the runtime-safe action and learning-only HOCBF correction."""
        nominal = np.asarray(nominal, dtype=np.float64)
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        if not active[agent_id]:
            zero = np.zeros(3, dtype=np.float64)
            return SafetyResult(zero, zero.copy())
        constraints = self.constraints_for(agent_id, states, nominal, active)
        return self.filter_with_constraints(agent_id, states, nominal, constraints, active)

    def filter_with_constraints(self, agent_id: int, states: Sequence[AgentState], nominal: np.ndarray, constraints: Sequence[HOCBFConstraint], active_mask: np.ndarray | None = None) -> SafetyResult:
        """Filter one agent using pair constraints precomputed for the current tick."""
        nominal = np.asarray(nominal, dtype=np.float64)
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        if not active[agent_id]:
            zero = np.zeros(3, dtype=np.float64)
            return SafetyResult(zero, zero.copy())
        own = states[agent_id]
        lower, upper = effective_bounds(own, self.dt, self.speed_constraint_mode)
        speed_kwargs = {}
        if self.speed_constraint_mode == "norm":
            speed_kwargs = {"velocity": own.velocity, "speed_limit": float(np.min(own.velocity_limit)), "dt": self.dt}
        safe, feasible = self.solver.solve(nominal, constraints, lower, upper, **speed_kwargs)
        if feasible:
            active = tuple(c.neighbour_id for c in constraints)
            result = SafetyResult(safe_action=safe, intervention_action=safe.copy(), active_neighbours=active, max_urgency=max((c.urgency for c in constraints), default=0.0))
        elif self.conflict_resolver_enabled:
            result = self.resolver.resolve(nominal, constraints, lower, upper, **speed_kwargs)
        else:
            bounded, _ = self.solver.solve(nominal, [], lower, upper, **speed_kwargs)
            result = SafetyResult(bounded, nominal.copy(), active_neighbours=tuple(c.neighbour_id for c in constraints), infeasible=True, max_urgency=max((c.urgency for c in constraints), default=0.0))

        # PDF slide 29: reconstruct the learning QP from every original HOCBF
        # constraint with U_eff removed.  Runtime skips must not hide pure safety
        # correction demand from the Intervention Cost.
        intervention_action, learning_feasible = self.solver.solve(nominal, constraints)
        if learning_feasible:
            result.intervention_action = intervention_action
        else:
            learning_result = self.resolver.resolve_learning(nominal, constraints)
            result.intervention_action = learning_result.safe_action
        result.responsibility_infeasible_count = sum(not c.responsibility_feasible for c in constraints)
        return result

    def project_physical(self, state: AgentState, nominal: np.ndarray) -> np.ndarray:
        """Project onto acceleration and next-speed limits without any HOCBF."""
        lower, upper = effective_bounds(state, self.dt, self.speed_constraint_mode)
        speed_kwargs = {}
        if self.speed_constraint_mode == "norm":
            speed_kwargs = {
                "velocity": state.velocity,
                "speed_limit": float(np.min(state.velocity_limit)),
                "dt": self.dt,
            }
        projected, feasible = self.solver.solve(np.asarray(nominal, dtype=np.float64), [], lower, upper, **speed_kwargs)
        if not feasible:
            raise RuntimeError("physical-only acceleration/speed constraints are infeasible")
        return projected

    def diagnose_intervention(self, agent_id: int, states: Sequence[AgentState], nominal: np.ndarray, active_mask: np.ndarray | None = None) -> SafetyResult:
        """Compute pure HOCBF correction without applying runtime U_eff."""
        nominal = np.asarray(nominal, dtype=np.float64)
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        if not active[agent_id]:
            zero = np.zeros(3, dtype=np.float64)
            return SafetyResult(zero, zero.copy())
        constraints = self.constraints_for(agent_id, states, nominal, active)
        return self.diagnose_with_constraints(agent_id, states, nominal, constraints, active)

    def diagnose_with_constraints(self, agent_id: int, states: Sequence[AgentState], nominal: np.ndarray, constraints: Sequence[HOCBFConstraint], active_mask: np.ndarray | None = None) -> SafetyResult:
        """Diagnose one agent from constraints precomputed for the current tick."""
        nominal = np.asarray(nominal, dtype=np.float64)
        active = np.ones(len(states), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
        if not active[agent_id]:
            zero = np.zeros(3, dtype=np.float64)
            return SafetyResult(zero, zero.copy())
        physical_action = self.project_physical(states[agent_id], nominal)
        intervention, feasible = self.solver.solve(nominal, constraints)
        if feasible:
            return SafetyResult(
                safe_action=physical_action,
                intervention_action=intervention,
                active_neighbours=tuple(c.neighbour_id for c in constraints),
                max_urgency=max((c.urgency for c in constraints), default=0.0),
                responsibility_infeasible_count=sum(not c.responsibility_feasible for c in constraints),
            )
        resolved = self.resolver.resolve_learning(nominal, constraints)
        resolved.safe_action = physical_action
        resolved.intervention_action = resolved.intervention_action.copy()
        resolved.responsibility_infeasible_count = sum(not c.responsibility_feasible for c in constraints)
        return resolved
