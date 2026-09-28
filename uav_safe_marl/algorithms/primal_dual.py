"""Projected dual update for the CMDP constraint."""

from dataclasses import dataclass


@dataclass(slots=True)
class ProjectedDual:
    value: float = 0.0
    learning_rate: float = 0.01

    def update(self, violation: float) -> float:
        """Apply ``lambda <- [lambda + eta * g]_+``."""
        self.value = max(0.0, self.value + self.learning_rate * violation)
        return self.value

