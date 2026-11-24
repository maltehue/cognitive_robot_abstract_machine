"""The errors the motion statechart tasks raise while they are built."""

from __future__ import annotations

from dataclasses import dataclass

from cramph.exceptions import NodeInitializationError

# %% pouring


@dataclass(kw_only=True)
class MissingOutflowEquationError(NodeInitializationError):
    """Raised when a task pours from a container that has no fill level to
    drain."""

    def error_message(self) -> str:
        return (
            f'The container of node "{self.node.unique_name}" has no fill connection '
            f"with an outflow equation."
        )

    def suggest_correction(self) -> str:
        return "Give the container a fill level via initialize_fill_level first."


@dataclass(kw_only=True)
class MissingInflowEquationError(NodeInitializationError):
    """Raised when a task needs a receiver's inflow equation but the receiver
    was never coupled to a source."""

    def error_message(self) -> str:
        return f'The receiver of node "{self.node.unique_name}" has no inflow equation.'

    def suggest_correction(self) -> str:
        return "Couple the receiver to a source via receive_outflow_from first."


@dataclass(kw_only=True)
class MissingExitSpeedError(NodeInitializationError):
    """Raised when neither the source nor the receiver's inflow equation
    provides an exit speed for the poured liquid."""

    def error_message(self) -> str:
        return (
            f'Node "{self.node.unique_name}" found no exit speed: the source reports '
            f"none and the receiver's inflow equation does not carry one."
        )

    def suggest_correction(self) -> str:
        return (
            "Couple the receiver via receive_outflow_from so a GatedInflowEquation is "
            "built."
        )


@dataclass(kw_only=True)
class NonPositiveClearanceError(NodeInitializationError):
    """Raised when a rim-clearance band does not lie entirely above the
    receiver's rim."""

    minimum_clearance: float
    """The rejected clearance floor, in metres."""

    def error_message(self) -> str:
        return (
            f'minimum_clearance of node "{self.node.unique_name}" must be positive to '
            f"keep the rims apart, got {self.minimum_clearance}."
        )

    def suggest_correction(self) -> str:
        return "Choose a clearance floor above zero."
