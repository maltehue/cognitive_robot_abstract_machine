"""
The errors the motion statechart tasks raise while they are built.
"""

from __future__ import annotations

from dataclasses import dataclass

from cramph.exceptions import NodeInitializationError

# %% pouring


@dataclass(kw_only=True)
class MissingFillLevelError(NodeInitializationError):
    """
    Raised when a task pours from a container that has no fill level to drain.
    """

    def error_message(self) -> str:
        return f'The container of node "{self.node.unique_name}" has no fill level.'

    def suggest_correction(self) -> str:
        return "Give the container a fill level via initialize_fill_level first."


@dataclass(kw_only=True)
class MissingSourceError(NodeInitializationError):
    """
    Raised when a task fills a receiver that no container pours into.
    """

    def error_message(self) -> str:
        return (
            f'No container pours into the receiver of node "{self.node.unique_name}".'
        )

    def suggest_correction(self) -> str:
        return "Let a source pour into the receiver via receive_outflow_from first."


@dataclass(kw_only=True)
class NonPositiveClearanceError(NodeInitializationError):
    """
    Raised when a rim-clearance band does not lie entirely above the receiver's rim.
    """

    minimum_clearance: float
    """
    The rejected clearance floor, in metres.
    """

    def error_message(self) -> str:
        return (
            f'minimum_clearance of node "{self.node.unique_name}" must be positive to '
            f"keep the rims apart, got {self.minimum_clearance}."
        )

    def suggest_correction(self) -> str:
        return "Choose a clearance floor above zero."
