"""
Domain-specific effect types for articulated container manipulation (D_artic).
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.reasoning.body_motion_problem.types import (
    MonotoneDecreasingEffect,
    MonotoneIncreasingEffect,
)


@dataclass(eq=False, kw_only=True)
class OpenedEffect(MonotoneIncreasingEffect):
    """
    Effect achieved when an articulated container (drawer, door) is open.
    """


@dataclass(eq=False, kw_only=True)
class ClosedEffect(MonotoneDecreasingEffect):
    """
    Effect achieved when an articulated container (drawer, door) is closed.
    """
