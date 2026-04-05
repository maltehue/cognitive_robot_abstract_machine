"""
TEE class definition for articulated container manipulation (D_artic).

D_artic scopes the BMP to the domain of opening and closing articulated
containers (cupboard doors, drawers, dishwasher and oven doors) in kitchen
environments using mobile manipulators under rigid-body kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.reasoning.body_motion_problem.types import TEEClass


@dataclass
class ArticulatedContainerTEEClass(TEEClass):
    """
    TEE class for articulated container manipulation.

    Scopes the BMP to open/close tasks on articulated containers under
    rigid-body kinematics (Φ_artic). Validity intervals I_Φ are defined
    by the joint position and velocity limits of the environment model.
    """

    task_types: frozenset = field(default=frozenset({"open", "close"}))
