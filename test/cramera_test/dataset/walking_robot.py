"""
The standing robot on an omnidirectional drive, the way a humanoid that walks is
spawned: its drive carries its root away from the origin of its localization frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import MobileBase
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from typing_extensions import List, Self

from .standing_robot import StandingRobot


@dataclass(eq=False)
class WalkingBase(MobileBase[OmniDrive]):
    """
    The base that carries the robot over the floor.
    """

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(root=robot_root)

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class WalkingRobot(StandingRobot, HasMobileBase[WalkingBase]):
    """
    The standing robot on a drive, so that odometry written into the drive carries it
    away from where it was placed.
    """
