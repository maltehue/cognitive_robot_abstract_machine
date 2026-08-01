from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from coraplex.datastructures.dataclasses import Context

# %% provider


class WorldProvider(ABC):
    """
    Supplies the world and robot a new session operates on.

    Abstracting the world source keeps the server independent of any particular robot,
    environment, or asset-loading mechanism.
    """

    @abstractmethod
    def create_context(self) -> Context:
        """
        :return: A fresh context holding the world and robot for a new session.
        """


@dataclass
class Pr2WorldProvider(WorldProvider):
    """
    Builds a PR2 in an empty world, loaded from the robot's URDF.
    """

    def create_context(self) -> Context:
        # ``URDFParser`` pulls in the ROS ``xacro`` toolchain; importing it here keeps
        # that dependency at the deployment boundary so the rest of the package imports
        # without a ROS installation.
        from semantic_digital_twin.adapters.urdf import URDFParser
        from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
        from semantic_digital_twin.robots.pr2 import PR2
        from semantic_digital_twin.spatial_types.spatial_types import (
            HomogeneousTransformationMatrix,
        )
        from semantic_digital_twin.world_description.connections import (
            Connection6DoF,
            OmniDrive,
        )
        from semantic_digital_twin.world_description.world_entity import Body

        world = URDFParser.from_file(file_path=PR2.get_ros_file_path()).parse()
        robot = PR2.from_world(world)
        with world.modify_world():
            robot_root = world.root
            map_body = Body(name=PrefixedName("map"))
            localization_body = Body(name=PrefixedName("odom_combined"))
            world.add_connection(
                Connection6DoF.create_with_dofs(world, map_body, localization_body)
            )
            drive = OmniDrive.create_with_dofs(
                parent=localization_body, child=robot_root, world=world
            )
            world.add_connection(drive)
            drive.has_hardware_interface = True
        return Context(world=world, robot=robot)
