import os
from dataclasses import dataclass

import pytest
from typing_extensions import Self

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    RevoluteConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    CollisionCheckingConfig,
)


@pytest.fixture
def pr2_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "resources",
        "urdf",
    )
    pr2 = os.path.join(urdf_dir, "pr2_kinematic_tree.urdf")
    pr2_parser = URDFParser.from_file(file_path=pr2)
    world_with_pr2 = pr2_parser.parse()
    with world_with_pr2.modify_world():
        pr2_root = world_with_pr2.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_pr2.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_with_pr2
        )
        world_with_pr2.add_connection(c_root_bf)

    return world_with_pr2


@pytest.fixture()
def mini_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("root"))
        body2 = Body(name=PrefixedName("tip"))
        connection = RevoluteConnection.create_with_dofs(
            world=world, parent=body, child=body2, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world


@dataclass
class BoxBot(AbstractRobot):
    """
    Class that describes the Human Support Robot variant B (https://upmroboticclub.wordpress.com/robot/).
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def load_srdf(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        with world.modify_world():
            boxbot = cls(
                name=PrefixedName("boxbot", prefix=world.name),
                root=world.get_body_by_name("bot"),
                _world=world,
            )
            world.add_semantic_annotation(boxbot, skip_duplicates=True)


@pytest.fixture()
def box_bot_world():
    world = World()
    with world.modify_world():
        body = Body(
            name=PrefixedName("map"),
        )
        body2 = Body(
            name=PrefixedName("bot"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
            collision_config=CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=3
            ),
        )
        connection = OmniDrive.create_with_dofs(world=world, parent=body, child=body2)
        world.add_connection(connection)
        connection.has_hardware_interface = True

        environment = Body(
            name=PrefixedName("environment"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
        )
        env_connection = FixedConnection(
            parent=body,
            child=environment,
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(1),
        )
        world.add_connection(env_connection)
        BoxBot.from_world(world)

    return world
