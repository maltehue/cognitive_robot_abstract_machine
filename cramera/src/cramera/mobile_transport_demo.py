"""Rehearse a complete semantic transport with CRAM's PR2 mobile manipulator."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, VisualizationBackend
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from cramera.live.placement_surface import PlacementSurface
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale


# %% semantic mobile transport scene
class TransportSceneBody(StrEnum):
    """Objects shared by the recorded scene and semantic task."""

    OBJECT = "transport_cube"
    """Movable object carried between the tables."""
    SOURCE = "pickup_table"
    """Initial supporting surface."""
    DESTINATION = "delivery_table"
    """Named supporting surface resolved during planning."""


@dataclass
class MobileTransportDemo(RobotDemonstration):
    """Carry an object between two tables through the normal Transport action."""

    collision_avoidance: bool = True
    """Enable the native collision controller throughout execution."""
    table_distance: float = 2.0
    """Distance between the table centers, in meters."""
    source_distance: float = 1.5
    """Initial table center in front of the robot, in meters."""
    table_height: float = 0.8
    """Height of the tabletop center, in meters."""
    table_size: Scale = field(default_factory=lambda: Scale(0.5, 0.5, 0.04))
    """Dimensions of each horizontal supporting tabletop."""
    object_size: Scale = field(default_factory=lambda: Scale(0.06, 0.06, 0.14))
    """Dimensions of the graspable transport object."""
    table_color: Color = field(default_factory=lambda: Color(0.26, 0.31, 0.36))
    """Muted surface color distinguishing the tables from the robot."""
    object_color: Color = field(default_factory=lambda: Color(0.95, 0.42, 0.12))
    """Contrasting color making the carried object visible during playback."""
    gripper_clearance: float = 0.005
    """Required gripper clearance from the supporting tabletops, in meters."""

    def build_simulated_world(self) -> World:
        """Spawn the annotated robot using its existing world specification."""
        return WorldSpecification(
            robots=[RobotSpecification(semantic_annotation_type=self.used_robot)]
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        """Identify an already populated transport scene.

        :param world: World inspected for the carried object.
        """
        return len(world.get_bodies_by_name(TransportSceneBody.OBJECT)) == 1

    def populate_scene(self, world: World) -> None:
        """Add a supported object and two semantically named tables.

        :param world: Robot world receiving the transport scene.
        """
        for name, distance in (
            (TransportSceneBody.SOURCE, self.source_distance),
            (
                TransportSceneBody.DESTINATION,
                self.source_distance + self.table_distance,
            ),
        ):
            BodySpecification.box(
                name,
                self.table_size,
                color=self.table_color,
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    distance, 0, self.table_height
                ),
            ).spawn(world)
            with world.modify_world():
                world.add_semantic_annotation(Table(root=world.get_body_by_name(name)))
        BodySpecification.box(
            TransportSceneBody.OBJECT,
            self.object_size,
            color=self.object_color,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                self.source_distance,
                0,
                self.table_height + (self.table_size.z + self.object_size.z) / 2,
            ),
            connection_specification=Connection6DoFSpecification(),
        ).spawn(world)
        with world.modify_world():
            self.configure_contacts(world)

    def configure_contacts(self, world: World) -> None:
        """Permit intended grasp and support contact while retaining table avoidance.

        :param world: World containing the robot, object and both tables.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        body = world.get_body_by_name(TransportSceneBody.OBJECT)
        tables = [
            world.get_body_by_name(name)
            for name in (TransportSceneBody.SOURCE, TransportSceneBody.DESTINATION)
        ]
        gripper = robot.left_arm.end_effector.bodies_with_collision
        world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(body_group_a=gripper, body_group_b=[body])
        )
        world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(body_group_a=[body], body_group_b=tables)
        )
        world.collision_manager.add_default_rule(
            AvoidCollisionBetweenGroups(
                body_group_a=gripper,
                body_group_b=tables,
                buffer_zone_distance=self.gripper_clearance,
                violated_distance=0.0,
            )
        )

    def build_context(self, world: World) -> Context:
        """Separate base navigation from reaching with the arm.

        :param world: Annotated robot world used for task execution.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        robot.mobile_base.full_body_controlled = False
        return Context(
            world=world,
            robot=robot,
            alternative_motion_mappings=self.alternative_motion_mappings,
            _debug=False,
            ros_node=self.ros_node,
        )

    def build_plan(self, context: Context) -> Plan:
        """Compose a transport whose destination comes from the named semantic table.

        :param context: Robot and world resolving the transport's action parameters.
        :return: Executable torso preparation and complete Transport action.
        """
        body = context.world.get_body_by_name(TransportSceneBody.OBJECT)
        destination = PlacementSurface(
            world=context.world,
            body=body,
            surface_type=Table,
            surface_name=TransportSceneBody.DESTINATION,
        )
        return sequential(
            [
                MoveTorsoAction(TorsoState.HIGH),
                a(TransportAction)(
                    object_designator=body,
                    target_location=variable(Pose, domain=destination),
                    arm=Arms.LEFT,
                    look_at_operation_site=False,
                ),
            ],
            context=context,
        ).plan


# %% runnable presentation
def main() -> None:
    """Execute the recorded mobile transport and retain the world for inspection."""
    demonstration = MobileTransportDemo(
        used_robot=PR2, default_visualization_backend=VisualizationBackend.CRAMERA
    )
    atexit.register(demonstration.stop_visualization)
    demonstration.run()


if __name__ == "__main__":
    main()
