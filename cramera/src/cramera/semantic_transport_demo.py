"""
A small, reproducible semantic Transport using CRAM's stationary Tracy robot.

Run with ``cramera-live --viewer cramera/src/cramera/semantic_transport_demo.py`` from
the repository root after installing the CRAM environment and Tracy model.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
    VisualizationBackend,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from cramera.live.placement_surface import PlacementSurface
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


# %% canonical Tracy rehearsal
@dataclass
class SemanticTransportDemo(RobotDemonstration):
    """
    Transport a cube from Tracy's canonical pickup pose onto a named tabletop.

    The pickup geometry and grasp follow the existing coraplex_real_tracy demo. The
    destination is resolved by the semantic placement backend at execution time.
    """

    collision_avoidance: bool = True
    """
    Enable CRAM's collision avoidance throughout the rehearsal.
    """

    placement_clearance: float = 0.005
    """
    Gripper clearance above the destination table in this simulated scene, in meters.
    """

    def build_simulated_world(self) -> World:
        """
        Spawn the annotated robot through CRAM's world specification.
        """
        return WorldSpecification(
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(),
                )
            ]
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        """
        Recognize an existing authored cube through CRAM's body-name lookup.

        :param world: World inspected before adding rehearsal geometry.
        """
        return len(world.get_bodies_by_name("box2.stl")) == 1

    def populate_scene(self, world: World) -> None:
        """
        Create the pickup cube and an annotated supporting tabletop.

        :param world: Robot world receiving the local rehearsal geometry.
        """
        BodySpecification.box(
            "box2.stl",
            Scale(0.1, 0.1, 0.1),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(0.8, 0.25, 0.92),
            connection_specification=Connection6DoFSpecification(),
        ).spawn(world)
        BodySpecification.box(
            "placement_table",
            Scale(0.2, 0.2, 0.04),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(0.8, 0, 0.95),
        ).spawn(world)
        with world.modify_world():
            world.add_semantic_annotation(
                Table(root=world.get_body_by_name("placement_table"))
            )
            self.configure_contacts(world)

    def configure_contacts(self, world: World) -> None:
        """
        Permit the intended grasp and support contacts while avoiding the tabletop.

        :param world: World containing the cube, destination table and annotated robot.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        cube = world.get_body_by_name("box2.stl")
        surface = world.get_body_by_name("placement_table")
        gripper = robot.left_arm.end_effector.bodies_with_collision
        world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(body_group_a=gripper, body_group_b=[cube])
        )
        world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(body_group_a=[cube], body_group_b=[surface])
        )
        world.collision_manager.add_default_rule(
            AvoidCollisionBetweenGroups(
                body_group_a=gripper,
                body_group_b=[surface],
                buffer_zone_distance=self.placement_clearance,
                violated_distance=0.0,
            )
        )

    def build_context(self, world: World) -> Context:
        """
        Use Tracy's existing simulated execution context and grasp annotations.

        :param world: World containing the robot and authored scene.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        return Context(
            world=world,
            robot=robot,
            _debug=False,
            ros_node=self.ros_node,
            evaluate_conditions=False,
        )

    def build_plan(self, context: Context) -> Plan:
        """
        Resolve a semantic destination and execute CRAM's Transport action.

        :param context: Robot and world providing the actual motion and grasp execution.
        :return: Plan whose target has no hand-entered object pose.
        """
        body = context.world.get_body_by_name("box2.stl")
        destination = PlacementSurface(
            world=context.world,
            body=body,
            surface_type=Table,
            surface_name="placement_table",
        )
        grasp = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.TOP,
            context.robot.left_arm.end_effector,
        )
        return sequential(
            [
                a(TransportAction)(
                    object_designator=body,
                    target_location=variable(Pose, domain=destination),
                    arm=Arms.LEFT,
                    grasp_description=grasp,
                    look_at_operation_site=False,
                )
            ],
            context=context,
        ).plan


def main() -> None:
    """
    Run the rehearsal and leave its world available to the live viewer.
    """
    SemanticTransportDemo(
        used_robot=Tracy,
        default_visualization_backend=VisualizationBackend.CRAMERA,
    ).run()


if __name__ == "__main__":
    main()
