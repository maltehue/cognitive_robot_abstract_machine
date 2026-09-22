"""Run a mobile robot around a blocking box with CRAM's Navigate action."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import VisualizationBackend
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import execute_single
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% obstacle navigation rehearsal


class NavigationSceneBody(StrEnum):
    """Geometry authored by the navigation rehearsal."""

    BARRIER = "navigation_barrier"
    """Box separating the initial robot position and destination."""


@dataclass
class NavigationDemo(RobotDemonstration):
    """Demonstrate continuous base navigation around a visible obstacle."""

    collision_avoidance: bool = True
    """Keep collision avoidance enabled throughout the rehearsal."""

    distance: float = 3.0
    """Goal distance along the world's X axis, in meters."""

    barrier_size: Scale = field(default_factory=lambda: Scale(0.6, 0.8, 1.0))
    """Size of the obstacle separating start and destination, in meters."""

    def build_simulated_world(self) -> World:
        """Spawn the annotated mobile robot through its existing specification."""
        return WorldSpecification(
            robots=[RobotSpecification(self.used_robot)]
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        """Recognize the named obstacle before adding scene geometry.

        :param world: World inspected for the rehearsal's obstacle.
        """
        return len(world.get_bodies_by_name(NavigationSceneBody.BARRIER)) == 1

    def populate_scene(self, world: World) -> None:
        """Put a collidable box across the direct path to the destination.

        :param world: Mobile-robot world receiving the obstacle.
        """
        BodySpecification.box(
            NavigationSceneBody.BARRIER,
            self.barrier_size,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                self.distance / 2, 0, self.barrier_size.z / 2
            ),
        ).spawn(world)

    def build_context(self, world: World) -> Context:
        """Use the robot's native alternative motion mappings and plan validation.

        :param world: World containing the robot and obstacle.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            alternative_motion_mappings=self.alternative_motion_mappings,
            _debug=False,
            ros_node=self.ros_node,
        )

    def build_plan(self, context: Context) -> Plan:
        """Navigate to the far side of the obstacle through the normal action.

        :param context: Robot and world used by the navigation action.
        :return: Executable navigation plan.
        """
        target = Pose.from_xyz_rpy(self.distance, reference_frame=context.world.root)
        return execute_single(NavigateAction(target), context=context).plan


def main() -> None:
    """Drive the rehearsal and leave the result available in the live viewer."""
    demo = NavigationDemo(
        used_robot=PR2, default_visualization_backend=VisualizationBackend.CRAMERA
    )
    atexit.register(demo.stop_visualization)
    demo.run()


if __name__ == "__main__":
    main()
