"""
A Unitree G1 shuttling a labelled screw box between a shelf and a workbench.

The Garmi's version of this run is ``garmi_screw_box_shuttle.py``; it carries the same box
over the same two stands. What the G1 needs on top:

* its root is the pelvis, so every pose it stands in is lifted by
  :data:`PELVIS_HEIGHT_ABOVE_FLOOR`, or the robot sinks through the floor;
* its torso has no named states to move to, and it bends while reaching, so the torso
  joints are driven back to zero after each pick and place instead;
* it stands closer to what it grasps, since it has a humanoid's reach rather than an
  arm on a lift.
"""

import os
from dataclasses import dataclass

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, VisualizationBackend
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans import MoveJointsMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

_HERE = os.path.dirname(__file__)
_WORLDS = os.path.join(_HERE, "..", "..", "resources", "worlds")
_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")

ENV_FILE = "warehouse6.urdf"

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""
How far the G1's pelvis stands above the floor with all leg joints at zero.

The pelvis is the robot's root, so every pose it is placed or navigated to carries this
height.
"""

ROBOT_XY = (1.5, 2.5)
"""
Where the robot starts, on the open hall floor.
"""

BASE_MAY_DRIVE_WHILE_REACHING = False
"""
Whether the base may drive to help an arm reach (whole-body control).

Off, so the robot grasps from the pose it was navigated to instead of walking on into the
shelving to shorten the reach.
"""

ROUND_TRIPS = 4
"""
How often the box travels to the workbench and back to the shelf.
"""

BOX_MESH = "screw_box.obj"
"""
The labelled cardboard box that is carried.
"""

CARRYING_ARM = Arms.LEFT
"""
The arm that picks and places, one hand throughout.
"""

STANDING_DISTANCE = 0.55
"""
How far in front of the box's grasped side the robot stands, in metres.

Within the G1's reach with the box at shelf height.
"""

ON_THE_SHELF = Pose.from_xyz_rpy(2.35, -0.05, 0.71)
"""
Where the box starts and is returned to.
"""

ON_THE_WORKBENCH = Pose.from_xyz_rpy(5.64, -0.93, 0.74, yaw=1.623)
"""
Where the box is delivered to.
"""


@dataclass(kw_only=True)
class ScrewBoxShuttleDemonstration(RobotDemonstration):
    """
    Carries one labelled box back and forth between two stands.
    """

    def build_simulated_world(self) -> World:
        return WorldSpecification.from_urdf(
            os.path.join(_WORLDS, ENV_FILE),
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                        ROBOT_XY[0], ROBOT_XY[1], PELVIS_HEIGHT_ABOVE_FLOOR
                    ),
                ),
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        try:
            world.get_body_by_name(BOX_MESH)
        except Exception:
            return False
        return True

    def populate_scene(self, world: World) -> None:
        # free to move (Connection6DoF), so the robot can pick it up and carry it
        BodySpecification.mesh(
            BOX_MESH,
            os.path.join(_OBJECTS, BOX_MESH),
            parent_T_self=ON_THE_SHELF.to_homogeneous_matrix(),
            connection_specification=Connection6DoFSpecification(),
        ).spawn(world)

    def build_context(self, world: World) -> Context:
        with world.modify_world():
            WorldReasoner(world).reason()
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        if isinstance(robot, HasMobileBase):
            robot.mobile_base.full_body_controlled = BASE_MAY_DRIVE_WHILE_REACHING
        context = Context(
            world=world, robot=robot, _debug=False, ros_node=self.ros_node
        )
        context.evaluate_conditions = False
        return context

    def build_plan(self, context: Context) -> PlanNode:
        world = context.world  # bodies/poses below are resolved against it
        box = world.get_body_by_name(BOX_MESH)
        steps = [ParkArmsAction(Arms.BOTH)]
        for _ in range(ROUND_TRIPS):
            steps.extend(self.carry(context, box, ON_THE_SHELF, ON_THE_WORKBENCH))
            steps.extend(self.carry(context, box, ON_THE_WORKBENCH, ON_THE_SHELF))
        return sequential(steps, context=context).plan

    def carry(
        self, context: Context, box: Body, origin: Pose, destination: Pose
    ) -> list:
        """
        One leg: fetch the box from ``origin`` and put it down at ``destination``.

        Where the box stands is named rather than read off the body, because the plan is
        built before any of it runs: by the second leg the body is still on the shelf,
        while the leg is the one that fetches it from the workbench.

        :param context: The plan context the actions are built against.
        :param box: The body being carried.
        :param origin: Where the box stands at the start of this leg.
        :param destination: Where the box is put down.
        """
        standing_at_the_box = self.standing_in_front_of(context, origin)
        standing_at_the_destination = self.standing_in_front_of(context, destination)
        return [
            NavigateAction(standing_at_the_box),
            PickUpAction(box, CARRYING_ARM, self.grasp_for(context, box, origin)),
            ParkArmsAction(Arms.BOTH),
            self.straighten_the_torso(context),
            NavigateAction(standing_at_the_destination),
            PlaceAction(
                box, self.against_world_root(context, destination), CARRYING_ARM
            ),
            ParkArmsAction(Arms.BOTH),
            self.straighten_the_torso(context),
        ]

    @staticmethod
    def straighten_the_torso(context: Context) -> MoveJointsMotion:
        """
        Drive the torso joints back to zero.

        The G1 leans in to reach, and carries that lean into the next leg unless it is
        undone; it has no named torso states to move to instead.

        :param context: The plan context, for the robot's torso.
        """
        connections = context.robot.torso.active_connections
        return MoveJointsMotion(
            names=[connection.name for connection in connections],
            positions=[0.0] * len(connections),
        )

    def grasp_for(self, context: Context, box: Body, origin: Pose) -> GraspDescription:
        """
        How to take the box: the side to approach from is left to the robot's reach, so
        the box is grasped from a side the robot can stand on however it was put down.

        :param context: The plan context, for the robot's end effector.
        :param box: The body being grasped.
        :param origin: Where the box stands when this leg grasps it.
        """
        return GraspDescription.robot_relative_default(
            ViewManager.get_end_effector_view(CARRYING_ARM, context.robot),
            self.against_world_root(context, origin),
            box,
        )

    def standing_in_front_of(self, context: Context, pose: Pose) -> Pose:
        """
        The pose the robot stands in to reach ``pose``: a stride in front of the side it
        approaches from, at pelvis height.

        :param context: The plan context holding the world the pose belongs to.
        :param pose: The pose being reached for.
        """
        yaw = float(pose.yaw)
        return Pose.from_xyz_rpy(
            float(pose.x) - STANDING_DISTANCE * np.cos(yaw),
            float(pose.y) - STANDING_DISTANCE * np.sin(yaw),
            PELVIS_HEIGHT_ABOVE_FLOOR,
            yaw=yaw,
            reference_frame=context.world.root,
        )

    @staticmethod
    def against_world_root(context: Context, pose: Pose) -> Pose:
        """
        A pose from this file, expressed in the world it is used in.

        :param context: The plan context holding that world.
        :param pose: The pose to re-reference.
        """
        return Pose(
            pose.to_position(),
            pose.to_quaternion(),
            reference_frame=context.world.root,
        )


def main() -> None:
    """
    Run the demonstration.

    RobotDemonstration.run() acquires the world, starts the visualization backend,
    attaches the plan and performs it. The backend defaults to CRAMERA (the browser
    viewer); CORAPLEX_VISUALIZATION overrides it, so `cramera-live` works unchanged
    and you can also force RVIZ / NONE from the outside.
    """
    ScrewBoxShuttleDemonstration(
        used_robot=UnitreeG1,
        collision_avoidance=False,
        default_visualization_backend=VisualizationBackend.CRAMERA,
    ).run()


if __name__ == "__main__":
    main()
