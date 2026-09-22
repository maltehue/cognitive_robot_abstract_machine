"""
mydemo5 — a working version of the Plan Builder milk demo (PR2, warehouse6).

The Plan Builder used a costmap-based TransportAction, which fails in warehouse6: the
hall is a single mesh with no navigable-floor semantics, so the transport's location
costmap comes up empty ("Merged locations is empty" -> EmptyUnderspecified). This spells
the transport out with explicit navigate/pick/place actions, which need no floor costmap.
The bowl was dropped; only the milk is carried.
"""
import os

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
    VisualizationBackend,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.view_manager import ViewManager
from coraplex.visualization import WorldVisualization
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color

_HERE = os.path.dirname(__file__)
_WORLDS = os.path.join(_HERE, "..", "..", "resources", "worlds")
_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")

HALL_MESH = os.path.join(_WORLDS, "warehouse6.glb")
MILK_MESH = os.path.join(_OBJECTS, "milk.stl")

STANDING_DISTANCE = 0.6
"""How far in front of a grasp pose the robot stands, within the PR2's reach."""

MILK_START = Pose.from_xyz_rpy(2.45, -0.1, 0.7, yaw=0.0)
"""Where the milk starts (Plan Builder pose). Turned so the robot approaches from the
aisle side (standing near x=1.85) rather than from x=3.05, where it stood inside a rack."""

MILK_DESTINATION = Pose.from_xyz_rpy(2.45, -1.05, 0.7, yaw=0.0)
"""Where the milk is carried to (Plan Builder target), approached from the same side."""

ROBOT_START = Pose.from_xyz_rpy(1.5, 2.5, 0.0, yaw=0.0)
"""Where the PR2 starts, on the open hall floor."""


def build_world() -> World:
    """The hall with the PR2 and the milk standing free in front of it."""
    return WorldSpecification(
        robots=[
            RobotSpecification(
                semantic_annotation_type=PR2,
                world_T_odom=ROBOT_START.to_homogeneous_matrix(),
            )
        ],
        objects=[
            BodySpecification.mesh("warehouse6", HALL_MESH),
            BodySpecification.mesh(
                "milk.stl",
                MILK_MESH,
                color=Color(0.6, 0.63, 0.68),
                parent_T_self=MILK_START.to_homogeneous_matrix(),
                # free to move, so the robot can pick it up and carry it
                connection_specification=Connection6DoFSpecification(),
            ),
        ],
    ).to_domain_object()


def standing_in_front_of(pose: Pose, world: World) -> Pose:
    """The base pose the robot stands in to reach ``pose`` with a FRONT grasp."""
    yaw = float(pose.yaw)
    return Pose.from_xyz_rpy(
        pose.x - STANDING_DISTANCE * np.cos(yaw),
        pose.y - STANDING_DISTANCE * np.sin(yaw),
        0.0,
        yaw=yaw,
        reference_frame=world.root,
    )


def build_plan(world: World, robot: PR2):
    """Park, raise the torso, drive to the milk, pick it, carry it, place it, park."""
    milk = world.get_body_by_name("milk.stl")
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(Arms.LEFT, robot),
    )
    context = Context(world=world, robot=robot, evaluate_conditions=False)

    def placed(pose: Pose) -> Pose:
        return Pose(pose.to_position(), pose.to_quaternion(), reference_frame=world.root)

    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            NavigateAction(standing_in_front_of(MILK_START, world)),
            PickUpAction(milk, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            NavigateAction(standing_in_front_of(MILK_DESTINATION, world)),
            PlaceAction(milk, placed(MILK_DESTINATION), Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    ).plan


world = build_world()
robot = world.get_semantic_annotations_by_type(PR2)[0]

visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()
plan = build_plan(world, robot)
visualization.attach_plan(plan)

with simulated_robot:
    plan.perform()
