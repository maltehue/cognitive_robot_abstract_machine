"""
mydemo4 — a working version of the Plan Builder demo.

The Plan Builder generated this for Garmi + a costmap-based TransportAction, which does
not run here: warehouse6 is a single hall mesh with no navigable-floor semantics, so the
transport's location costmap comes up empty, and Garmi's manipulation is mid-migration in
this checkout. So this is rebuilt on the pattern the shipped coraplex_warehouse_demo uses
and that runs cleanly: a UnitreeG1, standing on the hall floor, carrying the wrench with
explicit navigate/pick/place actions (no floor costmap, no mobile base to sink in).
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
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.view_manager import ViewManager
from coraplex.visualization import WorldVisualization
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color

_HERE = os.path.dirname(__file__)
_WORLDS = os.path.join(_HERE, "..", "..", "resources", "worlds")
_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")

HALL_MESH = os.path.join(_WORLDS, "warehouse6.glb")
WRENCH_MESH = os.path.join(_OBJECTS, "wrench.stl")

PELVIS_HEIGHT = 0.7923
"""How far the G1's pelvis (its root) stands above the floor with its legs at zero, so
its odom has to be lifted by this much for the feet to rest on the floor."""

STANDING_DISTANCE = 0.6
"""How far in front of a grasp pose the robot stands, within the G1's reach."""

WRENCH_START = Pose.from_xyz_rpy(2.4, -0.15, 0.83, yaw=np.pi)
"""Where the wrench starts, near the Plan Builder pose but at a reach the G1 can grasp
FRONT-on (upright, shelf height, turned to face the aisle the robot stands in)."""

WRENCH_DESTINATION = Pose.from_xyz_rpy(2.4, 4.1, 0.83, yaw=np.pi)
"""Where the wrench is carried to (the Plan Builder target, at the same reachable height)."""

ROBOT_START = Pose.from_xyz_rpy(1.0, 2.0, PELVIS_HEIGHT, yaw=0.0)
"""Where the robot starts, on the open hall floor."""


def build_world() -> World:
    """The hall with the G1 and the wrench standing free in front of it."""
    return WorldSpecification(
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START.to_homogeneous_matrix(),
            )
        ],
        objects=[
            BodySpecification.mesh("warehouse6", HALL_MESH),
            BodySpecification.mesh(
                "wrench.stl",
                WRENCH_MESH,
                color=Color(1.0, 0.62, 0.69),
                parent_T_self=WRENCH_START.to_homogeneous_matrix(),
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
        PELVIS_HEIGHT,
        yaw=yaw,
        reference_frame=world.root,
    )


def build_plan(world: World, robot: UnitreeG1):
    """Park, drive to the wrench, pick it, drive to the destination, place it, park."""
    wrench = world.get_body_by_name("wrench.stl")
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
            NavigateAction(standing_in_front_of(WRENCH_START, world)),
            PickUpAction(wrench, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            NavigateAction(standing_in_front_of(WRENCH_DESTINATION, world)),
            PlaceAction(wrench, placed(WRENCH_DESTINATION), Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    ).plan


world = build_world()
robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]

visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()
plan = build_plan(world, robot)
visualization.attach_plan(plan)

with simulated_robot:
    plan.perform()
