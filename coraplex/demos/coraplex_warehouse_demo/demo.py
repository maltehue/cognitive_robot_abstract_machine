"""
A Unitree G1 takes a crate off a warehouse shelf, sets it on a workbench, and carries
the wrench out of it to another shelf.

The hall, the crate and the wrench are the downloaded collections, prepared once by
``prepare_assets.py``: the hall with its roof cut off so the run can be watched from
above, and the crate with its lid cut off so what is inside can be seen and reached.
"""

from __future__ import annotations

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
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.view_manager import ViewManager
from coraplex.visualization import WorldVisualization
from pathlib import Path
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale

# %% the meshes prepare_assets.py wrote

RESOURCES = Path(__file__).parent / "resources"
"""
Where the prepared hall, crate and wrench are.
"""

HALL_COLOUR = Color(0.62, 0.63, 0.60)
"""
What the hall is painted.

It was downloaded as an STL, which carries no colour of any kind, so one has to be
chosen: the grey-green of a painted concrete hall, rather than the flat default grey
every uncoloured mesh gets.
"""

WRENCH_COLOUR = Color(0.55, 0.57, 0.60)
"""
What the wrench is painted, for the same reason: steel.
"""

# %% where everything stands in the hall

PELVIS_HEIGHT_ABOVE_FLOOR = 0.7923
"""
How far the G1's pelvis stands above the floor with all of its leg joints at zero.

The pelvis is the robot's root, so its ``odom`` has to be lifted by this much for the
robot's feet to rest on the floor rather than sink through it.
"""

SHELF_SURFACE_HEIGHT = 0.73
"""
Height of the shelf the crate starts on, measured off the hall's own geometry: the level
with enough room above it to lift a crate straight out.
"""

CRATE_BOTTOM_BELOW_ORIGIN = 0.110
"""
How far the crate's lowest point sits below its own origin, as ``prepare_assets.py``
reports it, so placing it on a surface rests it there rather than sinking it in.
"""

CRATE_FLOOR_THICKNESS = 0.02
"""
How thick the crate's own bottom is, which is how far above it its contents lie.
"""

WRENCH_BOTTOM_BELOW_ORIGIN = 0.096
"""
The same for the wrench, whose origin sits above the middle of its mesh too.
"""

CRATE_POSE = Pose.from_xyz_rpy(
    -7.07, -13.37, SHELF_SURFACE_HEIGHT + CRATE_BOTTOM_BELOW_ORIGIN, yaw=np.pi
)
"""
Where the crate starts: on the shelf of the rack along the hall's west wall.

Turned to face the aisle, since that is the side the robot can stand on and the side a
FRONT grasp approaches from -- between the rack and the wall there is no room.
"""

BENCH_SCALE = Scale(0.9, 0.6, 0.7)
"""
The workbench the crate is set down on.

Waist-high on purpose: measured on this robot, it places at 0.79 m and no longer at
0.55 m, so the floor is out of its reach and a bench is what it can put a crate on.
"""

BENCH_POSE = Pose.from_xyz_rpy(-5.6, -13.37, BENCH_SCALE.z / 2, yaw=np.pi)
"""
Where the workbench stands: in the aisle in front of the rack, where the hall's own
floor is clear.
"""

CRATE_BENCH_POSE = Pose.from_xyz_rpy(
    -5.6, -13.37, BENCH_SCALE.z + CRATE_BOTTOM_BELOW_ORIGIN, yaw=np.pi
)
"""
Where the crate is set down: on the workbench, so its contents can be reached standing.
"""

WRENCH_IN_CRATE_POSE = Pose.from_xyz_rpy(
    0.0,
    0.0,
    CRATE_FLOOR_THICKNESS + WRENCH_BOTTOM_BELOW_ORIGIN - CRATE_BOTTOM_BELOW_ORIGIN,
)
"""
Where the wrench lies, in the crate's own frame: on the crate's inside floor.

Held by the crate rather than by the world, so it travels with the crate when that is
carried off the rack -- a wrench in a crate goes where the crate goes.
"""

WRENCH_DESTINATION = Pose.from_xyz_rpy(
    -7.07, -12.17, SHELF_SURFACE_HEIGHT + WRENCH_BOTTOM_BELOW_ORIGIN, yaw=np.pi
)
"""
Where the wrench is carried: a shelf field of the same rack, a bay further along.
"""

STANDING_DISTANCE = 0.6
"""
How far the robot stands from a pose it grasps at, in metres, opposite its FRONT side.

Within the G1's reach, and far enough from a rack to leave its footprint free.
"""

ROBOT_START_POSE = Pose.from_xyz_rpy(-5.5, -13.37, PELVIS_HEIGHT_ABOVE_FLOOR, yaw=np.pi)
"""
Where the robot starts: in the aisle, facing the rack the crate is on.
"""

# %% building the world and the plan


def build_world() -> World:
    """
    The hall with the robot, the crate on its shelf and the wrench in the crate.
    """
    return WorldSpecification(
        robots=[
            RobotSpecification(
                semantic_annotation_type=UnitreeG1,
                world_T_odom=ROBOT_START_POSE.to_homogeneous_matrix(),
            )
        ],
        objects=[
            BodySpecification.mesh(
                "warehouse_hall",
                str(RESOURCES / "warehouse_hall.stl"),
                color=HALL_COLOUR,
            ),
            BodySpecification.box(
                "workbench",
                BENCH_SCALE,
                color=Color(0.35, 0.30, 0.26),
                parent_T_self=BENCH_POSE.to_homogeneous_matrix(),
            ),
            BodySpecification.mesh(
                "crate",
                str(RESOURCES / "open_crate.glb"),
                parent_T_self=CRATE_POSE.to_homogeneous_matrix(),
                # free to move, which is what makes it a thing the run carries around
                # rather than part of the hall
                connection_specification=Connection6DoFSpecification(),
                child_specifications=[
                    BodySpecification.mesh(
                        "wrench",
                        str(RESOURCES / "wrench.stl"),
                        color=WRENCH_COLOUR,
                        parent_T_self=WRENCH_IN_CRATE_POSE.to_homogeneous_matrix(),
                        connection_specification=Connection6DoFSpecification(),
                    )
                ],
            ),
        ],
    ).to_domain_object()


def standing_pose_in_front_of(pose: Pose, world: World) -> Pose:
    """
    The pose the robot stands in to reach one pose with a FRONT grasp.

    :param pose: The pose to approach from the robot's FRONT-facing side.
    :param world: The world the pose is expressed in.
    """
    yaw = float(pose.yaw)
    return Pose.from_xyz_rpy(
        pose.x - STANDING_DISTANCE * np.cos(yaw),
        pose.y - STANDING_DISTANCE * np.sin(yaw),
        PELVIS_HEIGHT_ABOVE_FLOOR,
        yaw=yaw,
        reference_frame=world.root,
    )


def build_plan(world: World, robot: UnitreeG1) -> Plan:
    """
    The plan: the crate off the shelf and onto the floor, then the wrench out of it and
    onto the next shelf along.

    :param world: The world the plan acts in.
    :param robot: The robot carrying it out.
    """
    crate = world.get_body_by_name("crate")
    wrench = world.get_body_by_name("wrench")
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(Arms.LEFT, robot),
    )
    context = Context(world=world, robot=robot, evaluate_conditions=False)

    def placed(pose: Pose) -> Pose:
        return Pose(
            pose.to_position(), pose.to_quaternion(), reference_frame=world.root
        )

    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            # the crate out of the rack and onto the bench
            NavigateAction(standing_pose_in_front_of(CRATE_POSE, world)),
            PickUpAction(crate, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            # driven to the bench first: a robot places what is in front of it, and the
            # bench is behind it after the crate has been taken off the rack
            NavigateAction(standing_pose_in_front_of(CRATE_BENCH_POSE, world)),
            PlaceAction(crate, placed(CRATE_BENCH_POSE), Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
            # the wrench out of the crate, where the robot is already standing
            PickUpAction(wrench, Arms.LEFT, grasp),
            ParkArmsAction(Arms.BOTH),
            # and onto a shelf a bay further along
            NavigateAction(standing_pose_in_front_of(WRENCH_DESTINATION, world)),
            PlaceAction(wrench, placed(WRENCH_DESTINATION), Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    ).plan


world = build_world()
robot = world.get_semantic_annotations_by_type(UnitreeG1)[0]

# the viewer follows the run live: the world through the visualization, and the plan
# through the plan attached to it -- its tree, each step's progress and the motion
# statechart being executed appear in the viewer's panels while the run happens
visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()
plan = build_plan(world, robot)
visualization.attach_plan(plan)

with simulated_robot:
    plan.perform()
