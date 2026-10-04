"""
An environment is read by the parser its file asks for, and the robots put into it stand
on its floor.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cramera.environment_file import (
    EnvironmentFile,
    GazeboEnvironmentFile,
    UnsupportedEnvironmentFileError,
    URDFEnvironmentFile,
    USDSceneEnvironmentFile,
)
from cramera.scene_presentation import ScenePresentation
from cramera.multi_robot import (
    RobotInstance,
    RobotPlacementNotFixedError,
    RobotScene,
    move_robot_to,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF

from .dataset.standing_robot import StandingRobot
from .dataset.walking_robot import WalkingRobot

DATASET = Path(__file__).parent / "dataset"
FLOOR_SLAB = DATASET / "floor_slab.urdf"


def floor_slab_top() -> float:
    """
    :return: How high the top of the slab ``floor_slab.urdf`` describes is.
    """
    slab = URDFParser.from_file(str(FLOOR_SLAB)).parse()
    return max(
        body.collision.as_bounding_box_collection_in_frame(slab.root)
        .bounding_box()
        .max_z
        for body in slab.bodies
        if body.collision
    )


# %% which parser a file is read by


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("kitchen.urdf", URDFEnvironmentFile),
        ("small_warehouse.world", GazeboEnvironmentFile),
        ("model.sdf", GazeboEnvironmentFile),
        ("world.usda", USDSceneEnvironmentFile),
        ("world.usd", USDSceneEnvironmentFile),
        ("world.usdc", USDSceneEnvironmentFile),
        ("scan.usdz", USDSceneEnvironmentFile),
    ],
)
def test_an_environment_file_is_read_as_its_suffix_says(
    file_name: str, expected: type[EnvironmentFile]
) -> None:
    assert type(EnvironmentFile.from_path(file_name)) is expected


def test_a_package_url_is_read_as_its_suffix_says() -> None:
    environment = EnvironmentFile.from_path(
        "package://aws_robomaker_small_warehouse_world/worlds/small.world"
    )

    assert type(environment) is GazeboEnvironmentFile


def test_a_file_no_parser_reads_is_refused() -> None:
    with pytest.raises(UnsupportedEnvironmentFileError):
        EnvironmentFile.from_path("notes.txt")


# %% how the viewer draws it


def test_a_scan_keeps_the_surfaces_it_was_photographed_with() -> None:
    presentation = USDSceneEnvironmentFile(path="world.usda").presentation()

    assert presentation.preserve_environment_materials is True


@pytest.mark.parametrize(
    "environment",
    [URDFEnvironmentFile(path="apartment.urdf"), GazeboEnvironmentFile(path="w.world")],
)
def test_a_modelled_environment_takes_the_viewers_palette(
    environment: EnvironmentFile,
) -> None:
    assert environment.presentation() == ScenePresentation()


# %% standing on the floor


def lowest_point_of(world: World, identifier: str) -> float:
    [robot] = [
        robot
        for robot in world.get_semantic_annotations_by_type(StandingRobot)
        if robot.root.name.prefix == identifier
    ]
    return world.height_of_lowest_collision_point_of_branch(robot.root)


def standing_scene(x: float) -> RobotScene:
    return RobotScene(
        instances=[
            RobotInstance(
                identifier="standing",
                label="Standing robot",
                robot_type=StandingRobot,
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=x),
            )
        ],
        active_identifier="standing",
    )


def walking_scene(x: float) -> RobotScene:
    return RobotScene(
        instances=[
            RobotInstance(
                identifier="walking",
                label="Walking robot",
                robot_type=WalkingRobot,
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=x),
            )
        ],
        active_identifier="walking",
    )


def test_a_robot_above_a_floor_stands_on_its_top() -> None:
    world = standing_scene(x=0.0).build_world(str(FLOOR_SLAB))

    assert lowest_point_of(world, "standing") == pytest.approx(
        floor_slab_top(), abs=1e-6
    )


def test_a_robot_beside_every_floor_stands_on_the_ground() -> None:
    world = standing_scene(x=10.0).build_world(str(FLOOR_SLAB))

    assert lowest_point_of(world, "standing") == pytest.approx(0.0, abs=1e-6)


def test_a_robot_in_no_environment_stands_on_the_ground() -> None:
    world = standing_scene(x=0.0).build_world()

    assert lowest_point_of(world, "standing") == pytest.approx(0.0, abs=1e-6)


# %% moving a robot elsewhere


def test_a_moved_robot_stands_where_it_was_moved_to() -> None:
    world = standing_scene(x=10.0).build_world(str(FLOOR_SLAB))
    [robot] = world.get_semantic_annotations_by_type(StandingRobot)

    move_robot_to(world, robot, x=0.5, y=-0.5, yaw=1.0)

    world_T_root = robot.root.global_pose.to_np()
    assert world_T_root[:2, 3] == pytest.approx([0.5, -0.5])
    assert math.atan2(world_T_root[1, 0], world_T_root[0, 0]) == pytest.approx(1.0)


def test_a_robot_moved_onto_a_floor_stands_on_it() -> None:
    world = standing_scene(x=10.0).build_world(str(FLOOR_SLAB))
    [robot] = world.get_semantic_annotations_by_type(StandingRobot)

    move_robot_to(world, robot, x=0.0, y=0.0, yaw=0.0)

    assert lowest_point_of(world, "standing") == pytest.approx(
        floor_slab_top(), abs=1e-6
    )


def test_a_robot_moved_off_a_floor_stands_on_the_ground() -> None:
    world = standing_scene(x=0.0).build_world(str(FLOOR_SLAB))
    [robot] = world.get_semantic_annotations_by_type(StandingRobot)

    move_robot_to(world, robot, x=10.0, y=0.0, yaw=0.0)

    assert lowest_point_of(world, "standing") == pytest.approx(0.0, abs=1e-6)


def test_a_robot_is_placed_where_it_stands_however_far_its_drive_carried_it() -> None:
    """
    A robot following a real one has the real robot's odometry written into its drive,
    so its root no longer stands at its localization frame's origin.

    Placing it stands the root where it is told, not the origin.
    """
    world = walking_scene(x=0.0).build_world()
    [robot] = world.get_semantic_annotations_by_type(WalkingRobot)
    robot.drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.5, yaw=math.pi / 2
    )

    move_robot_to(world, robot, x=3.0, y=4.0, yaw=0.0)

    world_T_root = robot.root.global_pose.to_np()
    assert world_T_root[:2, 3] == pytest.approx([3.0, 4.0])
    assert math.atan2(world_T_root[1, 0], world_T_root[0, 0]) == pytest.approx(0.0)


def test_a_robot_following_its_localization_is_not_moved() -> None:
    """
    A robot whose localization frame moves freely follows a real robot's localization;
    fixing it at a new place would stop that, so the move is refused and the robot keeps
    following.
    """
    world = standing_scene(x=0.0).build_world()
    [robot] = world.get_semantic_annotations_by_type(StandingRobot)
    odom = robot.root.parent_connection.parent
    placement = odom.parent_connection
    with world.modify_world():
        world.remove_connection(placement)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world, parent=placement.parent, child=odom
            )
        )

    with pytest.raises(RobotPlacementNotFixedError):
        move_robot_to(world, robot, x=1.0, y=0.0, yaw=0.0)

    assert isinstance(odom.parent_connection, Connection6DoF)


# %% the environment's own joints


def test_a_scene_stands_the_environments_joints_where_it_is_told() -> None:
    door = str(DATASET / "hinged_door.urdf")
    [hinge] = [
        connection
        for connection in standing_scene(x=0.0).build_world(door).connections
        if connection.child.name.name == "leaf"
    ]
    scene = standing_scene(x=0.0)
    scene.environment_joint_positions = {str(hinge.name): 1.2}

    world = scene.build_world(door)

    [opened] = [c for c in world.connections if str(c.name) == str(hinge.name)]
    assert opened.position == pytest.approx(1.2)


# %% how big the textures of a scan are drawn

SCANNED_SURFACE = DATASET / "scanned_surface.usda"


def test_a_scan_is_read_with_its_textures_capped_as_its_file_says() -> None:
    environment = USDSceneEnvironmentFile(
        path=str(SCANNED_SURFACE), maximum_texture_size=2048
    )

    assert environment.specification().world_parser.maximum_texture_size == 2048


def test_a_scan_keeps_its_textures_as_authored_unless_its_file_says() -> None:
    environment = USDSceneEnvironmentFile(path=str(SCANNED_SURFACE))

    assert environment.specification().world_parser.maximum_texture_size is None


# %% whether the environment is drawn into the shadow map


def test_a_scan_casts_no_shadows() -> None:
    presentation = USDSceneEnvironmentFile(path="world.usda").presentation()

    assert presentation.environment_casts_shadows is False


@pytest.mark.parametrize(
    "environment",
    [URDFEnvironmentFile(path="apartment.urdf"), GazeboEnvironmentFile(path="w.world")],
)
def test_a_modelled_environment_casts_shadows(environment: EnvironmentFile) -> None:
    assert environment.presentation().environment_casts_shadows is True
