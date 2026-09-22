"""
A collision-free base can leave an obstacle's extra clearance buffer safely.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coraplex.datastructures.enums import Arms
from coraplex.demonstrations import RobotDemonstration
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.locations.navigation import (
    NavigationDeparture,
    NavigationPath,
    NavigationPathUnavailable,
)
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox, Scale
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

from .test_builder_transport import builder_transport_demo
from .test_navigation_connectors import close_wall_robot_world
from .test_navigation_rotation_height import elevated_robot_world
from .test_navigation_motion import NavigationTrajectory


# %% unchanged browser scene
def test_authored_apartment_start_can_depart_its_clearance_buffer(
    builder_transport_demo: RobotDemonstration,
) -> None:
    """
    Preserve the authored start while finding a non-penetrating departure.

    :param builder_transport_demo: The original browser-generated acceptance scene.
    """
    world = builder_transport_demo.build_simulated_world()
    builder_transport_demo.populate_scene(world)
    context = builder_transport_demo.build_context(world)
    with simulated_robot_advanced:
        sequential(
            [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
            context=context,
        ).perform()
    start = context.robot.root.global_pose.to_np().copy()
    goal = Pose.from_xyz_rpy(1.75, 2.22, yaw=-0.19118, reference_frame=world.root)
    route = NavigationPath(
        world,
        context.robot,
        goal,
        clearance=0.05,
        keep_joint_states=True,
        face_travel_direction=True,
    ).plan_route()
    assert route.stages[0].target.x > start[0, 3]
    assert route.stages[0].complete_orientation is True
    np.testing.assert_allclose(route.stages[0].target.to_np()[:3, :3], start[:3, :3])
    np.testing.assert_array_equal(context.robot.root.global_pose.to_np(), start)
    np.testing.assert_allclose(route.stages[-1].target.to_np(), goal.to_np())
    plan = execute_single(
        NavigateAction(goal, keep_joint_states=True), context=context
    ).plan
    trajectory = NavigationTrajectory(context.robot)
    plan.node_callbacks.append(trajectory)
    with simulated_robot_advanced:
        plan.perform()
    np.testing.assert_allclose(
        context.robot.root.global_pose.to_np(), goal.to_np(), atol=0.01
    )
    assert trajectory.avoidance_counts == {1}
    positions = np.asarray(trajectory.positions)
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02


# %% bounded fixed-heading escape
@pytest.mark.parametrize("heading_changes", [True, False])
def test_departure_increases_wall_clearance_before_turning(
    close_wall_robot_world: World,
    heading_changes: bool,
) -> None:
    """:param close_wall_robot_world: Long robot next to one known wall.
    :param heading_changes: Whether the route needs a turning connector afterward.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    start = Pose.from_xyz_rpy(0.15, reference_frame=world.root)
    robot.set_root_pose(start)
    goal = Pose.from_xyz_rpy(
        -2, yaw=np.pi / 2 if heading_changes else 0, reference_frame=world.root
    )
    path = NavigationPath(
        world,
        robot,
        goal,
        keep_joint_states=True,
        face_travel_direction=heading_changes,
    )
    route = path.plan_route()
    departure = route.stages[0]
    assert departure.target.x < start.x
    assert departure.complete_orientation is True
    np.testing.assert_array_equal(
        departure.target.to_np()[:3, :3], start.to_np()[:3, :3]
    )
    physical = replace(path, clearance=0).translation_obstacles(
        start, path.bounds_at_pose(start)
    )
    beginning = start.to_np()[:3, 3]
    direction = departure.target.to_np()[:3, 3] - beginning
    for obstacle in physical:
        beginning[2] = (obstacle.min_z + obstacle.max_z) / 2
        assert obstacle.to_array_bounds().clip_segment(beginning, direction) is None


@pytest.mark.parametrize("start_x", [0.25, 0.299])
def test_departure_rejects_physical_or_tolerance_intersection(
    close_wall_robot_world: World,
    start_x: float,
) -> None:
    """:param close_wall_robot_world: Existing narrow robot and wall geometry.
    :param start_x: A start overlapping the wall or lacking waypoint safety tolerance.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    robot.set_root_pose(Pose.from_xyz_rpy(start_x, reference_frame=world.root))
    goal = Pose.from_xyz_rpy(-2, yaw=np.pi / 2, reference_frame=world.root)
    with pytest.raises(NavigationPathUnavailable):
        NavigationPath(
            world, robot, goal, keep_joint_states=True, face_travel_direction=True
        ).plan()


def test_departure_cannot_move_deeper_into_another_clearance_buffer(
    close_wall_robot_world: World,
) -> None:
    """:param close_wall_robot_world: Existing narrow robot and obstacle geometry."""
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    robot.set_root_pose(Pose.from_xyz_rpy(0.15, reference_frame=world.root))
    BodySpecification.box(
        "opposing_wall",
        Scale(0.2, 20, 0.5),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-0.06),
    ).spawn(world)
    goal = Pose.from_xyz_rpy(-2, yaw=np.pi / 2, reference_frame=world.root)
    with pytest.raises(NavigationPathUnavailable):
        NavigationPath(
            world, robot, goal, keep_joint_states=True, face_travel_direction=True
        ).plan()


def test_departure_rejects_a_gap_smaller_than_waypoint_tolerance(
    close_wall_robot_world: World,
) -> None:
    """
    Reject a non-touching robot with only three millimeters of wall clearance.

    :param close_wall_robot_world: Robot half-width 0.1 m and wall beginning at x=0.3 m.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    robot.set_root_pose(Pose.from_xyz_rpy(0.197, reference_frame=world.root))
    goal = Pose.from_xyz_rpy(-2, reference_frame=world.root)
    with pytest.raises(NavigationPathUnavailable):
        NavigationPath(world, robot, goal, keep_joint_states=True).plan()


def test_departure_requires_a_held_joint_posture(close_wall_robot_world: World) -> None:
    """
    Do not apply a fixed-footprint escape while the body can change shape.

    :param close_wall_robot_world: Long robot within a wall's extra clearance buffer.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    robot.set_root_pose(Pose.from_xyz_rpy(0.15, reference_frame=world.root))
    goal = Pose.from_xyz_rpy(-2, reference_frame=world.root)
    with pytest.raises(NavigationPathUnavailable):
        NavigationPath(world, robot, goal, keep_joint_states=False).plan()


# %% complete segment clearance
@pytest.fixture()
def departure_collision_space() -> NavigationDeparture:
    """
    Create a known forbidden unit square with 0.1 m extra clearance.

    :return: Native free-space graph and conservative obstacle bounds.
    """
    world = World.create_with_root_body()
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    search = BoundingBoxCollection(
        [BoundingBox(-2, -2, -1, 2, 2, 1, origin)], reference_frame=world.root
    )
    obstacles = BoundingBoxCollection(
        [BoundingBox(-0.105, -0.105, -1, 1.105, 1.105, 1, origin)],
        reference_frame=world.root,
    )
    graph = GraphOfBoundingBoxes.navigation_map_from_bounding_boxes(search, obstacles)
    return NavigationDeparture(graph, obstacles, clearance=0.1, tolerance=1e-6)


@pytest.mark.parametrize(
    "start,end,allowed",
    [
        ((-0.05, 0.5, 0), (-0.2, 0.5, 0), True),
        ((-0.05, 0.5, 0), (1.2, 0.5, 0), False),
        ((-0.05, -0.002, 0), (1.2, -0.002, 0), False),
        ((-0.2, -0.05, 0), (1.2, -0.05, 0), False),
        ((-0.05, -0.05, 0), (0.5, -0.2, 0), False),
        ((-0.2, -0.2, 0), (-0.3, -0.3, 0), True),
    ],
    ids=[
        "outward",
        "through-obstacle",
        "through-tolerance",
        "new-buffer",
        "approaching-axis",
        "clear",
    ],
)
def test_departure_checks_the_entire_segment(
    departure_collision_space: NavigationDeparture,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    allowed: bool,
) -> None:
    """
    Clear endpoints alone must not permit an unsafe connecting segment.

    :param departure_collision_space: Known unit-square obstacle and native graph.
    :param start: Start outside the physical obstacle and its waypoint tolerance.
    :param end: Proposed end outside the complete navigation margin.
    :param allowed: Whether the whole segment preserves the departure constraints.
    """
    buffer = next(iter(departure_collision_space.obstacles)).to_array_bounds()
    occupied = BoundingBox(
        -0.005,
        -0.005,
        -1,
        1.005,
        1.005,
        1,
        HomogeneousTransformationMatrix(
            reference_frame=departure_collision_space.obstacles.reference_frame
        ),
    ).to_array_bounds()
    assert (
        departure_collision_space.allows_segment(
            np.asarray(start), np.asarray(end), buffer, occupied
        )
        is allowed
    )
