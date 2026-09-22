"""
Plan planar base routes through the world's existing free-space graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import atan2, hypot

import numpy as np
import rustworkx as rx
from random_events.product_algebra import Event

from coraplex.plans.failures import PlanFailure
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.geometry import BoundingBox, Bounds
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

# %% route failures


class NavigationFailureReason(StrEnum):
    """
    Geometric reasons why a planar base route cannot be constructed.
    """

    NO_DRIVE = "The robot has no mobile base drive."
    """The robot has no connection capable of planar driving."""
    NO_GEOMETRY = "The robot has no collision geometry for its footprint."
    """No collision shapes are available to determine the robot's occupied space."""
    NONPLANAR_GOAL = "The requested base goal changes height, roll or pitch."
    """The destination cannot be represented by planar base motion."""
    OCCUPIED_ENDPOINT = "The start or goal is occupied by the inflated obstacles."
    """An endpoint does not provide the required robot clearance."""
    DISCONNECTED = "No connected route exists inside the navigation search bounds."
    """Free-space components containing the endpoints are disconnected."""


@dataclass
class NavigationPathUnavailable(PlanFailure):
    """
    A base route could not be found without entering occupied space.
    """

    target: Pose
    """
    Requested navigation destination.
    """

    reason: NavigationFailureReason
    """
    Condition preventing route construction.
    """

    def error_message(self) -> str:
        """
        Describe why navigation cannot proceed.
        """
        return f"Navigation path unavailable: {self.reason} Goal: {self.target}"


# %% route stages


@dataclass
class NavigationStage:
    """A base waypoint with an explicit orientation-completion requirement."""

    target: Pose
    """World-frame destination of this stage."""

    complete_orientation: bool = True
    """Wait for the requested heading before advancing to the next stage."""


@dataclass
class NavigationRoute:
    """Continuous base stages retaining fixed-heading approach boundaries."""

    stages: list[NavigationStage]
    """Ordered stages excluding the current base pose."""

    @property
    def poses(self) -> list[Pose]:
        """Return the waypoint poses in execution order."""
        return [stage.target for stage in self.stages]

    @classmethod
    def from_poses(
        cls, poses: list[Pose], blend_heading: bool = False
    ) -> NavigationRoute:
        """Require final orientation while optionally blending intermediate headings.

        :param poses: Ordered world-frame destinations.
        :param blend_heading: Whether intermediate waypoints may advance before yaw completion.
        :return: Route with explicit completion requirements.
        """
        return cls(
            [
                NavigationStage(pose, not blend_heading or index == len(poses) - 1)
                for index, pose in enumerate(poses)
            ]
        )


@dataclass
class NavigationConnector:
    """A fixed-heading path from an endpoint into a connected turning region."""

    component: int
    """Index of the reachable component in the rotational free-space graph."""

    points: list[Point3]
    """Ordered positions from the endpoint to a place with room to turn."""


# %% departure from an obstacle's clearance buffer


@dataclass
class NavigationDeparture:
    """Find a straight departure that increases existing obstacle separation."""

    graph: GraphOfBoundingBoxes
    """Fixed-heading free space including the full requested clearance."""
    obstacles: BoundingBoxCollection
    """Forbidden base positions including clearance and waypoint tolerance."""
    clearance: float
    """Extra clearance removable only while departing from an existing buffer."""
    tolerance: float
    """Inset keeping the departure destination inside a free graph cell."""

    def find(self, start: Point3) -> Point3 | None:
        """Find a safe translation into normal clearance without moving the world.

        :param start: Actual base position projected to the graph's height plane.
        :return: Nearest valid free-space entry, or None if departure is unsafe.
        """
        coordinates = start.to_np()[:3].flatten()
        buffers = [box.to_array_bounds() for box in self.obstacles]
        offset = np.array([self.clearance, self.clearance, 0])
        occupied = [Bounds(box.lower + offset, box.upper - offset) for box in buffers]
        if any(
            box.clip_segment(coordinates, np.zeros(3)) is not None for box in occupied
        ):
            return None
        candidates = [
            np.clip(
                coordinates,
                bounds.lower + self.tolerance,
                bounds.upper - self.tolerance,
            )
            for box in self.graph.graph.nodes()
            for bounds in [box.to_array_bounds()]
        ]
        candidates.sort(key=lambda point: np.linalg.norm(point - coordinates))
        for candidate in candidates:
            if all(
                self.allows_segment(coordinates, candidate, buffer, physical)
                for buffer, physical in zip(buffers, occupied)
            ):
                return Point3.from_iterable(
                    candidate, reference_frame=start.reference_frame
                )
        return None

    def allows_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        buffer: Bounds[np.ndarray],
        occupied: Bounds[np.ndarray],
    ) -> bool:
        """Reject penetration, new buffers and decreasing obstacle separation.

        :param start: Actual base coordinates.
        :param end: Proposed entry into full-clearance free space.
        :param buffer: One obstacle expanded by the complete navigation margin.
        :param occupied: The same obstacle retaining only waypoint safety tolerance.
        :return: Whether this obstacle permits the complete departure segment.
        """
        direction = end - start
        if occupied.clip_segment(start, direction) is not None:
            return False
        if buffer.clip_segment(start, direction) is None:
            return True
        if np.any(start < buffer.lower) or np.any(start > buffer.upper):
            return False
        if np.any((start < occupied.lower) & (direction > 0)) or np.any(
            (start > occupied.upper) & (direction < 0)
        ):
            return False
        before = np.maximum(occupied.lower - start, 0) + np.maximum(
            start - occupied.upper, 0
        )
        after = np.maximum(occupied.lower - end, 0) + np.maximum(
            end - occupied.upper, 0
        )
        return bool(np.linalg.norm(after) > np.linalg.norm(before))


# %% whole-robot route planning


@dataclass
class NavigationTransit:
    """Join turning regions through an orientation-constrained free-space layer."""

    path: NavigationPath
    """Planner supplying held-posture geometry and the unchanged obstacle clearance."""

    search_space: BoundingBoxCollection
    """Finite search region shared with the rotational free-space graph."""

    heading_samples: int = 8
    """Number of evenly spaced world-axis headings considered in addition to endpoints."""

    def heading_poses(self, start: Pose, goal: Pose) -> list[Pose]:
        """Sample planar headings while retaining the actual base position.

        :param start: Actual base pose supplying position and initial heading.
        :param goal: Requested destination supplying its final heading.
        :return: Distinct endpoint and uniformly sampled world-frame orientations.
        """
        candidates = [start, goal]
        candidates.extend(
            Pose.from_xyz_rpy(yaw=yaw, reference_frame=self.path.world.root)
            for yaw in np.linspace(-np.pi, np.pi, self.heading_samples, endpoint=False)
        )
        poses = []
        for candidate in candidates:
            if any(
                np.allclose(
                    candidate.to_np()[:3, :3],
                    pose.to_np()[:3, :3],
                    atol=self.path.geometry_tolerance,
                    rtol=0,
                )
                for pose in poses
            ):
                continue
            poses.append(
                Pose(
                    Point3(start.x, start.y, start.z),
                    candidate.to_quaternion(),
                    reference_frame=self.path.world.root,
                )
            )
        return poses

    def plan(
        self,
        start: Pose,
        goal: Pose,
        departures: list[NavigationConnector],
        arrivals: list[NavigationConnector],
    ) -> NavigationRoute | None:
        """Find a fixed-heading passage with turns confined to known turning regions.

        :param start: Actual base pose before its fixed-heading departure.
        :param goal: Requested pose after its fixed-heading final approach.
        :param departures: Safe connectors from the start into turning regions.
        :param arrivals: Safe connectors from the goal into turning regions.
        :return: Shortest sampled-heading route, or None if no safe transit exists.
        """
        if (
            not departures
            or not arrivals
            or not np.allclose(
                start.to_np()[:3, 2],
                [0, 0, 1],
                atol=self.path.geometry_tolerance,
                rtol=0,
            )
        ):
            return None
        best_route = None
        best_distance = float("inf")
        for heading in self.heading_poses(start, goal):
            graph = GraphOfBoundingBoxes.navigation_map_from_bounding_boxes(
                self.search_space,
                self.path.translation_obstacles(
                    heading, self.path.bounds_at_pose(heading)
                ),
            )
            for departure in departures:
                for arrival in arrivals:
                    first, last = departure.points[-1], arrival.points[-1]
                    if (
                        graph.node_of_point(first) is None
                        or graph.node_of_point(last) is None
                    ):
                        continue
                    middle = graph.path_from_to(first, last)
                    if middle is None:
                        continue
                    approach = list(reversed(arrival.points))
                    points = departure.points + middle[1:] + approach[1:]
                    distance = sum(
                        float(first.euclidean_distance(second))
                        for first, second in zip(points, points[1:])
                    )
                    if distance >= best_distance:
                        continue
                    poses = self.path.poses_at_heading(departure.points[1:], start)
                    poses.extend(self.path.poses_at_heading(middle, heading))
                    poses.extend(self.path.poses_at_heading(approach, goal))
                    best_route = NavigationRoute.from_poses(poses)
                    best_distance = distance
        return best_route


@dataclass
class NavigationPath:
    """
    Construct base waypoints with a whole-robot collision footprint.
    """

    world: World
    """
    World supplying obstacle collision geometry.
    """

    robot: AbstractRobot
    """
    Mobile robot whose collision bodies and attachments determine clearance.
    """

    target: Pose
    """
    Destination expressed in any frame of the world.
    """

    clearance: float = 0.1
    """
    Additional obstacle clearance in meters beyond the robot footprint.
    """

    waypoint_tolerance: float = 0.005
    """
    Controller position tolerance included in obstacle inflation, in meters.
    """

    search_margin: float = 2.0
    """
    Extra search distance around the start and destination envelope, in meters.
    """

    geometry_tolerance: float = 1e-6
    """
    Numerical tolerance for planar height and supporting contact, in meters.
    """

    keep_joint_states: bool = False
    """
    Whether execution holds the current joint posture throughout the route.
    """

    face_travel_direction: bool = field(default=False, kw_only=True)
    """
    Face each travel segment when a route with room for base rotation is available.
    """

    def translation_obstacles(
        self, start: Pose, robot_bounds: BoundingBoxCollection
    ) -> BoundingBoxCollection:
        """Expand obstacles by the fixed-orientation robot at overlapping heights.

        :param start: Base pose at which the collision shapes were measured.
        :param robot_bounds: World-frame collision boxes including attached objects.
        :return: Forbidden base positions for translation without turning.
        """
        margin = self.clearance + self.waypoint_tolerance
        lower_height = min(part.min_z for part in robot_bounds)
        upper_height = max(part.max_z for part in robot_bounds)
        excluded = set(self.robot.bodies_with_collision)
        obstacles = []
        for body in self.world.bodies_with_collision:
            if body in excluded:
                continue
            for box in body.collision.as_bounding_box_collection_in_frame(
                self.world.root
            ):
                overlapping = [
                    part
                    for part in robot_bounds
                    if box.max_z > part.min_z + self.geometry_tolerance
                    and box.min_z < part.max_z - self.geometry_tolerance
                ]
                if not overlapping:
                    continue
                obstacles.append(
                    BoundingBox(
                        box.min_x
                        - max(part.max_x for part in overlapping)
                        + start.x
                        - margin,
                        box.min_y
                        - max(part.max_y for part in overlapping)
                        + start.y
                        - margin,
                        lower_height,
                        box.max_x
                        - min(part.min_x for part in overlapping)
                        + start.x
                        + margin,
                        box.max_y
                        - min(part.min_y for part in overlapping)
                        + start.y
                        + margin,
                        upper_height,
                        HomogeneousTransformationMatrix(
                            reference_frame=self.world.root
                        ),
                    )
                )
        return BoundingBoxCollection(obstacles, reference_frame=self.world.root)

    def rotation_obstacles(
        self, start: Pose, robot_bounds: BoundingBoxCollection
    ) -> BoundingBoxCollection:
        """Expand obstacles by overlapping parts' complete planar turning envelopes.

        :param start: Base pose at which the collision shapes were measured.
        :param robot_bounds: World-frame collision boxes including attached objects.
        :return: Forbidden base positions for translation with arbitrary heading.
        """
        envelopes = []
        for part in robot_bounds:
            radius = hypot(
                max(abs(part.min_x - start.x), abs(part.max_x - start.x)),
                max(abs(part.min_y - start.y), abs(part.max_y - start.y)),
            )
            envelopes.append(
                BoundingBox(
                    start.x - radius,
                    start.y - radius,
                    part.min_z,
                    start.x + radius,
                    start.y + radius,
                    part.max_z,
                    HomogeneousTransformationMatrix(reference_frame=self.world.root),
                )
            )
        return self.translation_obstacles(
            start, BoundingBoxCollection(envelopes, reference_frame=self.world.root)
        )

    def plan(self) -> list[Pose]:
        """
        Find planar waypoints ending at the requested position and orientation.

        :return: Waypoints in world coordinates, excluding the current base pose.
        :raises NavigationPathUnavailable: If the robot or route cannot navigate.
        """
        return self.plan_route().poses

    def plan_route(self) -> NavigationRoute:
        """Plan driving stages with safe orientation transitions at tight endpoints.

        :return: Route ending at the requested position and orientation.
        :raises NavigationPathUnavailable: If no supported collision-free route exists.
        """
        world_T_start = self.robot.root.global_pose
        world_T_target = self.world.transform(self.target, self.world.root)
        if self.robot.drive is None:
            raise NavigationPathUnavailable(
                self.target, NavigationFailureReason.NO_DRIVE
            )
        if abs(
            world_T_target.z - world_T_start.z
        ) > self.geometry_tolerance or not np.allclose(
            world_T_target.to_np()[:3, 2], world_T_start.to_np()[:3, 2]
        ):
            raise NavigationPathUnavailable(
                self.target, NavigationFailureReason.NONPLANAR_GOAL
            )
        world_bounds = self.robot.as_bounding_box_collection_in_frame(self.world.root)
        if not world_bounds:
            raise NavigationPathUnavailable(
                self.target, NavigationFailureReason.NO_GEOMETRY
            )
        lower_height = min(box.min_z for box in world_bounds) + self.geometry_tolerance
        upper_height = max(box.max_z for box in world_bounds)
        search = BoundingBoxCollection(
            [
                BoundingBox(
                    min(world_T_start.x, world_T_target.x) - self.search_margin,
                    min(world_T_start.y, world_T_target.y) - self.search_margin,
                    lower_height,
                    max(world_T_start.x, world_T_target.x) + self.search_margin,
                    max(world_T_start.y, world_T_target.y) + self.search_margin,
                    upper_height,
                    HomogeneousTransformationMatrix(reference_frame=self.world.root),
                )
            ],
            reference_frame=self.world.root,
        )
        fixed_orientation = (
            self.keep_joint_states
            and isinstance(self.robot.drive, OmniDrive)
            and np.allclose(
                world_T_start.to_np()[:3, :3],
                world_T_target.to_np()[:3, :3],
                atol=self.geometry_tolerance,
                rtol=0,
            )
        )
        height = (lower_height + upper_height) / 2
        start = Point3(
            world_T_start.x, world_T_start.y, height, reference_frame=self.world.root
        )
        goal = Point3(
            world_T_target.x, world_T_target.y, height, reference_frame=self.world.root
        )
        footprints = (
            [False, True]
            if fixed_orientation and self.face_travel_direction
            else [fixed_orientation]
        )
        rotation_graph = None
        for fixed_orientation in footprints:
            departure = None
            if fixed_orientation:
                obstacles = self.translation_obstacles(world_T_start, world_bounds)
                graph = GraphOfBoundingBoxes.navigation_map_from_bounding_boxes(
                    search, obstacles
                )
                if graph.node_of_point(start) is None:
                    departure = NavigationDeparture(
                        graph, obstacles, self.clearance, self.geometry_tolerance
                    ).find(start)
            else:
                graph = GraphOfBoundingBoxes.navigation_map_from_bounding_boxes(
                    search, self.rotation_obstacles(world_T_start, world_bounds)
                )
                rotation_graph = graph
            route_start = departure if departure is not None else start
            if (
                graph.node_of_point(route_start) is None
                or graph.node_of_point(goal) is None
            ):
                reason = NavigationFailureReason.OCCUPIED_ENDPOINT
                continue
            points = graph.path_from_to(route_start, goal)
            if points is not None:
                if departure is not None:
                    points = [start, *points]
                break
            reason = NavigationFailureReason.DISCONNECTED
        else:
            if (
                self.keep_joint_states
                and isinstance(self.robot.drive, OmniDrive)
                and rotation_graph is not None
            ):
                return self.connect_endpoints(
                    world_T_start, world_T_target, start, goal, rotation_graph
                )
            raise NavigationPathUnavailable(self.target, reason)
        poses = [
            Pose(
                Point3(point.x, point.y, world_T_start.z),
                world_T_start.to_quaternion(),
                reference_frame=self.world.root,
            )
            for point in points[1:]
        ]
        poses[-1] = world_T_target
        if (
            self.face_travel_direction
            and not fixed_orientation
            and isinstance(self.robot.drive, OmniDrive)
        ):
            return NavigationRoute.from_poses(
                self.face_waypoints(world_T_start, poses), blend_heading=True
            )
        return NavigationRoute.from_poses(poses)

    def bounds_at_pose(self, pose: Pose) -> BoundingBoxCollection:
        """Project the held robot posture at a hypothetical base pose without moving it.

        :param pose: World-frame base pose at which to measure occupied geometry.
        :return: World-frame collision boxes including attached objects.
        """
        origin = HomogeneousTransformationMatrix(reference_frame=self.world.root)
        projected = []
        for body in self.robot.bodies_with_collision:
            for shape in body.collision.shapes:
                box = shape.local_frame_bounding_box
                base_T_shape = self.world.transform(box.origin, self.robot.root)
                projected.append(
                    replace(
                        box, origin=pose.to_homogeneous_matrix() @ base_T_shape
                    ).transform_to_origin(origin)
                )
        return BoundingBoxCollection(projected, reference_frame=self.world.root)

    def endpoint_connections(
        self,
        pose: Pose,
        point: Point3,
        rotation_graph: GraphOfBoundingBoxes,
        *,
        allow_departure: bool = False,
    ) -> list[NavigationConnector]:
        """Connect a fixed-heading endpoint to each reachable turning component.

        :param pose: Base pose whose orientation remains fixed along the connector.
        :param point: Endpoint projected to the graph's height plane.
        :param rotation_graph: Free space allowing arbitrary base heading.
        :param allow_departure: Permit an improving escape from the actual start's buffer.
        :return: Fixed-heading paths into reachable components.
        :raises NavigationPathUnavailable: If the actual endpoint footprint is occupied.
        """
        obstacles = self.translation_obstacles(pose, self.bounds_at_pose(pose))
        graph = GraphOfBoundingBoxes.navigation_map_from_bounding_boxes(
            rotation_graph.search_space,
            obstacles,
        )
        node = graph.node_of_point(point)
        prefix = []
        if node is None and allow_departure:
            departure = NavigationDeparture(
                graph, obstacles, self.clearance, self.geometry_tolerance
            ).find(point)
            if departure is not None:
                prefix = [point]
                point = departure
                node = graph.node_of_point(point)
        if node is None:
            raise NavigationPathUnavailable(
                self.target, NavigationFailureReason.OCCUPIED_ENDPOINT
            )
        endpoint_index = graph.box_to_index_map[node]
        reachable = next(
            component
            for component in rx.connected_components(graph.graph)
            if endpoint_index in component
        )
        reachable_space = Event.from_simple_sets(
            *[graph.graph[index].simple_event for index in reachable]
        )
        connections = []
        for component_index, component in enumerate(
            rx.connected_components(rotation_graph.graph)
        ):
            turning_space = Event.from_simple_sets(
                *[rotation_graph.graph[index].simple_event for index in component]
            )
            overlap = BoundingBoxCollection.from_event(
                self.world.root, reachable_space & turning_space
            )
            if not overlap:
                continue
            candidates = []
            for box in overlap:
                bounds = box.to_array_bounds()
                coordinates = np.clip(
                    point.to_np()[:3].flatten(),
                    bounds.lower + self.geometry_tolerance,
                    bounds.upper - self.geometry_tolerance,
                )
                candidates.append(
                    Point3.from_iterable(coordinates, reference_frame=self.world.root)
                )
            portal = min(
                candidates,
                key=lambda candidate: float(point.euclidean_distance(candidate)),
            )
            points = graph.path_from_to(point, portal)
            if points is not None:
                connections.append(
                    NavigationConnector(component_index, prefix + points)
                )
        return connections

    def connect_endpoints(
        self,
        start_pose: Pose,
        goal_pose: Pose,
        start: Point3,
        goal: Point3,
        rotation_graph: GraphOfBoundingBoxes,
    ) -> NavigationRoute:
        """Join fixed-heading departure and approach paths through turning free space.

        :param start_pose: Actual world-frame base pose.
        :param goal_pose: Requested world-frame base pose.
        :param start: Start projected to the graph height plane.
        :param goal: Goal projected to the graph height plane.
        :param rotation_graph: Existing collision-free space for turning motion.
        :return: Route containing explicit orientation-completion boundaries.
        :raises NavigationPathUnavailable: If the endpoints cannot share a turning component.
        """
        departures = self.endpoint_connections(
            start_pose, start, rotation_graph, allow_departure=True
        )
        arrivals = self.endpoint_connections(goal_pose, goal, rotation_graph)
        best_route = None
        best_distance = float("inf")
        for departure in departures:
            for arrival in arrivals:
                if departure.component != arrival.component:
                    continue
                middle = rotation_graph.path_from_to(
                    departure.points[-1], arrival.points[-1]
                )
                if middle is None:
                    continue
                approach = list(reversed(arrival.points))
                points = departure.points + middle[1:] + approach[1:]
                distance = sum(
                    float(first.euclidean_distance(second))
                    for first, second in zip(points, points[1:])
                )
                if distance >= best_distance:
                    continue
                departure_poses = self.poses_at_heading(
                    departure.points[1:], start_pose
                )
                middle_poses = self.poses_at_heading(middle[1:], start_pose)
                middle_poses[-1] = self.poses_at_heading([middle[-1]], goal_pose)[0]
                middle_start = self.poses_at_heading([middle[0]], start_pose)[0]
                if self.face_travel_direction:
                    middle_poses = self.face_waypoints(middle_start, middle_poses)
                approach_poses = self.poses_at_heading(approach[1:], goal_pose)
                stages = NavigationRoute.from_poses(departure_poses).stages
                stages.extend(
                    NavigationRoute.from_poses(
                        middle_poses, self.face_travel_direction
                    ).stages
                )
                stages.extend(NavigationRoute.from_poses(approach_poses).stages)
                best_route = NavigationRoute(stages)
                best_distance = distance
        if best_route is None:
            best_route = NavigationTransit(self, rotation_graph.search_space).plan(
                start_pose, goal_pose, departures, arrivals
            )
        if best_route is None:
            raise NavigationPathUnavailable(
                self.target, NavigationFailureReason.DISCONNECTED
            )
        return best_route

    def poses_at_heading(self, points: list[Point3], pose: Pose) -> list[Pose]:
        """Restore the base height and a fixed orientation to graph positions.

        :param points: Ordered positions in the planar graph.
        :param pose: Base pose supplying the physical height and orientation.
        :return: World-frame base poses at the requested graph positions.
        """
        return [
            Pose(
                Point3(point.x, point.y, pose.z),
                pose.to_quaternion(),
                reference_frame=self.world.root,
            )
            for point in points
        ]

    def face_waypoints(self, start: Pose, poses: list[Pose]) -> list[Pose]:
        """Orient travel segments and retain the requested final orientation.

        Tilted bases retain the planned orientation to preserve their height plane.

        :param start: World-frame base pose before driving.
        :param poses: Planned world-frame waypoints ending at the requested pose.
        :return: Travel-facing poses followed by the final orientation if necessary.
        """
        if not np.allclose(
            start.to_np()[:3, 2], [0, 0, 1], atol=self.geometry_tolerance, rtol=0
        ):
            return poses
        waypoints = []
        previous = start
        for pose in poses:
            if pose.x == previous.x and pose.y == previous.y:
                continue
            waypoints.append(
                Pose.from_xyz_rpy(
                    pose.x,
                    pose.y,
                    pose.z,
                    yaw=atan2(pose.y - previous.y, pose.x - previous.x),
                    reference_frame=self.world.root,
                )
            )
            previous = pose
        if not waypoints or not np.allclose(
            waypoints[-1].to_np(),
            poses[-1].to_np(),
            atol=self.geometry_tolerance,
            rtol=0,
        ):
            waypoints.append(poses[-1])
        else:
            waypoints[-1] = poses[-1]
        return waypoints
