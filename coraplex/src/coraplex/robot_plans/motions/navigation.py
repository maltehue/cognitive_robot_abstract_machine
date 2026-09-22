from dataclasses import dataclass, field
from math import hypot, atan2
from typing_extensions import ClassVar

from coraplex.datastructures.enums import ExecutionType
from coraplex.locations.navigation import NavigationPath
from coraplex.plans.executables import GiskardExecutable
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianOrientation,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood.symbolic_math.symbolic_math import trinary_logic_and
from semantic_digital_twin.datastructures.joint_state import JointState
from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    ActiveConnection1DOF,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% collision-aware waypoint execution


@dataclass(eq=False, repr=False)
class TravelFacingWaypoint(CartesianPose):
    """
    Turn toward travel while driving, advancing once the waypoint position is reached.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Complete translation while allowing heading to blend into the next stage.

        :param context: Context used to build the controller's observation.
        :return: Observation of the inherited position task.
        """
        position = next(
            node for node in self.nodes if isinstance(node, CartesianPosition)
        )
        return NodeArtifacts(observation=position.observation_variable)


@dataclass(eq=False, repr=False)
class CollisionAvoidingNavigation(Sequence):
    """
    Follow planned base waypoints while continuously avoiding external collisions.
    """

    path: NavigationPath = field(kw_only=True)
    """
    World geometry, robot footprint and requested destination for this navigation.
    """

    _joint_hold: JointPositionList | None = field(default=None, init=False)
    """
    Parallel position goal maintained throughout navigation.
    """

    _completion_node: MotionStatechartNode = field(init=False)
    """
    Final waypoint whose observation determines successful navigation.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Build driving stages and keep collision avoidance active across them.

        :param context: Context used to construct the controller statechart.
        """
        start = self.path.robot.root.global_pose
        self.nodes = []
        stages = self.path.plan_route().stages
        for index, stage in enumerate(stages):
            self.nodes.append(
                self.create_segment(
                    start,
                    stage.target,
                    index == len(stages) - 1,
                    complete_orientation=stage.complete_orientation,
                )
            )
            start = stage.target
        self._completion_node = self.nodes[-1]
        super().expand(context)
        if self.path.keep_joint_states:
            goal_state = JointState.from_mapping(
                {
                    connection: float(connection.position)
                    for connection in self.path.robot.controlled_connections
                    if isinstance(connection, ActiveConnection1DOF)
                }
            )
            if len(goal_state) > 0:
                self._joint_hold = JointPositionList(goal_state=goal_state)
                self.add_node(self._joint_hold)
        if not any(
            node.parent_node is None
            for node in self.motion_statechart.get_nodes_by_type(
                ExternalCollisionAvoidance
            )
        ):
            self.add_node(ExternalCollisionAvoidance(robot=self.path.robot))

    def create_segment(
        self,
        start: Pose,
        target: Pose,
        final: bool,
        *,
        complete_orientation: bool = False
    ) -> MotionStatechartNode:
        """
        Select the existing controller matching the mobile base's drive.

        :param start: Planned start pose of this route segment.
        :param target: World-frame pose at the end of the segment.
        :param final: Whether this segment must reach the requested final orientation.
        :param complete_orientation: Whether a route boundary requires the heading
            before advancing.
        :return: Continuous driving or orientation controller.
        """
        robot = self.path.robot
        if isinstance(robot.drive, DifferentialDrive):
            if not final:
                target = Pose.from_xyz_rpy(
                    target.x,
                    target.y,
                    target.z,
                    yaw=atan2(target.y - start.y, target.x - start.x),
                    reference_frame=self.path.world.root,
                )
            if (
                hypot(target.x - start.x, target.y - start.y)
                <= self.path.waypoint_tolerance
            ):
                return CartesianOrientation(
                    root_link=self.path.world.root,
                    tip_link=robot.root,
                    goal_orientation=target.to_rotation_matrix(),
                    threshold=self.path.waypoint_tolerance,
                )
            return DifferentialDriveBaseGoal(
                diff_drive_connection=robot.drive,
                goal_pose=target,
                start_pose=start,
                threshold=self.path.waypoint_tolerance,
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            )
        controller = (
            TravelFacingWaypoint
            if self.path.face_travel_direction
            and not final
            and not complete_orientation
            else CartesianPose
        )
        return controller(
            root_link=self.path.world.root,
            tip_link=robot.root,
            goal_pose=target,
            translation_threshold=self.path.waypoint_tolerance,
            orientation_threshold=self.path.waypoint_tolerance,
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Complete at the final waypoint while avoidance remains a parallel task.

        :param context: Context used to build the controller's observation.
        :return: Observation of the final driving stage.
        """
        observation = self._completion_node.observation_variable
        if self._joint_hold is not None:
            observation = trinary_logic_and(
                observation, self._joint_hold.observation_variable
            )
        return NodeArtifacts(observation=observation)


# %% base motion


@dataclass
class MoveMotion(BaseMotion):
    """
    Drive the robot base to a pose with obstacle-aware simulated navigation.

    Simulated execution updates the world on every control tick, so the resulting
    trajectory is available to visualization and recording.
    """

    requires_individual_execution: ClassVar[bool] = True
    """
    Plan routes from the actual world after preceding motions have finished.
    """

    target: Pose
    """
    Location to which the robot should be moved.
    """

    keep_joint_states: bool = False
    """
    Keep the joint states of the robot during/at the end of the motion.
    """

    obstacle_clearance: float = field(default=0.05, kw_only=True)
    """
    Extra route clearance in meters; the controller also enforces native robot rules.
    """

    face_travel_direction: bool = field(default=True, kw_only=True)
    """
    Turn omnidirectional bases toward travel when the planned clearance permits it.
    """

    @property
    def _motion_chart(self) -> CartesianPose | CollisionAvoidingNavigation:
        """
        Plan a simulated base route or delegate real execution to its controller.

        :return: Continuous motion goal for the active execution environment.
        """
        if GiskardExecutable.execution_type != ExecutionType.REAL:
            return CollisionAvoidingNavigation(
                path=NavigationPath(
                    self.world,
                    self.robot,
                    self.target,
                    clearance=self.obstacle_clearance,
                    keep_joint_states=self.keep_joint_states,
                    face_travel_direction=self.face_travel_direction,
                ),
            )
        return CartesianPose(
            root_link=self.world.root,
            tip_link=self.robot.root,
            goal_pose=self.target,
        )
