"""Resolve placement tool goals from the attachment present at execution time."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from typing_extensions import ClassVar

from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


# %% attachment-dependent target sequence
class PlacementStage(IntEnum):
    """Ordered tool poses for supported placement."""

    APPROACH = 0
    """Approach above the object goal."""
    RELEASE = 1
    """Reach the supported object pose."""
    RETRACT = 2
    """Withdraw after releasing the object."""


@dataclass
class PlacementPoseSequence:
    """Retain one measured attachment through approach, release and retraction."""

    grasp: GraspDescription
    """Native grasp defining approach and retract clearances."""
    body: Body
    """Object whose measured attachment determines the tool goals."""
    target: Pose
    """Requested final object pose."""
    _poses: list[Pose] = field(default_factory=list, init=False)
    """Tool poses captured at the start of the current placement attempt."""

    def resolve(self, stage: PlacementStage) -> Pose:
        """Refresh the attachment on approach and retain it through detachment.

        :param stage: Placement stage whose tool pose is requested.
        """
        if stage is PlacementStage.APPROACH or not self._poses:
            self._poses = self.grasp.place_pose_sequence(self.target, self.body)
        return self._poses[stage]


# %% native deferred tool motion
@dataclass(eq=False, repr=False)
class PlacementGoal(Sequence):
    """Expand the ordinary tool controller after preceding actions have finished."""

    motion: MovePlacementMotion = field(kw_only=True)
    """Placement motion retaining the native tool tuning and mapping context."""

    def expand(self, context: MotionStatechartContext) -> None:
        """Resolve the measured attachment and construct the configured tool goal.

        :param context: Active native controller context.
        """
        self.nodes = [self.motion.resolved_motion().motion_chart]
        super().expand(context)


@dataclass
class MovePlacementMotion(MoveToolCenterPointMotion):
    """Move the tool to a placement stage using the current measured attachment."""

    requires_individual_execution: ClassVar[bool] = True
    """Resolve attachment-dependent goals after preceding motions finish."""
    placement: PlacementPoseSequence = field(kw_only=True)
    """Shared pose sequence retained through detachment."""
    stage: PlacementStage = field(kw_only=True)
    """Tool stage executed by this motion."""

    @property
    def motion_chart(self) -> MotionStatechartNode:
        """Resolve remote goals locally and simulated goals during expansion."""
        if GiskardExecutable.execution_type is ExecutionType.REAL:
            return self.resolved_motion().motion_chart
        return PlacementGoal(motion=self)

    def resolved_motion(self) -> MoveToolCenterPointMotion:
        """Construct the existing tool motion with the resolved pose and tuning."""
        self.target = self.placement.resolve(self.stage)
        motion = MoveToolCenterPointMotion(
            target=self.target,
            arm=self.arm,
            allow_gripper_collision=self.allow_gripper_collision,
            movement_type=self.movement_type,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            position_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
        )
        motion.plan_node = self.plan_node
        return motion
