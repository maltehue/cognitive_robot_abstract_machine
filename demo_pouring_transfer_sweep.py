"""
Sweep the cup-to-cup transfer demo over start poses of the held cup.

Runs the transfer of :mod:`demo_pouring_transfer` once per start of the held cup
against a running Giskard. The starts lie on a half circle around the destination cup
on the robot's side, each turned so the cup pours towards the destination, with a
range of initial tilts. One result line per case lets the aim and fill behaviour be
compared across starts.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path

import numpy as np
import rclpy
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.middleware.ros2.event_loop_manager import get_event_loop
from giskardpy.middleware.ros2.exceptions import (
    ExecutionCanceledException,
    NoActiveGoalToCancelError,
)
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from giskardpy.qp.control_cycle_recording import ControlCycleRecording
from rclpy.executors import SingleThreadedExecutor
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% constants shared with the demo

_JEROEN_CUP_STL = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)
_JEROEN_CUP_SCALE = Scale(1, 1, 1)
_TABLE_SURFACE_Z = 0.875 + 0.1

START_FILL = 1.0
GOAL_FILL = 0.7
FILL_TOLERANCE = 0.05

RECEIVING_CUP_X = 1.0
RECEIVING_CUP_Y = 0.099

_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]
_LEFT_ARM_PARK_POSITIONS = [2.62, -1.035, 1.13, -0.966, -0.88, 2.07]
_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]
_RIGHT_ARM_PARK_POSITIONS = [3.72, -2.07, -1.17, 4.0, 0.82, 0.75]

FILL_SETTLE_TIMEOUT = 10.0
"""
Seconds to wait for the fill levels to be synchronized back after a transfer.
"""

FILL_SETTLE_RESOLUTION = 1e-4
"""
Fill change between two polls below which the synchronized fill counts as settled.
"""

WORLD_SYNC_PAUSE = 0.5
"""
Seconds to give a world change time to reach the Giskard process.
"""

PARK_TIMEOUT = 30.0
"""
Seconds a parking motion may take before it is cancelled.
"""

CARRY_POSE_TIMEOUT = 30.0
"""
Seconds a move to a carry pose may take before it is cancelled.
"""

AIMING_TIMEOUT = 30.0
"""
Seconds the aiming goal may take before it is cancelled.
"""

TRANSFER_TIMEOUT = 90.0
"""
Seconds the transfer goal may take before it is cancelled.
"""

GOAL_POLL_INTERVAL = 0.05
"""
Seconds between two checks whether a sent goal was accepted.
"""

RESULTS_FILE = Path("pouring_transfer_sweep_results.jsonl")
"""
One JSON line per swept case, appended as the sweep progresses.
"""

RECORDINGS_DIRECTORY = Path(tempfile.gettempdir()) / "control_cycles"
"""
Where Giskard's control-cycle recorder writes one file per goal when it runs on this
machine.
"""

# %% sweep definition


@dataclass(frozen=True)
class HeldCupStart:
    """
    One start configuration of the held cup around the destination cup.

    The held cup stands on a circle around the destination cup and is turned so that its
    pouring plane contains the destination, then tilted a little towards it, so every
    start pours in the direction of the destination whatever side it starts on.
    """

    heading: float
    """
    Direction from the destination cup to the held cup in the table plane, in radians
    counter-clockwise from the robot's forward axis.
    """

    radius: float
    """
    Distance between the destination cup and the held cup in the table plane, in
    metres.
    """

    initial_tilt: float
    """
    Tilt of the held cup towards the destination cup before the pour starts, in
    radians.
    """

    height: float
    """
    Height of the gripper tool frame above the world root, in metres.
    """

    def _root_T_heading(self, world: World) -> HomogeneousTransformationMatrix:
        """
        Frame at the destination cup whose ``y`` axis points towards the held cup.

        The demo carries the cup on the destination's ``+y`` side, so this frame turns
        that arrangement to the start's heading.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=RECEIVING_CUP_X,
            y=RECEIVING_CUP_Y,
            z=self.height,
            yaw=self.heading - math.pi / 2.0,
            reference_frame=world.root,
        )

    def carry_pose(self, world: World) -> Pose:
        """
        The carry pose of the gripper tool frame for this start.

        The upright tool orientation and the tilt about the tool's pouring axis are the
        demo's; only the heading turns them around the destination cup.
        """
        heading_T_upright = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.0,
            pos_y=self.radius,
            pos_z=0.0,
            quat_z=0.5,
            quat_x=0.5,
            quat_y=0.5,
            quat_w=0.5,
        )
        upright_T_tilted = HomogeneousTransformationMatrix.from_xyz_rpy(
            yaw=self.initial_tilt
        )
        return (
            self._root_T_heading(world) @ heading_T_upright @ upright_T_tilted
        ).to_pose()

    def pouring_plane_normal(self, world: World) -> Vector3:
        """
        Horizontal normal of the plane the cup pours in, which passes through both cups.

        It is the tool frame's ``z`` axis in the carry pose, the axis the cup tilts
        about.
        """
        return Vector3(
            math.sin(self.heading),
            -math.cos(self.heading),
            0.0,
            reference_frame=world.root,
        )

    def label(self) -> str:
        """
        Short description for logs and result lines.
        """
        return (
            f"heading={math.degrees(self.heading):.0f}deg radius={self.radius:.2f} "
            f"tilt={self.initial_tilt:.2f}"
        )


HEADINGS_DEGREES = [90, 120, 150, 180, 210, 240, 270]
"""
Headings of the held cup around the destination cup in degrees: the half circle on the
robot's side, from the destination's left over the space between destination and robot
base to its right. The demo starts at 90 degrees.
"""

START_RADII = [0.2]
"""
Distances between the cups at the start, in metres; the demo's is 0.2.
"""

INITIAL_TILTS = [0.1, 0.4, 0.8]
"""
Tilts towards the destination cup before the pour, in radians; the demo uses 0.1.
"""

CARRY_HEIGHT = _TABLE_SURFACE_Z + 0.15
"""
Height of the gripper tool frame at the start, the demo's.
"""

SWEEP = [
    HeldCupStart(
        heading=math.radians(heading_degrees),
        radius=radius,
        initial_tilt=initial_tilt,
        height=CARRY_HEIGHT,
    )
    for heading_degrees, radius, initial_tilt in itertools.product(
        HEADINGS_DEGREES, START_RADII, INITIAL_TILTS
    )
]
"""
Every combination of heading, radius and initial tilt, in execution order.
"""


# %% bounded goal execution


@dataclass
class GoalTimedOutError(Exception):
    """
    A goal exceeded its time budget and was cancelled.
    """

    timeout: float
    """
    The budget in seconds the goal ran out of.
    """

    def __str__(self) -> str:
        return f"goal exceeded its budget of {self.timeout:.0f} s and was cancelled"


def _cancel_running_goal(giskard: GiskardWrapper) -> None:
    """
    Cancel the goal Giskard is executing and wait until it has stopped.

    A goal that finished in the meantime has nothing left to cancel, and the cancelled
    goal reports itself through the cancel exception; both mean the robot is standing
    still, which is all the caller needs.
    """
    try:
        giskard.cancel_goal_async()
    except NoActiveGoalToCancelError:
        return
    try:
        get_event_loop().run_until_complete(giskard.get_result())
    except ExecutionCanceledException:
        return


def execute_within(
    giskard: GiskardWrapper, statechart: MotionStatechart, timeout: float
) -> None:
    """
    Execute a motion statechart, cancelling it when it exceeds ``timeout`` seconds.

    :raises GoalTimedOutError: If the goal was not finished within the budget; it has
        been cancelled by then.
    """
    started = time.monotonic()
    accepted = giskard.execute_async(statechart)
    while not accepted.done():
        if time.monotonic() - started > timeout:
            raise GoalTimedOutError(timeout=timeout)
        time.sleep(GOAL_POLL_INTERVAL)
    remaining = timeout - (time.monotonic() - started)
    try:
        get_event_loop().run_until_complete(
            asyncio.wait_for(giskard.get_result(), remaining)
        )
    except TimeoutError:
        _cancel_running_goal(giskard)
        raise GoalTimedOutError(timeout=timeout)


# %% outcome of one case


class TransferStatus(StrEnum):
    """
    How one swept transfer ended.
    """

    FILLED = "filled"
    """
    The receiver ended within the fill tolerance of the goal.
    """

    MISSED_GOAL = "missed_goal"
    """
    The motion finished but the receiver ended outside the fill tolerance.
    """

    ABORTED = "aborted"
    """
    Giskard aborted one of the case's goals.
    """

    TIMED_OUT = "timed_out"
    """
    One of the case's goals ran out of its time budget and was cancelled.
    """


@dataclass
class AimMetrics:
    """
    Landing error of the pour read off the transfer goal's control-cycle recording.

    The projectile rows are equality rows whose bound is the current landing error, so
    the recording carries the error the aim still had in every cycle.
    """

    mean_error_while_pouring: float
    """
    Mean landing error in metres over the cycles in which the fill task was running.
    """

    max_error: float
    """
    Largest landing error in metres over the whole transfer goal.
    """

    @classmethod
    def from_recording(
        cls, recording: ControlCycleRecording, receiver: HasFillLevel
    ) -> AimMetrics:
        """
        Read the landing error series of the receiver's projectile rows.
        """
        row_names = recording.structure.row_names
        projectile_marker = f"{receiver.root.name}_projectile"
        projectile_rows = [
            index for index, name in enumerate(row_names) if projectile_marker in name
        ]
        fill_rows = [
            index
            for index, name in enumerate(row_names)
            if str(receiver.fill_connection.name) in name
        ]
        landing_error = np.linalg.norm(
            recording.row_lower_bounds[:, projectile_rows], axis=1
        )
        pouring = recording.row_weights[:, fill_rows].max(axis=1) > 0
        return cls(
            mean_error_while_pouring=(
                float(landing_error[pouring].mean()) if pouring.any() else float("nan")
            ),
            max_error=float(landing_error.max()),
        )


@dataclass
class TransferOutcome:
    """
    What one swept start produced.
    """

    start: HeldCupStart
    """
    The start configuration that was run.
    """

    status: TransferStatus
    """
    How the transfer ended.
    """

    receiver_fill: float
    """
    Fill level of the receiving cup after the transfer.
    """

    source_fill: float
    """
    Fill level of the held cup after the transfer.
    """

    duration: float
    """
    Wall-clock seconds from sending the aiming goal to the end of the transfer goal.
    """

    aim: AimMetrics | None
    """
    Landing-error metrics, when the transfer goal's recording was found on this machine.
    """

    failure: str | None
    """
    Name of the exception the case ended with, when it did not finish.
    """

    def to_json(self) -> dict:
        """
        Flat JSON representation for the results file.
        """
        return asdict(self) | {"status": str(self.status)}

    def summary(self) -> str:
        """
        One log line for this outcome.
        """
        aim = (
            "aim n/a"
            if self.aim is None
            else f"aim mean {self.aim.mean_error_while_pouring * 1000:.1f} mm "
            f"max {self.aim.max_error * 1000:.1f} mm"
        )
        return (
            f"{self.start.label()} -> {self.status}: receiver {self.receiver_fill:.3f} "
            f"source {self.source_fill:.3f} in {self.duration:.1f} s, {aim}"
            + (f", {self.failure}" if self.failure else "")
        )


# %% world setup


def _spawn_jeroen_cup_body(name: str) -> Body:
    """
    Create a Body with the Jeroen cup mesh geometry.
    """
    mesh = Mesh(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
        filename=_JEROEN_CUP_STL,
        scale=_JEROEN_CUP_SCALE,
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([mesh]), name=PrefixedName(name)
    )


def park_arms(giskard: GiskardWrapper, world: World) -> None:
    """
    Move both arms to their park configuration.
    """
    park_state = JointState.from_mapping(
        mapping={
            world.get_connection_by_name(name): value
            for name, value in zip(
                _LEFT_ARM_JOINT_NAMES + _RIGHT_ARM_JOINT_NAMES,
                _LEFT_ARM_PARK_POSITIONS + _RIGHT_ARM_PARK_POSITIONS,
            )
        }
    )
    statechart = MotionStatechart()
    park_task = JointPositionList(goal_state=park_state)
    statechart.add_node(park_task)
    statechart.add_node(EndMotion.when_true(park_task))
    execute_within(giskard, statechart, PARK_TIMEOUT)


def move_to_carry_pose(
    giskard: GiskardWrapper, world: World, left_tool_frame: Body, pose: Pose
) -> None:
    """
    Move the left gripper to the given upright carry pose.
    """
    statechart = MotionStatechart()
    cartesian_task = CartesianPose(
        root_link=world.root, tip_link=left_tool_frame, goal_pose=pose
    )
    statechart.add_node(cartesian_task)
    statechart.add_node(minimum_reached := LocalMinimumReached())
    statechart.add_node(EndMotion.when_true(minimum_reached))
    execute_within(giskard, statechart, CARRY_POSE_TIMEOUT)


@dataclass
class TransferCups:
    """
    The two cups of the transfer, resident in the fetched world.
    """

    source: HasFillLevel
    """
    The cup held in the left gripper.
    """

    receiver: HasFillLevel
    """
    The cup standing on the table.
    """

    def reset_fill_levels(self, world: World) -> None:
        """
        Put the start fill back into the source and empty the receiver, and give the
        change time to reach Giskard.
        """
        with world.modify_world():
            JointState.from_mapping(
                {
                    self.source.fill_connection: START_FILL,
                    self.receiver.fill_connection: 0.0,
                }
            ).apply_to(world)
        time.sleep(WORLD_SYNC_PAUSE)


def _reuse_cups(world: World) -> TransferCups:
    """
    Pick up the cups a previous run left in the world and re-resolve their fill
    connections to the world-resident ones.
    """
    source = world.get_semantic_annotation_by_name("source_cup")
    receiver = world.get_semantic_annotation_by_name("receiving_cup")
    for cup in (source, receiver):
        cup.fill_connection = world.get_connection(
            cup.fill_connection.parent, cup.fill_connection.child
        )
    return TransferCups(source=source, receiver=receiver)


def _create_cups(world: World, left_tool_frame: Body) -> TransferCups:
    """
    Attach the source cup to the gripper, place the receiver on the table and couple
    the receiver's inflow to the source's outflow.
    """
    source_body = _spawn_jeroen_cup_body("source_cup")
    with world.modify_world():
        world.add_body(source_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=left_tool_frame,
                child=source_body,
                name=PrefixedName("l_gripper_T_source_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0, y=-0.05
                ),
            )
        )
    source = HasFillLevel(name=PrefixedName("source_cup"), root=source_body)
    with world.modify_world():
        world.add_semantic_annotation(source)
    source.initialize_fill_level(
        world=world,
        initial_fill=START_FILL,
        outflow_rate_constant=0.8,
        discharge_coefficient=0.2,
    )

    receiver_body = _spawn_jeroen_cup_body("receiving_cup")
    with world.modify_world():
        world.add_body(receiver_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world,
                world.root,
                receiver_body,
                name=PrefixedName("table_T_receiving_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    RECEIVING_CUP_X, RECEIVING_CUP_Y, _TABLE_SURFACE_Z
                ),
            )
        )
    receiver = HasFillLevel(name=PrefixedName("receiving_cup"), root=receiver_body)
    with world.modify_world():
        world.add_semantic_annotation(receiver)
    time.sleep(WORLD_SYNC_PAUSE)
    receiver.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )
    receiver.receive_outflow_from(source=source, world=world)
    time.sleep(WORLD_SYNC_PAUSE)
    return TransferCups(source=source, receiver=receiver)


def setup_cups(world: World, left_tool_frame: Body) -> TransferCups:
    """
    Create the cups on the first run against a Giskard process, reuse them afterwards.
    """
    if world.get_semantic_annotations_by_name("source_cup"):
        print("Cups already present; reusing them.")
        return _reuse_cups(world)
    return _create_cups(world, left_tool_frame)


# %% transfer motions


def build_aiming_motion(
    world: World, cups: TransferCups, left_tool_frame: Body, start: HeldCupStart
) -> tuple[MotionStatechart, MotionStatechart]:
    """
    Build the two goals of a transfer: aiming the pour first, then filling while the
    aim, the rim clearance and the start's pouring plane are kept.
    """
    transfer_task = FillByTransferTask(
        receiver=cups.receiver,
        goal_value=GOAL_FILL,
        fill_level_tolerance=FILL_TOLERANCE,
        reference_velocity=0.03,
    )
    no_spill = KeepProjectileInReceiver(
        receiver=cups.receiver,
        source=cups.source,
        weight=DefaultWeights.WEIGHT_MAXIMUM,
        reference_velocity=0.1,
    )
    keep_above = KeepSourceRimAboveReceiverRim(
        receiver=cups.receiver,
        source=cups.source,
        minimum_clearance=0.07,
        clearance_band=0.05,
        weight=DefaultWeights.WEIGHT_MAXIMUM,
    )
    keep_plane = AlignPlanes(
        root_link=world.root,
        tip_link=left_tool_frame,
        goal_normal=start.pouring_plane_normal(world),
        tip_normal=Vector3.Z(reference_frame=left_tool_frame),
    )
    aiming = Parallel([no_spill, keep_above, keep_plane])
    aiming_statechart = MotionStatechart()
    aiming_statechart.add_node(aiming)
    aiming_statechart.add_node(EndMotion.when_true(aiming))

    transfer = Parallel([transfer_task, no_spill, keep_above, keep_plane])
    transfer_statechart = MotionStatechart()
    transfer_statechart.add_node(transfer)
    transfer_statechart.add_node(EndMotion.when_true(transfer))
    return aiming_statechart, transfer_statechart


def wait_for_synchronized_fill(receiver: HasFillLevel) -> None:
    """
    Wait until the receiver's fill reflects the finished transfer and stops changing.

    The goal result arrives before the fill levels are synchronized back from the
    Giskard process, so the values are read only once they have settled.
    """
    deadline = time.time() + FILL_SETTLE_TIMEOUT
    previous_fill = None
    while time.time() < deadline:
        current_fill = receiver.fill_level
        transfer_reflected = current_fill > FILL_TOLERANCE
        settled = (
            previous_fill is not None
            and abs(current_fill - previous_fill) < FILL_SETTLE_RESOLUTION
        )
        if transfer_reflected and settled:
            return
        previous_fill = current_fill
        time.sleep(0.1)


def new_recording(known_recordings: set[Path]) -> ControlCycleRecording | None:
    """
    The control-cycle recording written since ``known_recordings`` was listed, if
    Giskard records on this machine.
    """
    new_files = set(RECORDINGS_DIRECTORY.glob("*.npz")) - known_recordings
    if not new_files:
        return None
    return ControlCycleRecording.load(
        str(max(new_files, key=lambda f: f.stat().st_mtime))
    )


def known_recordings() -> set[Path]:
    """
    The recordings present right now, so a later listing can tell which goal wrote a
    new one.
    """
    if not RECORDINGS_DIRECTORY.exists():
        return set()
    return set(RECORDINGS_DIRECTORY.glob("*.npz"))


def run_case(
    start: HeldCupStart,
    giskard: GiskardWrapper,
    world: World,
    cups: TransferCups,
    left_tool_frame: Body,
) -> TransferOutcome:
    """
    Move the held cup to the start, reset the fills and run one aim-then-transfer.

    A goal Giskard aborts ends the case with :attr:`TransferStatus.ABORTED`, one that
    exceeds its time budget is cancelled and ends it with
    :attr:`TransferStatus.TIMED_OUT`; either way the remaining starts still run.
    """
    move_to_carry_pose(giskard, world, left_tool_frame, start.carry_pose(world))
    cups.reset_fill_levels(world)
    aiming_statechart, transfer_statechart = build_aiming_motion(
        world, cups, left_tool_frame, start
    )
    started_at = time.perf_counter()
    recordings_before_transfer: set[Path] = set()
    failure: str | None = None
    status: TransferStatus | None = None
    try:
        execute_within(giskard, aiming_statechart, AIMING_TIMEOUT)
        recordings_before_transfer = known_recordings()
        execute_within(giskard, transfer_statechart, TRANSFER_TIMEOUT)
    except GoalTimedOutError as error:
        failure = type(error).__name__
        status = TransferStatus.TIMED_OUT
    except GiskardException as error:
        failure = type(error).__name__
        status = TransferStatus.ABORTED
    duration = time.perf_counter() - started_at
    wait_for_synchronized_fill(cups.receiver)
    recording = None if failure else new_recording(recordings_before_transfer)
    aim = (
        None
        if recording is None
        else AimMetrics.from_recording(recording, cups.receiver)
    )
    receiver_fill = float(cups.receiver.fill_level)
    if status is None:
        status = (
            TransferStatus.FILLED
            if abs(receiver_fill - GOAL_FILL) <= FILL_TOLERANCE
            else TransferStatus.MISSED_GOAL
        )
    return TransferOutcome(
        start=start,
        status=status,
        receiver_fill=receiver_fill,
        source_fill=float(cups.source.fill_level),
        duration=duration,
        aim=aim,
        failure=failure,
    )


# %% sweep


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("tracy_transfer_sweep")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()

    giskard = GiskardWrapper(node)
    world = giskard.world
    left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")

    park_arms(giskard, world)
    move_to_carry_pose(giskard, world, left_tool_frame, SWEEP[0].carry_pose(world))
    cups = setup_cups(world, left_tool_frame)

    outcomes: list[TransferOutcome] = []
    for index, start in enumerate(SWEEP, start=1):
        print(f"[{index}/{len(SWEEP)}] {start.label()}")
        outcome = run_case(start, giskard, world, cups, left_tool_frame)
        outcomes.append(outcome)
        print(outcome.summary())
        with RESULTS_FILE.open("a") as results:
            results.write(json.dumps(outcome.to_json()) + "\n")

    move_to_carry_pose(giskard, world, left_tool_frame, SWEEP[0].carry_pose(world))

    print("\nSweep summary")
    for outcome in outcomes:
        print(outcome.summary())
    filled = sum(outcome.status is TransferStatus.FILLED for outcome in outcomes)
    print(f"{filled}/{len(outcomes)} starts filled the receiver to the goal")
    print(f"results written to {RESULTS_FILE.resolve()}")


if __name__ == "__main__":
    main()
