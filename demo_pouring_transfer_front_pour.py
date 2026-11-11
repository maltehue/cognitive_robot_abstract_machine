"""
Two-handed front-pour demo: the cups in the two grippers pour into each other in turn.

Both cups are held at their middle with the grippers facing each other. The pouring cup
tilts about the axis through its gripper's two fingers, so its content leaves over the
side of the rim that points away from the palm and no finger lies between the two cups.

Every pour empties the pouring cup into the other one. The two cups then trade places:
the emptied cup comes down to where the other one stood and the filled cup rises beside
it, upright, to pour back.
"""

from __future__ import annotations

import math
import time
from itertools import cycle, islice
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path

from typing_extensions import Iterator, Optional, Protocol

from experiments.pouring_model_verification.hypothesis_bank import ModelPoint
from experiments.real_pouring.poured_material import PouredMaterial

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountSimulationTimeSeconds,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    HeldAtTilt,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
    MinimizeLandingSpread,
    TiltPacing,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from tracy_pouring_setup import (
    LEFT_TOOL_FRAME,
    TABLE_SURFACE_Z,
    connect_to_giskard,
    park_arms,
)

# %% cups and fills

_JEROEN_CUP_STL = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)

RIGHT_TOOL_FRAME = "r_gripper_tool_frame"
"""
The frame the right gripper's cup is held in.
"""

LEFT_CUP_NAME = "front_pour_left_cup"
RIGHT_CUP_NAME = "front_pour_right_cup"

GRASP_HEIGHT = 0.08
"""
Height of a tool frame above the base of the cup it holds, in metres: the middle of the
cup.
"""

MATERIAL = PouredMaterial.CORN
"""
What the cups pour back and forth.
"""

LOADED_MASS_GRAMS = 217.0
"""
What the material loaded into the left cup weighs, in grams.
"""

START_FILL = LOADED_MASS_GRAMS / MATERIAL.full_cup_mass_grams
"""
Fill of the left cup before the first pour; the right cup starts empty.

Kept well below the brim, so that the cup poured into never runs over and nothing leaves
the pouring cup at :data:`PRE_TILT`.
"""

FILL_TOLERANCE = 0.02
"""
Fill that may stay behind in the pouring cup for it to count as emptied.
"""

NUMBER_OF_POURS: Optional[int] = None
"""
How many pours the demo runs, or ``None`` to pour back and forth until it is stopped.
"""

DRAIN = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)
"""
The drain the controller pours on: the uncalibrated one the corn sweeps of 2026-10-23
and 2026-10-24 started every pour from.
"""

CORN_DISCHARGE_CORRECTION = 0.174
"""
Factor on the drain's discharge coefficient for corn, which falls from the lip almost
straight down: the median the live bank had corrected the coefficient by at the end of
the ten corn pours of 2026-10-24 (0.159 to 0.284, one pour uncorrected;
experiments/results/real_pouring/sweep_2026-10-24/dashboard/pours.json).

Nothing corrects the throw while this demo runs, so it starts from that value.
"""

REPOSE_ANGLE = DRAIN.repose_angle
"""
Tilt past the geometric onset before the material starts to flow, in radians.
"""

OUTFLOW_RATE_CONSTANT = DRAIN.outflow_rate_constant
"""
How fast a pouring cup drains per unit of head above the lip.
"""

DISCHARGE_COEFFICIENT = DRAIN.discharge_coefficient * CORN_DISCHARGE_CORRECTION
"""
Scales the exit speed and with it how far the stream is thrown.
"""

# %% layout

CARRY_ORIENTATION = HomogeneousTransformationMatrix.from_xyz_quaternion(
    quat_x=0.5, quat_y=0.5, quat_z=0.5, quat_w=0.5
)
"""
Orientation of a tool frame that holds a cup upright while reaching along the world's
``x`` axis: its ``y`` axis points up, its ``z`` axis forward and its ``x`` axis, the
line through the two fingers, to the side.
"""

RECEIVER_RIM_X = 0.9
RECEIVER_RIM_Y = 0.0
RECEIVER_RIM_HEIGHT = 0.4
"""
Height of the receiving cup's rim above the table top, in metres.

Whichever cup is poured into stands at this place.
"""

LEFT_REACH_HEADING = -math.pi / 2
"""
Direction the left gripper reaches in, and so the direction its cup pours in, as an
angle about the world's ``z`` axis from its ``x`` axis, in radians: towards the right
arm.
"""

RIGHT_REACH_HEADING = math.pi / 2
"""
Direction the right gripper reaches in, in radians: towards the left arm.
"""

LOADING_GAP = 0.06
"""
Free distance between the walls of the two upright cups while they are loaded, in
metres.
"""

LOADING_RISE = 0.09
"""
Height of the upright source's rim above the receiver's rim while loading, in metres.
"""

PRE_TILT = math.radians(30.0)
"""
Tilt the source is turned to beside the receiver before the transfer starts, in radians.
"""

PRE_TILT_LIP_INSET = 0.034
"""
How far past the receiver's near rim edge the source's lip hangs at :data:`PRE_TILT`, in
metres.
"""

PRE_TILT_LIP_RISE = 0.08
"""
Height of the source's lip above the receiver's rim at :data:`PRE_TILT`, in metres.
"""

_ARM_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]
_ARM_LOADING_POSITIONS = [
    2.871,
    -0.959,
    1.419,
    0.958,
    2.074,
    1.261,
    3.395,
    -2.323,
    -1.497,
    2.392,
    -2.076,
    -1.28,
]
"""
Joint positions that put both tool frames where the first pour starts, in the order of
:data:`_ARM_JOINT_NAMES`: the left cup upright beside the right one.

Reached in joint space, so the arms arrive in the elbow and wrist configuration the
pours were checked in; the two arms mirror each other in it, so either can take either
role.
"""

# %% transfer

MINIMUM_CLEARANCE = 0.06
"""
The lowest the source's lip may come above the receiver's rim, in metres.
"""

CLEARANCE_BAND = 0.02
"""
How far above :data:`MINIMUM_CLEARANCE` the lip may rise, in metres.
"""

POURING_VELOCITY = 0.06
"""
Reference rate of the receiver's fill while pouring, per second.
"""

STOPPING_VELOCITY = 1.0
"""
Reference rate of the receiver's fill once the pour has to stop, per second.
"""

SETTLED_INFLOW = 0.005
"""
Rate of the receiver's fill below which the pour counts as stopped, per second.
"""

APPROACH_TILT_RATE = math.radians(12.0)
MAXIMUM_TILT_RATE = math.radians(12.0)
MAXIMUM_RETURN_TILT_RATE = math.radians(15.0)
MAXIMUM_TILT_ACCELERATION = math.radians(60.0)
MAXIMUM_TILT = math.radians(110.0)
"""
The tilt the source is never turned past, in radians.
"""

CEILING_SLOWDOWN_WIDTH = math.radians(5.0)
"""
The span of tilt below :data:`MAXIMUM_TILT` over which the source slows to a stop, in
radians.
"""

CEILING_HOLD_SECONDS = 5.0
"""
How long the source may stay at :data:`MAXIMUM_TILT` before the transfer ends.
"""

EXIT_SPEED_UNCERTAINTY = 0.5
"""
Standard deviation of the stream's exit speed, as a share of that speed; the wider it
is, the harder the lip is drawn down towards the receiver's rim to shorten the throw.
"""

TRANSFER_TIME_LIMIT = 90.0
"""
Seconds after which the transfer ends whatever the fill.
"""

FILL_SETTLE_TIMEOUT = 10.0
"""
Seconds to wait for the fills Giskard synchronizes back after a transfer.
"""


class MotionExecutor(Protocol):
    """
    Anything that owns a world and can run a motion statechart in it.
    """

    world: World

    def execute(self, motion_statechart: MotionStatechart) -> None: ...


@dataclass
class HeldCup:
    """
    A cup together with the gripper that holds it.
    """

    cup: HasFillLevel
    """
    The held cup.
    """

    tool_frame: Body
    """
    The tool frame of the gripper holding the cup.
    """

    reach_heading: float
    """
    Direction the gripper reaches in, towards the other cup, as an angle about the
    world's ``z`` axis from its ``x`` axis, in radians.
    """


@dataclass
class Pour:
    """
    One pour from a held cup into the other, and the tool poses around it.
    """

    world: World
    """
    The world the poses are expressed in.
    """

    source: HeldCup
    """
    The cup poured from.
    """

    receiver: HeldCup
    """
    The cup poured into.
    """

    @property
    def cup_height(self) -> float:
        """
        Height of the pouring cup, in metres.
        """
        return self.source.cup.root.collision.height

    @property
    def rim_radius(self) -> float:
        """
        Radius of the pouring cup's rim, in metres.
        """
        return self.source.cup.root.collision.width / 2

    def _tool_pose(
        self, forward: float, rise: float, held: HeldCup, tilt: float
    ) -> Pose:
        """
        The pose of a tool frame whose cup has the point of its rim away from the palm
        at the given place.

        :param forward: Position of that rim point along the pour direction, measured
            from the receiver's rim centre, in metres.
        :param rise: Height of that rim point above the receiver's rim, in metres.
        :param held: The cup the tool frame holds.
        :param tilt: Tilt of the cup away from the palm, in radians.
        """
        pour_heading = self.source.reach_heading
        world_T_rim_point = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=RECEIVER_RIM_X + forward * math.cos(pour_heading),
            y=RECEIVER_RIM_Y + forward * math.sin(pour_heading),
            z=TABLE_SURFACE_Z + RECEIVER_RIM_HEIGHT + rise,
            yaw=held.reach_heading,
            reference_frame=self.world.root,
        )
        tool_T_rim_point = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=self.cup_height - GRASP_HEIGHT, z=self.rim_radius
        )
        return (
            world_T_rim_point
            @ CARRY_ORIENTATION
            @ HomogeneousTransformationMatrix.from_xyz_rpy(roll=tilt)
            @ tool_T_rim_point.inverse()
        ).to_pose()

    def receiver_tool_pose(self) -> Pose:
        """
        The pose of the receiver's tool frame that holds it upright, facing the source.
        """
        return self._tool_pose(
            forward=-self.rim_radius, rise=0.0, held=self.receiver, tilt=0.0
        )

    def loading_tool_pose(self) -> Pose:
        """
        The pose of the source's tool frame that holds it upright beside the receiver.
        """
        return self._tool_pose(
            forward=-(self.rim_radius + LOADING_GAP),
            rise=LOADING_RISE,
            held=self.source,
            tilt=0.0,
        )

    def pre_tilted_tool_pose(self) -> Pose:
        """
        The pose of the source's tool frame the pouring starts from: the source leaning
        towards the receiver with its lip over the receiver's opening.
        """
        return self._tool_pose(
            forward=-(self.rim_radius - PRE_TILT_LIP_INSET),
            rise=PRE_TILT_LIP_RISE,
            held=self.source,
            tilt=PRE_TILT,
        )

    def pose_task(self, name: str, held: HeldCup, goal_pose: Pose) -> CartesianPose:
        """
        :return: The task that brings the tool frame holding ``held`` to ``goal_pose``.
        """
        return CartesianPose(
            name=name,
            root_link=self.world.root,
            tip_link=held.tool_frame,
            goal_pose=goal_pose,
        )

    def reversed(self) -> Pour:
        """
        :return: The pour back, from this pour's receiver into its source.
        """
        return Pour(world=self.world, source=self.receiver, receiver=self.source)

    def pour_motion(self) -> MotionStatechart:
        """
        The motion that leans the source towards the receiver up to :data:`PRE_TILT` and
        then pours until the source is empty, the pour taking over from the lean without
        a stop in between.
        """
        world = self.world
        source, receiver = self.source.cup, self.receiver.cup
        pour_heading = self.source.reach_heading
        goal_fill = receiver.fill_level + source.fill_level
        transfer_task = FillByTransferTask(
            receiver=receiver,
            goal_value=goal_fill,
            fill_level_tolerance=FILL_TOLERANCE,
            reference_velocity=POURING_VELOCITY,
            stopping_velocity=STOPPING_VELOCITY,
            outflow_tolerance=SETTLED_INFLOW,
        )
        no_spill = KeepProjectileInReceiver(
            receiver=receiver,
            source=source,
            weight=DefaultWeights.WEIGHT_MAXIMUM,
            reference_velocity=0.1,
        )
        keep_above = KeepSourceRimAboveReceiverRim(
            receiver=receiver,
            source=source,
            minimum_clearance=MINIMUM_CLEARANCE,
            clearance_band=CLEARANCE_BAND,
            weight=DefaultWeights.WEIGHT_MAXIMUM,
        )
        # The line through the source gripper's fingers stays level and across the pour
        # direction, so the cup can only tip away from the palm.
        keep_pouring_plane = AlignPlanes(
            name="keep pouring plane",
            root_link=world.root,
            tip_link=self.source.tool_frame,
            goal_normal=Vector3(
                -math.sin(pour_heading),
                math.cos(pour_heading),
                0.0,
                reference_frame=world.root,
            ),
            tip_normal=Vector3.X(reference_frame=self.source.tool_frame),
        )
        keep_receiver_upright = AlignPlanes(
            name="keep receiver upright",
            root_link=world.root,
            tip_link=self.receiver.tool_frame,
            goal_normal=Vector3.Z(reference_frame=world.root),
            tip_normal=Vector3.Y(reference_frame=self.receiver.tool_frame),
        )
        tilt_pacing = TiltPacing(
            source=source,
            approach_rate=APPROACH_TILT_RATE,
            maximum_rate=MAXIMUM_TILT_RATE,
            maximum_tilt=MAXIMUM_TILT,
            maximum_return_rate=MAXIMUM_RETURN_TILT_RATE,
            maximum_acceleration=MAXIMUM_TILT_ACCELERATION,
            ceiling_slowdown_width=CEILING_SLOWDOWN_WIDTH,
            receiver=receiver,
            approach_until_fill=goal_fill - FILL_TOLERANCE,
        )
        # Brings the lip down to the clearance floor wherever nothing else needs it up.
        keep_lip_low = MinimizeLandingSpread(
            receiver=receiver,
            source=source,
            exit_speed_uncertainty=EXIT_SPEED_UNCERTAINTY,
        )
        pouring = Parallel(
            [
                transfer_task,
                no_spill,
                keep_above,
                keep_pouring_plane,
                tilt_pacing,
                keep_lip_low,
            ]
        )
        lean_then_pour = Sequence(
            [
                self.pose_task(
                    "lean towards the receiver",
                    self.source,
                    self.pre_tilted_tool_pose(),
                ),
                pouring,
            ]
        )
        # An emptied cup the model still believes to hold a rest ends the pour here.
        held_at_ceiling = HeldAtTilt(
            source=source, tilt=MAXIMUM_TILT, seconds=CEILING_HOLD_SECONDS
        )
        time_limit = CountSimulationTimeSeconds(seconds=TRANSFER_TIME_LIMIT)
        motion = MotionStatechart()
        motion.add_node(lean_then_pour)
        motion.add_node(keep_receiver_upright)
        motion.add_node(held_at_ceiling)
        motion.add_node(time_limit)
        for end_condition in (lean_then_pour, held_at_ceiling, time_limit):
            motion.add_node(EndMotion.when_true(end_condition))
        return motion

    def trade_places_motion(self) -> MotionStatechart:
        """
        The motion after a pour: the emptied source tilts back and stands upright, the
        filled receiver rises to where it pours back from, and the source comes down to
        where it is poured into.
        """
        pour_back = self.reversed()
        steps = Sequence(
            [
                self.pose_task("tilt back", self.source, self.pre_tilted_tool_pose()),
                self.pose_task("stand upright", self.source, self.loading_tool_pose()),
                self.pose_task(
                    "raise the filled cup", self.receiver, pour_back.loading_tool_pose()
                ),
                self.pose_task(
                    "lower the emptied cup", self.source, pour_back.receiver_tool_pose()
                ),
            ]
        )
        motion = MotionStatechart()
        motion.add_node(steps)
        motion.add_node(EndMotion.when_true(steps))
        return motion


# %% the demo


@dataclass
class FrontPourDemo:
    """
    Puts a cup into each gripper and pours the content back and forth between them.
    """

    giskard: MotionExecutor
    """
    Runs the motions.
    """

    left: HeldCup = field(init=False)
    """
    The cup in the left gripper, full at the start.
    """

    right: HeldCup = field(init=False)
    """
    The cup in the right gripper, empty at the start.
    """

    @property
    def world(self) -> World:
        return self.giskard.world

    def set_up(self) -> None:
        """
        Bring both arms to where the first pour starts and put a cup into each gripper.
        """
        park_arms(self.giskard)
        self.move_to_loading_poses()
        if self.world.get_semantic_annotations_by_name(LEFT_CUP_NAME):
            left_cup, right_cup = self._reuse_cups()
        else:
            left_cup, right_cup = self._attach_cups()
        self.left = HeldCup(
            cup=left_cup,
            tool_frame=self.world.get_body_by_name(LEFT_TOOL_FRAME),
            reach_heading=LEFT_REACH_HEADING,
        )
        self.right = HeldCup(
            cup=right_cup,
            tool_frame=self.world.get_body_by_name(RIGHT_TOOL_FRAME),
            reach_heading=RIGHT_REACH_HEADING,
        )

    def move_to_loading_poses(self) -> None:
        """
        Move both arms to :data:`_ARM_LOADING_POSITIONS`.
        """
        loading_state = JointState.from_mapping(
            {
                self.world.get_connection_by_name(name): position
                for name, position in zip(_ARM_JOINT_NAMES, _ARM_LOADING_POSITIONS)
            }
        )
        motion = MotionStatechart()
        loading_task = JointPositionList(goal_state=loading_state)
        motion.add_node(loading_task)
        motion.add_node(EndMotion.when_true(loading_task))
        self.giskard.execute(motion)

    def _attach_cup(self, name: str, tool_frame_name: str) -> HasFillLevel:
        """
        Hang a Jeroen cup from a tool frame, upright and gripped at its middle.
        """
        mesh = Mesh(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
            filename=_JEROEN_CUP_STL,
            scale=Scale(1, 1, 1),
        )
        body = Body.from_shape_collection(
            shape_collection=ShapeCollection([mesh]), name=PrefixedName(name)
        )
        with self.world.modify_world():
            self.world.add_body(body)
            self.world.add_connection(
                FixedConnection.create_with_dofs(
                    world=self.world,
                    parent=self.world.get_body_by_name(tool_frame_name),
                    child=body,
                    name=PrefixedName(f"{tool_frame_name}_T_{name}"),
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        roll=-math.pi / 2.0, y=-GRASP_HEIGHT
                    ),
                )
            )
        cup = HasFillLevel(name=PrefixedName(name), root=body)
        with self.world.modify_world():
            self.world.add_semantic_annotation(cup)
        return cup

    def _attach_cups(self) -> tuple[HasFillLevel, HasFillLevel]:
        """
        Put a filled cup into the left gripper and an empty one into the right, and
        couple each cup's fill to the other's outflow.

        :return: The left and the right cup.
        """
        left_cup = self._attach_cup(LEFT_CUP_NAME, LEFT_TOOL_FRAME)
        right_cup = self._attach_cup(RIGHT_CUP_NAME, RIGHT_TOOL_FRAME)
        for cup, fill in ((left_cup, START_FILL), (right_cup, 0.0)):
            cup.initialize_fill_level(
                world=self.world,
                initial_fill=fill,
                outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
                discharge_coefficient=DISCHARGE_COEFFICIENT,
                repose_angle=REPOSE_ANGLE,
            )
        right_cup.receive_outflow_from(source=left_cup, world=self.world)
        left_cup.receive_outflow_from(source=right_cup, world=self.world)
        return left_cup, right_cup

    def _reuse_cups(self) -> tuple[HasFillLevel, HasFillLevel]:
        """
        Take the cups a previous run left in a still running Giskard and reset their
        fills and their coupling.

        :return: The left and the right cup.
        """
        left_cup = self.world.get_semantic_annotation_by_name(LEFT_CUP_NAME)
        right_cup = self.world.get_semantic_annotation_by_name(RIGHT_CUP_NAME)
        for cup in (left_cup, right_cup):
            cup.fill_connection = self.world.get_connection(
                cup.fill_connection.parent, cup.fill_connection.child
            )
        with self.world.modify_world():
            JointState.from_mapping(
                {left_cup.fill_connection: START_FILL, right_cup.fill_connection: 0.0}
            ).apply_to(self.world)
        for source, receiver in ((left_cup, right_cup), (right_cup, left_cup)):
            receiver.recouple_outflow_from(
                source=source,
                world=self.world,
                fill_equation=replace(
                    source.create_pouring_equation(
                        self.world, OUTFLOW_RATE_CONSTANT, DISCHARGE_COEFFICIENT
                    ),
                    repose_angle=REPOSE_ANGLE,
                ),
            )
        return left_cup, right_cup

    def pours(self, number_of_pours: Optional[int]) -> Iterator[Pour]:
        """
        :param number_of_pours: How many pours to yield, or ``None`` for no end.
        :return: The pours in the order they run, starting from the left cup and
            alternating.
        """
        first = Pour(world=self.world, source=self.left, receiver=self.right)
        return islice(cycle([first, first.reversed()]), number_of_pours)

    def wait_for_synchronized_fills(self, pour: Pour) -> None:
        """
        Wait until the receiver's fill, which Giskard synchronizes back after the motion
        has ended, has arrived and stopped changing.
        """
        deadline = time.time() + FILL_SETTLE_TIMEOUT
        previous_fill = None
        while time.time() < deadline:
            current_fill = pour.receiver.cup.fill_level
            arrived = current_fill > FILL_TOLERANCE
            settled = (
                previous_fill is not None and abs(current_fill - previous_fill) < 1e-4
            )
            if arrived and settled:
                return
            previous_fill = current_fill
            time.sleep(0.1)

    def believe_everything_was_poured(self, pour: Pour, content: float) -> None:
        """
        Set the fills to an emptied source and a receiver holding ``content``.

        Nothing measures the cups, and what the two hold together does not change, so
        the believed fills are put back on that total after every pour instead of
        carrying the drain model's error into the next one.

        :param content: What both cups held together before the pour, as a fill.
        """
        with self.world.modify_world():
            JointState.from_mapping(
                {
                    pour.source.cup.fill_connection: 0.0,
                    pour.receiver.cup.fill_connection: content,
                }
            ).apply_to(self.world)

    def pour_back_and_forth(self, number_of_pours: Optional[int]) -> None:
        """
        Empty the filled cup into the other one, let the two trade places, and repeat.
        """
        for pour in self.pours(number_of_pours):
            content = pour.source.cup.fill_level + pour.receiver.cup.fill_level
            self.giskard.execute(pour.pour_motion())
            self.wait_for_synchronized_fills(pour)
            self.believe_everything_was_poured(pour, content)
            print(
                f"left cup: {self.left.cup.fill_level:.3f}, "
                f"right cup: {self.right.cup.fill_level:.3f}"
            )
            self.giskard.execute(pour.trade_places_motion())


if __name__ == "__main__":
    demo = FrontPourDemo(connect_to_giskard("tracy_front_pour_demo"))
    demo.set_up()
    demo.pour_back_and_forth(NUMBER_OF_POURS)
