"""
Pour from the cup in Tracy's left gripper into the receiving cup on the table.

How to run it on the robot:

1. Start the robot and camera drivers, then ``tracy_velocity.py`` (it can keep running).
2. Start the perception once, for the material in the cup and the cup poured into, and
   leave it running::

       python pouring_perception.py --ros-args -p material:=corn -p receiver:=translucent_cup

   The translucent cup stands on the blue mat where the Jeroen cup stood; the
   perception finds it by the gap it makes in the mat, so keep the mat around it clear.
3. Set the switches below (:data:`MATERIAL`, :data:`RECEIVING_CUP`, :data:`DRAIN`,
   :data:`USE_LIVE_BANK`, :data:`CORRECT_DRAIN`) and run this script for every pour; after changing
   :data:`RECEIVING_CUP`, restart Giskard so it creates the cups anew. It prints what it runs with before the robot
   moves, refuses to start the live bank against a perception watching for another
   material, and records a bag with the bank's trace and the weighed result.
4. After the pour, weigh what reached the receiver and what landed beside it when asked
   (the receiver's fill is computed from its full mass in ``receiving_cup.py``);
   also weigh what is left in the source cup (note it down), which checks the full cup's
   mass.

Watch the stop button while the cup is above about 90 degrees.
"""

import json
import math
import shutil
import time
from importlib.resources import files
from pathlib import Path


from giskardpy.middleware.ros2.particle_weighing import (
    WEIGHED_PARTICLES_FILE_NAME,
    WeighedParticles,
)
from giskardpy.middleware.ros2.post_goal_plotters import GoalControlCycleRecorder
from giskardpy.middleware.ros2.rosbag_recording import RosbagRecording
from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.pouring_model_verification.sensor import CameraCondition
from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.live_bank_listener import LiveBankListener
from experiments.real_pouring.transfer_ending import (
    TRANSFER_ENDING_FILE_NAME,
    CancelOnInterrupt,
    TransferEnding,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.pour_perception import PourPerception
from experiments.real_pouring.pour_settings import (
    POUR_SETTINGS_FILE_NAME,
    PourLabel,
    PourSettings,
    RepositoryState,
    TiltPacingSettings,
)
from experiments.real_pouring.receiving_cup import (
    RECEIVING_CUP_FILE_NAME,
    ReceivingCup,
)
from giskardpy.middleware.ros2.exceptions import ExecutionCanceledException
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountSimulationTimeSeconds,
)
from krrood.adapters.json_serializer import to_json
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.pouring import (
    HeldAtTilt,
    TiltPacing,
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.equations.learned_pouring_equations import (
    LearnedHeadModelReference,
    couple_source_with_learned_head,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from tracy_pouring_setup import (
    LEFT_TOOL_FRAME,
    TABLE_SURFACE_Z,
    connect_to_giskard,
    move_left_tool_to,
    park_arms,
    turn_cup_upright_in_place,
    upright_carry_pose,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Cylinder, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# ------ Constants ----
_JEROEN_CUP_STL = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)
_JEROEN_CUP_SCALE = Scale(1, 1, 1)
IS_SIM = False

# START_FILL = 0.8
# GOAL_FILL_CONST = 0.7
START_FILL = 1.0

UNCALIBRATED_DRAIN = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)
"""
The drain the 2026-10-12 demo pours ran with, before any calibration: it delivered about
0.33 of a cup against 0.7 believed.
"""

FITTED_DRAIN = ModelPoint(
    repose_angle=math.radians(27.0),
    discharge_coefficient=0.2,
    outflow_rate_constant=2.03,
)
"""
Repose angle and drain rate fitted to the 2026-10-12 session on Tracy (five full-cup
calibration ramps and five weighed demo pours); the discharge coefficient is not
calibrated.

The bulk stream starts later than this angle, at about 70-75 degrees, but stops at the
fill it predicts; one angle cannot capture both. Open loop it tilts to about 106 degrees
and empties the cup.
"""

DRAIN = UNCALIBRATED_DRAIN
"""
The drain the controller and the hypothesis bank start from.
"""

REPOSE_ANGLE = DRAIN.repose_angle
"""
Tilt past the geometric onset before the balls start to flow, in radians.
"""

OUTFLOW_RATE_CONSTANT = DRAIN.outflow_rate_constant
"""
How fast the source drains per unit of head above the lip.
"""

DISCHARGE_COEFFICIENT = DRAIN.discharge_coefficient
"""
Scales the exit speed and with it how far the stream is thrown.
"""

MATERIAL = PouredMaterial.CORN
"""
What the source cup is filled with; ``pouring_perception.py`` has to watch for the same.
"""

RECEIVING_CUP = ReceivingCup.TRANSLUCENT_CUP
"""
The cup poured into; ``pouring_perception.py`` has to watch the same.
"""

GOAL_FILL_CONST = RECEIVING_CUP.goal_fill

RECEIVER_FULL_MASS_GRAMS = RECEIVING_CUP.full_mass_grams(MATERIAL)
"""
What the receiving cup full of the material weighs, read before the robot moves so a cup
nobody weighed full stops the demo here rather than after the pour.
"""

FULL_CUP_MASS_GRAMS = MATERIAL.full_cup_mass_grams
"""
What a full cup of the material weighs, read before the robot moves so a material nobody
weighed stops the demo here rather than after the pour.
"""

POURING_VELOCITY = 0.03
"""
How fast the transfer may raise the predicted fill of the receiver, in fill per second.
"""

STOPPING_VELOCITY = 1.0
"""
How fast the transfer may take back fill predicted beyond the goal, in fill per second:

a whole cup, so it never limits how fast the cup is turned back once the model predicts
an overshoot; :data:`MAXIMUM_RETURN_TILT_RATE` does.
"""

USE_LIVE_BANK = True
"""
Let the hypothesis bank watch the pour through the camera and correct the controller's
drain while it runs.

Needs ``pouring_perception.py`` running for :attr:`MATERIAL`.
"""

CORRECT_DRAIN = True
"""
Let the live bank correct the controller's drain; without, it tells the controller only
the fills the camera measures, and the controller pours on :data:`DRAIN` as built.
"""

BANK_SETTINGS = BankSettings(tracks_onset=True, trickle=0.0)
"""
How the live bank weighs the pour: every hypothesis estimates the onset, so a hand-over
keeps what the pour showed of it and a stream that stops where the model pours moves it
again; the controller runs on the onset the bank estimates, not on one held just below
the tilt, since the tilt pacing turns the cup at :data:`APPROACH_TILT_RATE` where the
model pours nothing.

Every hypothesis explains flow through the tilt-dependent drain represented by the
controller's equation.
"""

POUR_PERCEPTION = PourPerception.for_cup(RECEIVING_CUP, MATERIAL)
"""
Matching camera model, receiver evidence and requested stopping tolerance.
"""

RECEIVER_FILL_REPORT = POUR_PERCEPTION.fill_report
"""
The calibrated wall measurement determines received corn in the translucent cup.

Other cup and material combinations retain their existing reporting policy. Predicted
late arrivals are added separately.
"""

STOP_LEAD_TIME = 1.0
"""
How long grains keep arriving once the cup stops turning, in seconds: the count's rate
over this time is reported as already received, so the controller stops that much short
of the goal.

The ramps of 2026-10-12 relaxed within one to two seconds of reaching their tilt.
"""

COUPLE_SOURCE_FILL = True
"""
Let the source cup's fill drain along in the receiver's fill prediction, instead of
holding the source's current outflow over the whole prediction window; ``False`` for the
baseline of an A/B comparison.
"""

SETTLED_INFLOW = 0.005
"""
The inflow below which the receiver counts as no longer filling, in fill per second:

one of the bank's grains a second, a stray kernel rather than a stream.
"""

APPROACH_TILT_RATE = math.radians(3.0)
"""
How fast the cup keeps turning where the model predicts no flow, in radians per second:

slow enough for the camera and the bank to stop the pour within the degree or two the
delivered amount changes by five percent of a cup over.
"""

MAXIMUM_TILT_RATE = math.radians(3.0)
"""
The fastest the fill task may tilt the cup further, in radians per second.
"""

MAXIMUM_TILT = math.radians(110.0)
"""
The tilt the cup is never turned past, in radians: the calibration ramps emptied it
completely there.
"""

MINIMUM_TRANSFER_START_TILT = math.radians(2.0)
"""
The least tilt the transfer starts from, in radians: upright, the fill the model
predicts does not change with the tilt, which leaves the transfer no gradient to start
from.
"""

TRANSFER_START_TILT = max(
    MINIMUM_TRANSFER_START_TILT, MATERIAL.dry_tilt - math.radians(20.0)
)
"""
The tilt a plain pose goal turns the cup to before the transfer starts, in radians:
short of the material's dry tilt, so the transfer's approach covers the last stretch to
the first stream at its own rate; barely tilted for water, which spills at any tilt. At
50 degrees a brimful cup already let a few kernels fall on 2026-10-16.

The pose goal turns the cup without the pouring tasks, which chase a landing point and
an upright plane that move with the tilt: when the transfer itself turned the cup at 10
degrees per second through the dry stretch, the arm shook on 2026-10-15.
"""

CEILING_SLOWDOWN_WIDTH = math.radians(5.0)
"""
The span of tilt below :data:`MAXIMUM_TILT` over which the cup slows to a stop, in
radians: stopped at once, the arm chattered at the ceiling on 2026-10-15.
"""

CEILING_HOLD_SECONDS = 5.0
"""
How long the cup may stay at :data:`MAXIMUM_TILT` before the transfer ends whatever the
fill estimate says, in seconds: a cup held there that long has poured all it will.
"""

MAXIMUM_RETURN_TILT_RATE = math.radians(10.0)
"""
The fastest the cup may tilt back, in radians per second.

At the ramps' 30 degrees per second the arm lunged after every avalanche the count
reported on 2026-10-16 and lost the cup's axis by up to 5 degrees; the pour approaches
at :data:`APPROACH_TILT_RATE` and the stream stops within a few degrees of the onset, so
the return needs no more than this.
"""

MAXIMUM_TILT_ACCELERATION = math.radians(30.0)
"""
The fastest the cup's tilt rate may change, in radians per second squared.

Without a limit the fill task turned the cup back at once; faster than the wrist joint
that tilts it could follow, so the other joints joined in and moved the cup by
centimetres and its axis by up to 2.7 degrees (bag 2026-10-23_16-02-07). Replayed
through the controller, 30 degrees per second squared kept the other joints at the speed
they move at during the steady pour while tilting back nearly as far within two seconds.
"""

TRANSFER_TIME_LIMIT = 60.0
"""
How long the transfer may run before it is ended, in seconds, so a pour that stalls ends
by itself and the bag gets all its files; the longest pour of 2026-10-24 took 45 s.

Ctrl+C during the transfer ends it the same way: the robot stops and the script goes on
to weigh and save.
"""

MINIMUM_CLEARANCE = 0.07
"""
The lowest the source's lip may come above the receiver's rim during the transfer, in
metres.
"""

CLEARANCE_BAND = 0.05
"""
How far above :data:`MINIMUM_CLEARANCE` the lip may rise, in metres.

With a small :data:`TRANSFER_START_TILT` (a brimful cup of water allows about 2 degrees)
the stream falls almost straight down and the aim pulls the hand over the receiver; with
:data:`AVOID_RECEIVER_COLLISIONS`, widen this to 0.15 so collision avoidance has room to
lift the lip until the tilt grows.
"""

AVOID_RECEIVER_COLLISIONS = False
"""
Keep the left gripper and the source cup out of the receiver with collision avoidance.

On its own it cannot lift the hand above :data:`CLEARANCE_BAND`'s upper end, which the
clearance task holds at maximum weight.
"""

RECEIVER_COLLISION_BUFFER = 0.05
"""
Distance between the hand and the receiver at which collision avoidance starts pushing
them apart, in metres.
"""

CONTROL_CYCLES_FILE_NAME = "transfer_control_cycles.npz"
"""
Name the transfer's control-cycle recording gets inside the bag: what the controller was
asked and predicted every cycle, including the fill task's predicted fill and fill rate.
"""

LIVE_BANK_TRACE_FILE_NAME = "live_bank_trace.json"
"""
Name the bank's trace gets inside the bag.
"""

GRAINS_PER_FILL = 100.0
"""
How many of the bank's grains fill the source cup: one grain is a percent of the cup.
"""

USE_LEARNED_HEAD = False
"""
Drive the transfer with the learned head-above-lip surrogate instead of the analytic
head.

Only the model reference is sent to Giskard, which loads the checkpoint itself.
"""

_LEARNED_HEAD_CHECKPOINT = (
    "semantic_digital_twin/resources/learned_models/head_surrogate.pt"
)
"""
Workspace-relative path of the trained head surrogate; resolves on whichever machine
runs the Giskard process, since both sides run against the same workspace.
"""


def _learned_head_reference(source_cup: HasFillLevel) -> LearnedHeadModelReference:
    """
    Build the model reference for the source cup's geometry, failing early if untrained.

    The Giskard process rematerializes the torch model from this descriptor, so the
    checkpoint must exist in the workspace before the goal is sent.
    """
    equation = source_cup.fill_equation
    reference = LearnedHeadModelReference(
        checkpoint_path=_LEARNED_HEAD_CHECKPOINT,
        trained_container_height=equation.container_height,
        trained_container_width=equation.container_width,
    )
    if not reference.resolved_checkpoint_path().exists():
        raise SystemExit(
            f"learned head checkpoint missing: {reference.resolved_checkpoint_path()}\n"
            "train it first:\n"
            "python -m semantic_digital_twin.physics.equations.head_surrogate_training "
            f"--container-height {equation.container_height} "
            f"--container-width {equation.container_width} "
            f"--checkpoint {reference.resolved_checkpoint_path()}"
        )
    return reference


def _analytic_drain(source_cup: HasFillLevel) -> ArticulatedPouringEquation:
    """
    :return: The source's drain with this demo's onset and outflow constants, on the
        cup's own geometry.
    """
    equation = source_cup.fill_equation
    return ArticulatedPouringEquation(
        container_height=equation.container_height,
        container_width=equation.container_width,
        outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
        discharge_coefficient=DISCHARGE_COEFFICIENT,
        repose_angle=REPOSE_ANGLE,
    )


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


def _spawn_receiving_cup_body(cup: ReceivingCup) -> Body:
    """
    Create the receiving cup's Body: the Jeroen cup's mesh, or a cylinder of the cup's
    inner size standing on its frame.
    """
    if cup is ReceivingCup.JEROEN_CUP:
        return _spawn_jeroen_cup_body("receiving_cup")
    cylinder = Cylinder(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=cup.height / 2),
        width=2 * cup.inner_radius,
        height=cup.height,
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([cylinder]),
        name=PrefixedName("receiving_cup"),
    )


# ----- Record the run; the bag is closed again at the end of the script -----
recording = RosbagRecording(
    label=PourLabel(
        material=MATERIAL,
        receiving_cup=RECEIVING_CUP,
        goal_fill=GOAL_FILL_CONST,
        start_fill=START_FILL,
        approach_tilt_rate=APPROACH_TILT_RATE,
        live_bank=USE_LIVE_BANK,
        learned_head=USE_LEARNED_HEAD,
        corrects_drain=CORRECT_DRAIN,
    ).text()
)
print(f"Recording into {recording.start()}.")
recording.add_file(RECEIVING_CUP_FILE_NAME, RECEIVING_CUP.value)
print(
    f"Pouring {MATERIAL} (a full cup weighs {FULL_CUP_MASS_GRAMS} g) into the "
    f"{RECEIVING_CUP} (full {RECEIVER_FULL_MASS_GRAMS} g) to a fill of "
    f"{GOAL_FILL_CONST}; drain repose {math.degrees(REPOSE_ANGLE):.0f} deg, rate "
    f"{OUTFLOW_RATE_CONSTANT}, flowing down to "
    f"{math.degrees(MATERIAL.repose_hysteresis):.0f} deg below the repose angle with "
    f"the cup full, {math.degrees(MATERIAL.repose_hysteresis + MATERIAL.repose_hysteresis_growth):.0f} "
    "deg empty; "
    f"live bank {'off' if not USE_LIVE_BANK else 'on' if CORRECT_DRAIN else 'on, telling only the fills'}; source fill "
    f"{'drains along in' if COUPLE_SOURCE_FILL else 'held in'} the prediction; transfer from "
    f"{math.degrees(TRANSFER_START_TILT):.0f} deg at most "
    f"{math.degrees(MAXIMUM_TILT_RATE):.0f} deg/s, never past "
    f"{math.degrees(MAXIMUM_TILT):.0f} deg."
)

giskard = connect_to_giskard("tracy_transfer_demo")
robot = giskard.robot
world = giskard.world
print(f"world root: {world.root.name.name}")

# ----- Park arms -----
# park_arms(giskard)

# ------ Move the left gripper to the upright carry pose ----
left_tool_frame = world.get_body_by_name(LEFT_TOOL_FRAME)
upright_pose = upright_carry_pose(giskard)
move_left_tool_to(giskard, upright_pose)

# ----- Set up the cups on the first run, reuse them on subsequent runs -----
# Re-running the demo against an already-running Giskard process fetches a world that may still
# contain the cups from a previous run. Re-adding them would fail on duplicate bodies, so on reuse
# the cups are kept and only their fill levels are reset. This avoids restarting Giskard each time.
cups_already_present = bool(world.get_semantic_annotations_by_name("source_cup"))

if cups_already_present:
    print("Cups already present; reusing them and resetting fill levels.")
    source_cup = world.get_semantic_annotation_by_name("source_cup")
    receiving_cup = world.get_semantic_annotation_by_name("receiving_cup")
    RECEIVING_CUP.check_reused(
        receiver_capacity=receiving_cup.capacity,
        source_capacity=source_cup.capacity,
        material=MATERIAL,
    )
    # The fetched annotations reference detached copies of their fill connections; re-resolve the
    # world-resident ones so their fill state can be read and written.
    for cup in (source_cup, receiving_cup):
        cup.fill_connection = world.get_connection(
            cup.fill_connection.parent, cup.fill_connection.child
        )
    with world.modify_world():
        JointState.from_mapping(
            {source_cup.fill_connection: START_FILL, receiving_cup.fill_connection: 0.0}
        ).apply_to(world)
    # Give the reset fill state time to propagate to the Giskard process before the transfer goal.
    time.sleep(0.5)
    # Tear down whatever coupling the previous run left (analytic or learned) and re-couple with
    # the selected head model. The published equation swap marks Giskard's coupling stale, so it
    # rebuilds gate, inflow, and gated drain from the new model without a restart.
    if USE_LEARNED_HEAD:
        print("Re-coupling the transfer with the learned head surrogate.")
        couple_source_with_learned_head(
            receiving_cup, source_cup, world, _learned_head_reference(source_cup)
        )
    else:
        print("Re-coupling the transfer with the analytic head.")
        receiving_cup.recouple_outflow_from(
            source=source_cup, world=world, fill_equation=_analytic_drain(source_cup)
        )
else:
    # ----- Attach the source cup to the gripper -----
    old_number_bodies = len(world.bodies)

    source_cup_body = _spawn_jeroen_cup_body("source_cup")
    with world.modify_world():
        world.add_body(source_cup_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=left_tool_frame,
                child=source_cup_body,
                name=PrefixedName("l_gripper_T_source_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0, y=-0.08
                ),
            )
        )

    assert len(world.bodies) > old_number_bodies

    source_cup = HasFillLevel(name=PrefixedName("source_cup"), root=source_cup_body)
    with world.modify_world():
        world.add_semantic_annotation(source_cup)
    source_cup.initialize_fill_level(
        world=world,
        initial_fill=START_FILL,
        outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
        discharge_coefficient=DISCHARGE_COEFFICIENT,
        repose_angle=REPOSE_ANGLE,
    )

    # ----- Place the receiving cup on the table -----
    receiving_cup_body = _spawn_receiving_cup_body(RECEIVING_CUP)
    with world.modify_world():
        world.add_body(receiving_cup_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world,
                world.root,
                receiving_cup_body,
                name=PrefixedName("table_T_receiving_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 0.099, TABLE_SURFACE_Z
                ),
            )
        )
    receiving_cup = HasFillLevel(
        name=PrefixedName("receiving_cup"),
        root=receiving_cup_body,
        measured_capacity=source_cup.capacity
        * RECEIVING_CUP.source_cups_held(MATERIAL),
    )
    with world.modify_world():
        world.add_semantic_annotation(receiving_cup)
    time.sleep(0.2)
    receiving_cup.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )

    # ----- Couple the receiver's inflow to the source's gated outflow -----
    if USE_LEARNED_HEAD:
        print("Coupling the transfer with the learned head surrogate.")
        couple_source_with_learned_head(
            receiving_cup, source_cup, world, _learned_head_reference(source_cup)
        )
    else:
        # The source already has its calibrated drain: Giskard integrates the drain a cup
        # was created with until a pouring task rebuilds the coupling, so swapping it in
        # here would leave the tilt to the pour start draining by the old one.
        receiving_cup.receive_outflow_from(source=source_cup, world=world)

time.sleep(0.2)

assert source_cup.fill_level == START_FILL
assert receiving_cup.fill_level == 0.0

# ----- Tilt through the dry stretch with a plain pose goal -----
print(f"Tilting to the transfer start at {math.degrees(TRANSFER_START_TILT):.1f} deg.")
move_left_tool_to(giskard, upright_carry_pose(giskard, tilt=TRANSFER_START_TILT))
# Giskard integrates the drain during the pose goal too, and the uncalibrated drain lets a
# full cup flow at any tilt: start the transfer from the loaded cups again, as the live
# bank would tell them, so a pour without the bank does not count a phantom stream.
print(
    f"Fills believed after the tilt: source {source_cup.fill_level:.3f}, "
    f"receiver {receiving_cup.fill_level:.3f}; resetting them."
)
with world.modify_world():
    JointState.from_mapping(
        {source_cup.fill_connection: START_FILL, receiving_cup.fill_connection: 0.0}
    ).apply_to(world)

# ----- Transfer -----
goal_fill = GOAL_FILL_CONST
tolerance = POUR_PERCEPTION.fill_tolerance
# input("Press Enter to continue...")
transfer_task = FillByTransferTask(
    receiver=receiving_cup,
    goal_value=goal_fill,
    fill_level_tolerance=tolerance,
    reference_velocity=POURING_VELOCITY,
    stopping_velocity=STOPPING_VELOCITY,
    outflow_tolerance=SETTLED_INFLOW,
    couple_source_fill=COUPLE_SOURCE_FILL,
)
# Keep the liquid's projectile landing in the receiver so the optimizer repositions the gripper
# upstream as the source tilts and the arc reaches forward.
no_spill = KeepProjectileInReceiver(
    receiver=receiving_cup,
    source=source_cup,
    weight=DefaultWeights.WEIGHT_MAXIMUM,
    reference_velocity=0.1,
)
# Keep the pouring cup's rim (its lowest lip while tilting) above the receiving cup's rim, so the
# rims never collide no matter how far the source tilts. ``minimum_clearance`` is a true rim-to-rim
# gap: the task derives the lip from the live kinematics on Giskard's side, so it descends with the
# tilt instead of relying on a hand-picked offset on the cup origins.
keep_above = KeepSourceRimAboveReceiverRim(
    receiver=receiving_cup,
    source=source_cup,
    minimum_clearance=MINIMUM_CLEARANCE,
    clearance_band=CLEARANCE_BAND,
    weight=DefaultWeights.WEIGHT_MAXIMUM,
)
keep_plane = AlignPlanes(
    root_link=world.root,
    tip_link=left_tool_frame,
    goal_normal=Vector3.X(reference_frame=world.root),
    tip_normal=Vector3.Z(reference_frame=left_tool_frame),
)
# motion = Parallel([no_spill, keep_above, keep_plane])
# msc_transfer = MotionStatechart()
# msc_transfer.add_node(motion)
# msc_transfer.add_node(EndMotion.when_true(motion))
# giskard.execute(msc_transfer)

# Pace the tilt: keep the cup turning where the model predicts no flow until the receiver
# holds the goal, never faster than the cap once it does, and never past the tilt that
# empties it.
tilt_pacing = TiltPacing(
    source=source_cup,
    approach_rate=APPROACH_TILT_RATE,
    maximum_rate=MAXIMUM_TILT_RATE,
    maximum_tilt=MAXIMUM_TILT,
    maximum_return_rate=MAXIMUM_RETURN_TILT_RATE,
    maximum_acceleration=MAXIMUM_TILT_ACCELERATION,
    ceiling_slowdown_width=CEILING_SLOWDOWN_WIDTH,
    receiver=receiving_cup,
    approach_until_fill=goal_fill - tolerance,
)
recording.add_file(
    POUR_SETTINGS_FILE_NAME,
    PourSettings(
        goal_fill=transfer_task.goal_value,
        fill_tolerance=transfer_task.fill_level_tolerance,
        couple_source_fill=transfer_task.couple_source_fill,
        prediction_duration=transfer_task.prediction_duration,
        start_fill=START_FILL,
        material=MATERIAL,
        receiving_cup=RECEIVING_CUP,
        pouring_velocity=transfer_task.reference_velocity,
        stopping_velocity=transfer_task.stopping_velocity,
        settled_inflow=transfer_task.outflow_tolerance,
        tilt_pacing=TiltPacingSettings(
            approach_rate=tilt_pacing.approach_rate,
            maximum_rate=tilt_pacing.maximum_rate,
            maximum_return_rate=tilt_pacing.maximum_return_rate,
            maximum_acceleration=tilt_pacing.maximum_acceleration,
            maximum_tilt=tilt_pacing.maximum_tilt,
            ceiling_slowdown_width=tilt_pacing.ceiling_slowdown_width,
            transfer_start_tilt=TRANSFER_START_TILT,
            ceiling_hold_seconds=CEILING_HOLD_SECONDS,
        ),
        drain=MATERIAL.drain_of(DRAIN),
        bank=BANK_SETTINGS if USE_LIVE_BANK else None,
        stop_lead_time=STOP_LEAD_TIME if USE_LIVE_BANK else None,
        corrects_drain=CORRECT_DRAIN if USE_LIVE_BANK else None,
        learned_head=USE_LEARNED_HEAD,
        transfer_time_limit=TRANSFER_TIME_LIMIT,
        repository=RepositoryState.of_working_tree(Path(__file__).parent),
    ).as_json_text(),
)
extended_motion = Parallel(
    [transfer_task, no_spill, keep_above, keep_plane, tilt_pacing]
)
# End the transfer if the cup has stayed at its ceiling for long, even when the fill
# estimate never reached the goal (a camera that counted too little).
held_at_ceiling = HeldAtTilt(
    source=source_cup, tilt=MAXIMUM_TILT, seconds=CEILING_HOLD_SECONDS
)
msc = MotionStatechart()
msc.add_node(extended_motion)
msc.add_node(EndMotion.when_true(extended_motion))
msc.add_node(held_at_ceiling)
msc.add_node(EndMotion.when_true(held_at_ceiling))
time_limit = CountSimulationTimeSeconds(seconds=TRANSFER_TIME_LIMIT)
msc.add_node(time_limit)
msc.add_node(EndMotion.when_true(time_limit))
if AVOID_RECEIVER_COLLISIONS:
    [tracy] = world.get_semantic_annotations_by_type(Tracy)
    hand = [source_cup.root] + [
        body
        for body in tracy.left_arm.end_effector.bodies_with_collision
        if body is not source_cup.root
    ]
    msc.add_node(
        UpdateTemporaryCollisionRules(
            temporary_rules=[
                AvoidCollisionBetweenGroups(
                    buffer_zone_distance=RECEIVER_COLLISION_BUFFER,
                    violated_distance=0.0,
                    body_group_a=hand,
                    body_group_b=[receiving_cup.root],
                )
            ]
        )
    )
    msc.add_node(
        ExternalCollisionAvoidance(robot=tracy, cancel_if_collision_violated=False)
    )
live_bank_listener = None
if USE_LIVE_BANK:
    live_bank_listener = LiveBankListener(
        node=giskard.node_handle,
        material=MATERIAL,
        bank=LiveBank.for_cups(
            source=source_cup,
            receiver=receiving_cup,
            world=world,
            built=MATERIAL.drain_of(DRAIN),
            settings=BANK_SETTINGS,
            camera=CameraCondition.POLE_COUNTING.model,
            grains_per_fill=GRAINS_PER_FILL,
            loaded_grains=START_FILL * GRAINS_PER_FILL,
            goal_grains=goal_fill
            * GRAINS_PER_FILL
            * receiving_cup.capacity
            / source_cup.capacity,
            opening_radius=receiving_cup.opening_radius,
            prediction_window=transfer_task.prediction_duration,
            fill_report=RECEIVER_FILL_REPORT,
            receiver_evidence=POUR_PERCEPTION.receiver_evidence,
            stop_lead_time=STOP_LEAD_TIME,
            corrects_drain=CORRECT_DRAIN,
            in_flight_limit=tolerance
            * GRAINS_PER_FILL
            * receiving_cup.capacity
            / source_cup.capacity,
        ),
    )
    live_bank_listener.start()
print("Start transfer.")
transfer_started_at = time.time()
with CancelOnInterrupt(giskard.cancel_goal_async) as interruption:
    try:
        giskard.execute(msc)
    except ExecutionCanceledException:
        pass
if interruption.interrupted:
    transfer_ending = TransferEnding.STOPPED_BY_OPERATOR
elif msc.observation_state[held_at_ceiling] == ObservationStateValues.TRUE:
    transfer_ending = TransferEnding.HELD_AT_CEILING
elif msc.observation_state[time_limit] == ObservationStateValues.TRUE:
    transfer_ending = TransferEnding.TIME_LIMIT
else:
    transfer_ending = TransferEnding.GOAL_REACHED
print(f"Transfer ended: {transfer_ending}.")
recording.add_file(TRANSFER_ENDING_FILE_NAME, transfer_ending)
if live_bank_listener is not None:
    live_bank_listener.stop()
transfer_control_cycles = GoalControlCycleRecorder.recording_written_since(
    transfer_started_at
)
if transfer_control_cycles is None:
    print(
        "No control-cycle recording of the transfer on this machine: start Giskard "
        "with record_control_cycles:=true on this machine to keep it in the bag."
    )
else:
    shutil.copy(transfer_control_cycles, recording.bag_path / CONTROL_CYCLES_FILE_NAME)


# ``giskard.execute`` returns on the action result, but the final fill levels are synchronized
# back from the Giskard process asynchronously. Wait until the receiver's fill reflects the
# completed transfer and settles, so the printed values are the transferred amounts rather than
# a stale pre-transfer belief.
settle_deadline = time.time() + 10.0
previous_fill = None
while time.time() < settle_deadline:
    current_fill = receiving_cup.fill_level
    transfer_reflected = current_fill > tolerance
    settled = previous_fill is not None and abs(current_fill - previous_fill) < 1e-4
    if transfer_reflected and settled:
        break
    previous_fill = current_fill
    time.sleep(0.1)

print(f"receiving cup fill level: {receiving_cup.fill_level}")
print(f"source cup fill level: {source_cup.fill_level}")

# ----- Turn the cup upright over the receiver before carrying it away -----
# A pose goal turns and moves the cup at once, carrying it off the receiver while it is
# still tilted far enough to pour.
turn_cup_upright_in_place(
    giskard,
    tilt=float(source_cup.pour_tilt_expression.evaluate()[0]),
    rate=MAXIMUM_RETURN_TILT_RATE,
)
move_left_tool_to(giskard, upright_pose)

recording.stop()

# ----- Record what the pour actually delivered and spilled, measured on a scale -----
weighed_particles = WeighedParticles.ask(full_cup_mass_grams=RECEIVER_FULL_MASS_GRAMS)
recording.add_file(WEIGHED_PARTICLES_FILE_NAME, weighed_particles.as_json_text())
if live_bank_listener is not None:
    weighed_grains = (
        (weighed_particles.received_mass_grams + weighed_particles.spilled_mass_grams)
        / FULL_CUP_MASS_GRAMS
        * GRAINS_PER_FILL
    )
    print(
        "Camera count against the scale: multiply the pixel seconds per grain of "
        f"{MATERIAL} in experiments/real_pouring/poured_material.py by "
        f"{live_bank_listener.bank.count_calibration_factor(weighed_grains):.2f} "
        "to make them agree."
    )
    recording.add_file(
        LIVE_BANK_TRACE_FILE_NAME,
        json.dumps([to_json(step) for step in live_bank_listener.bank.trace]),
    )
print(f"Measured fill level: {weighed_particles.received_fill_level:.3f}")
print(f"Spilled: {weighed_particles.spilled_mass_grams} g")
print(f"Recorded {recording.bag_path}.")
