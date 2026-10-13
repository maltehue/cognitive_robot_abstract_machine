import json
import math
import time
from importlib.resources import files
from pathlib import Path


from giskardpy.middleware.ros2.particle_weighing import (
    FULL_CUP_MASS_GRAMS,
    WEIGHED_PARTICLES_FILE_NAME,
    WeighedParticles,
)
from giskardpy.middleware.ros2.rosbag_recording import RosbagRecording
from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.pouring_model_verification.sensor import CameraCondition
from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.live_bank_listener import LiveBankListener
from giskardpy.motion_statechart.data_types import DefaultWeights
from krrood.adapters.json_serializer import to_json
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
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
    upright_carry_pose,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale
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
GOAL_FILL_CONST = 0.7
POUR_START_TILT_MARGIN = 0.05
"""
How far past the model's onset the cup is tilted before the transfer starts, in radians.

Below the onset the model has neither flow nor a gradient for the fill task to act on,
and the transfer never starts.
"""

REPOSE_ANGLE = math.radians(27.0)
"""
Tilt past the geometric onset before the balls start to flow, in radians.

Fitted with :attr:`OUTFLOW_RATE_CONSTANT` to the 2026-10-12 session on Tracy (five full-
cup calibration ramps and five weighed demo pours). The bulk stream starts later than
this, at about 70-75 degrees, but stops at the fill this angle predicts; one angle
cannot capture both.
"""

OUTFLOW_RATE_CONSTANT = 2.03
"""
How fast the source drains per unit of head above the lip; fitted together with
:attr:`REPOSE_ANGLE`.
"""

DISCHARGE_COEFFICIENT = 0.2
"""
Scales the exit speed and with it how far the stream is thrown; not yet calibrated for
the styrofoam balls.
"""

USE_LIVE_BANK = False
"""
Let the hypothesis bank watch the pour through the camera and correct the controller's
drain while it runs.

Needs ``stream_detector_node.py`` running and calibrated.
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


# ----- Record the run; the bag is closed again at the end of the script -----
recording = RosbagRecording()
print(f"Recording into {recording.start()}.")

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
                    roll=-math.pi / 2.0, y=-0.00
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
    )

    # ----- Place the receiving cup on the table -----
    receiving_cup_body = _spawn_jeroen_cup_body("receiving_cup")
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
        name=PrefixedName("receiving_cup"), root=receiving_cup_body
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
        # ``initialize_fill_level`` builds its drain without an onset angle; replacing it
        # here gives the first run the same drain as every later one.
        receiving_cup.recouple_outflow_from(
            source=source_cup, world=world, fill_equation=_analytic_drain(source_cup)
        )

time.sleep(0.2)

assert source_cup.fill_level == START_FILL
assert receiving_cup.fill_level == 0.0

# ----- Tilt to just past the model's onset -----
onset = _analytic_drain(source_cup).onset_tilt(START_FILL)
pour_start_tilt = float(onset.evaluate()[0]) + POUR_START_TILT_MARGIN
print(f"Tilting to the pour start at {math.degrees(pour_start_tilt):.1f} deg.")
move_left_tool_to(giskard, upright_carry_pose(giskard, tilt=pour_start_tilt))

# ----- Transfer -----
goal_fill = GOAL_FILL_CONST
tolerance = 0.05
# input("Press Enter to continue...")
transfer_task = FillByTransferTask(
    receiver=receiving_cup,
    goal_value=goal_fill,
    fill_level_tolerance=tolerance,
    reference_velocity=0.03,
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
    minimum_clearance=0.07,
    clearance_band=0.05,
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

extended_motion = Parallel([transfer_task, no_spill, keep_above, keep_plane])
msc = MotionStatechart()
msc.add_node(extended_motion)
msc.add_node(EndMotion.when_true(extended_motion))
live_bank_listener = None
if USE_LIVE_BANK:
    live_bank_listener = LiveBankListener(
        node=giskard.node_handle,
        bank=LiveBank.for_cups(
            source=source_cup,
            receiver=receiving_cup,
            world=world,
            built=ModelPoint(
                repose_angle=REPOSE_ANGLE,
                discharge_coefficient=DISCHARGE_COEFFICIENT,
                outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
            ),
            settings=BankSettings(),
            camera=CameraCondition.POLE_COUNTING.model,
            grains_per_fill=GRAINS_PER_FILL,
            loaded_grains=START_FILL * GRAINS_PER_FILL,
            goal_grains=goal_fill
            * GRAINS_PER_FILL
            * receiving_cup.capacity
            / source_cup.capacity,
            opening_radius=receiving_cup.opening_radius,
            prediction_window=transfer_task.prediction_duration,
        ),
    )
    live_bank_listener.start()
print("Start transfer.")
giskard.execute(msc)
if live_bank_listener is not None:
    live_bank_listener.stop()


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


move_left_tool_to(giskard, upright_pose)

recording.stop()

# ----- Record what the pour actually delivered and spilled, measured on a scale -----
weighed_particles = WeighedParticles.ask()
recording.add_file(WEIGHED_PARTICLES_FILE_NAME, weighed_particles.as_json_text())
if live_bank_listener is not None:
    weighed_grains = (
        (weighed_particles.received_mass_grams + weighed_particles.spilled_mass_grams)
        / FULL_CUP_MASS_GRAMS
        * GRAINS_PER_FILL
    )
    print(
        "Camera count against the scale: multiply PIXEL_SECONDS_PER_GRAIN in "
        "pouring_perception.py by "
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
