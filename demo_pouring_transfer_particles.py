"""
Tracy pours a cup of particles into another cup, under the pouring tasks.

The motion is the one :mod:`giskardpy.motion_statechart.tasks.pouring` drives from the
receiver's fill level: the controller tilts the grasped source until the receiver's
level reaches the goal, keeping the projectile in the receiver and the source's rim
above it. What the controller reasons about is still that level, integrated from the
pouring equation; what actually leaves the cup is a few hundred spheres, so the run
says where the pour it commanded really put them.

Giskard's control loop and the physics run in lockstep: every cycle's command becomes
the servos' set point and the physics advances one cycle before the next.

Run it with ``--headless`` for the numbers alone, or without for the MuJoCo viewer.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from datetime import timedelta

from typing_extensions import Callable, List, Optional

import mujoco

from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.motion_statechart.data_types import DefaultWeights
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.physics.drain_calibration import CalibratedDrainScale
from semantic_digital_twin.physics.particles import (
    HollowCylinder,
    MeasuredCommittedFillLevel,
    MeasuredFillLevel,
    MeasuredInflowRate,
    ParticleFill,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% the scene

SOURCE = HollowCylinder(inner_radius=0.035, height=0.06)
"""
The cup the robot holds and pours from.
"""

RECEIVER = HollowCylinder(inner_radius=0.06, height=0.1)
"""
The cup standing on the table, which the pour has to reach.
"""

PARTICLE_RADIUS = 0.005
"""
Radius of one particle, in metres.
"""

PACKED_SHARE = 0.45
"""
The share of the source cup's cavity to pack with grains.

What the grains settle to is read off them rather than predicted from this, so the fill
the model starts at is right whatever this is set to. Below a full cup on purpose: the
physics costs a contact per pair of touching grains, and a cup packed to its rim runs
several times slower than the rest of the scene.
"""

GOAL_FILL = 0.06
"""
The receiver's fill level the motion is commanded to reach.

A fill level is a share of the container's own capacity, and this receiver holds about
four and a half times what the source does, so everything the source can pour reaches
only about ``0.13`` of it. The goal is set below that with room to stop in.
"""

FILL_TOLERANCE = 0.01
"""
How close to :data:`GOAL_FILL` counts as reached.
"""

RECEIVER_STAND = (1.0, 0.1)
"""
Where the receiving cup stands on the table, as x and y in metres.
"""

CARRY_STAND = (1.0, 0.2, 0.3)
"""
Where the source cup is carried before the pour, as x and y in metres and a height
above the table: beside the receiver and well clear of it.
"""

CARRY_ORIENTATION = (0.5, 0.5, 0.5, 0.5)
"""
The orientation the hand carries the cup in, as a quaternion in x, y, z, w: the one
that stands the cup upright in the gripper.
"""

POUR_START_TILT_MARGIN = 0.05
"""
How far past the tilt at which the contents reach the lip the cup is carried, in
radians.

Below that tilt the drain model has no flow and no gradient, so the fill task could not
discover that tilting further starts the pour.
"""

WRIST_NUDGE = 0.1
"""
How far the wrist is turned out of the carry pose, in radians, so the pour does not
start from a singular wrist.
"""

GRASP_ROLL = -math.pi / 2
"""
How the source cup sits in the gripper, as a roll about the tool's own x axis: upright,
with its opening away from the palm.
"""

MINIMUM_RIM_CLEARANCE = 0.1
"""
How far the source's rim is kept above the receiver's, in metres.
"""

CONTROL_FREQUENCY = 80
"""
How often the controller ticks, in hertz.
"""

PREDICTION_HORIZON = 20
"""
How many control cycles the quadratic program looks ahead.
"""

STEP_SIZE = 2e-3
"""
Physics step, in seconds.
"""

INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
"""
Integrator the physics runs with; a pour is decided by the contact solver rather than
by the smooth dynamics between contacts.
"""

TICK_LIMIT = 4000
"""
How many control cycles the motion is given before the run gives up.
"""

SETTLE_TIME = timedelta(seconds=3)
"""
How long the contents settle in the cup before the pour starts.

A packing is released looser than what it settles into and takes a couple of seconds to
compact; read too early, a full cup reads half full and the pour starts against a fill
level the contents do not have.
"""

PERCEPTION_FREQUENCY = 10
"""
How often the cups' fill levels are measured from their contents and reported into the
controller's model, in hertz.

Without it the controller reasons about the pour its own drain model predicts. With it
the levels it steers by are the ones the grains actually produced, on both cups.
"""

TRANSFER_REFERENCE_VELOCITY = 0.03
"""
How fast the fill task asks its fill level to change, per second.

It sets how quickly the cup turns, and so how far past the tilt at which the contents
start moving the cup gets before any of them have arrived to say so.
"""

CALIBRATION_SMOOTHING = 0.3
"""
How far towards each measured inflow the drain's factor moves.
"""

REPORT_EVERY = 200
"""
How many control cycles pass between two lines of the report.
"""

OUTFLOW_RATE_CONSTANT = 0.24
"""
Outflow rate constant of the source's drain.

Fitted to the relaxation the grains of this scene actually show, 6.35 s at 85 degrees,
against the model's own time constant at :data:`SOURCE_REPOSE_ANGLE`. The two cannot be
fitted apart: the time constant moves 7.6-fold between a repose of zero and this one.
"""

SOURCE_REPOSE_ANGLE = math.radians(36.8)
"""
Angle the source's contents hold before they start to flow, in radians.

Fitted to the tilt-against-retained-fill curve of this cup, which it reproduces to 0.007
in fill with every geometric parameter left at its measured value. Fitted rather than
measured on purpose: the grains heap at about zero degrees on a flat plate, so this
number is not the contents' angle of repose and does not carry to another container.
See ``semantic_digital_twin/doc/pouring_effect_model_evaluation.md``.
"""

FILL_PREDICTION_WINDOW = 3.0
"""
How far ahead the fill row predicts, in seconds.

About half the contents' measured relaxation. The task's own default of 1.5 s is a
quarter of it, which leaves the fill barely moving inside the horizon, so the controller
reads its tilt as ineffective and asks for more of it.
"""


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    A container whose fill level the pouring equation integrates as it tilts.
    """


# %% building the world


@dataclass
class TransferScene:
    """
    Everything a run of the transfer needs to reach.
    """

    world: World
    """
    The world the robot and the cups stand in.
    """

    robot: Tracy
    """
    The robot holding the source cup.
    """

    source: PourableContainer
    """
    The cup the robot pours from.
    """

    receiver: PourableContainer
    """
    The cup on the table the pour has to reach.
    """

    tool_frame: Body
    """
    The frame of the hand holding the source.
    """

    drain_scale: FloatVariable
    """
    The factor the source's drain is scaled by, which the measured inflow corrects.
    """

    variables: FloatVariableData
    """
    The one place the factor's value lives, shared by every controller of the run so
    they all read the same correction.
    """


def build_scene(repose_angle: float = SOURCE_REPOSE_ANGLE) -> TransferScene:
    """
    Build Tracy with both arms parked, a source cup in its left hand and a receiving
    cup on the table, coupled so what leaves one enters the other.

    :param repose_angle: Angle the source's contents hold before they flow. Zero is the
        model as it stands before any parameter of the contents is estimated.
    :return: The scene.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    robot = RobotSpecification(Tracy).spawn(world)
    for arm in robot.get_arms():
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()

    tool_frame = robot.left_arm.end_effector.tool_frame
    source_body = SOURCE.body(PrefixedName("source_cup"))
    receiver_body = RECEIVER.body(PrefixedName("receiving_cup"))
    with world.modify_world():
        world.add_kinematic_structure_entity(source_body)
        world.add_connection(
            FixedConnection(
                parent=tool_frame,
                child=source_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=GRASP_ROLL, reference_frame=tool_frame
                ),
            )
        )
        world.add_kinematic_structure_entity(receiver_body)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=receiver_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=RECEIVER_STAND[0],
                    y=RECEIVER_STAND[1],
                    z=robot.table.top_z,
                    reference_frame=world.root,
                ),
            )
        )

    source = PourableContainer(name=PrefixedName("source"), root=source_body)
    receiver = PourableContainer(name=PrefixedName("receiver"), root=receiver_body)
    with world.modify_world():
        world.add_semantic_annotation(source)
        world.add_semantic_annotation(receiver)
    source.initialize_fill_level(
        world=world,
        initial_fill=PACKED_SHARE,
        outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
    )
    if repose_angle:
        with world.modify_world():
            source.add_fill_equation(
                replace(source.fill_equation, repose_angle=repose_angle)
            )
    receiver.initialize_fill_level(world=world, initial_fill=0.0)
    drain_scale = FloatVariable("source_drain_scale")
    variables = FloatVariableData()
    variables.register_expression(drain_scale)
    variables.set_value(drain_scale, 1.0)
    with world.modify_world():
        source.add_fill_equation(
            replace(source.fill_equation, outflow_scale=drain_scale)
        )
    receiver.receive_outflow_from(source=source, world=world)
    return TransferScene(
        world=world,
        robot=robot,
        source=source,
        receiver=receiver,
        tool_frame=tool_frame,
        drain_scale=drain_scale,
        variables=variables,
    )


# %% the motion


def build_carry_motion(scene: TransferScene, tilt: float = 0.0) -> MotionStatechart:
    """
    Build the motion that brings the hand to the pose it carries the cup in.

    :param scene: The scene the motion runs in.
    :param tilt: How far the cup is turned about its own pouring axis on the way, in
        radians.
    :return: The statechart, ending when the hand has reached the carry pose.
    """
    world_T_carry = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=CARRY_STAND[0],
        pos_y=CARRY_STAND[1],
        pos_z=scene.robot.table.top_z + CARRY_STAND[2],
        quat_x=CARRY_ORIENTATION[0],
        quat_y=CARRY_ORIENTATION[1],
        quat_z=CARRY_ORIENTATION[2],
        quat_w=CARRY_ORIENTATION[3],
        reference_frame=scene.world.root,
    )
    carry = CartesianPose(
        root_link=scene.world.root,
        tip_link=scene.tool_frame,
        goal_pose=(
            world_T_carry @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=tilt)
        ).to_pose(),
    )
    statechart = MotionStatechart()
    statechart.add_node(carry)
    statechart.add_node(EndMotion.when_true(carry))
    return statechart


def _opening_height(world: World, container: PourableContainer) -> float:
    """
    How high a container's opening stands in the world, in metres.

    Its origin sits at the base of its collision geometry, so the opening is one
    container height above it.

    :param world: The world the container stands in.
    :param container: The container.
    :return: The height, in metres.
    """
    base = world.compute_forward_kinematics_np(world.root, container.root)[2, 3]
    return float(base) + container.root.collision.height


def pour_start_tilt(source: PourableContainer, contents: ParticleFill) -> float:
    """
    The tilt at which the source's contents reach its lip, plus a margin, so the drain
    has flow and gradient from the start.

    Read off how far the contents reach up the cup rather than off its fill level: the
    level the controller steers by counts what is in the cup, while the lip is a
    question about how deep it stands. Contents that stand in a heap reach the lip only
    once the surface has steepened past the angle they hold, so that angle is part of
    the tilt the pour has to start from.

    :param source: The cup about to pour.
    :param contents: The contents standing in it.
    :return: The tilt, in radians.
    """
    equation = source.fill_equation.ungated()
    depth = contents.filled_height_in(source.root)
    onset = equation.onset_tilt(depth).evaluate()[0]
    return float(onset) + POUR_START_TILT_MARGIN


def build_transfer_motion(
    scene: TransferScene, prediction_window: float = FILL_PREDICTION_WINDOW
) -> MotionStatechart:
    """
    Build the motion that pours the source into the receiver.

    :param scene: The scene the motion runs in.
    :param prediction_window: How far ahead the fill row predicts, in seconds.
    :return: The statechart, ending when the receiver has reached its goal.
    """
    transfer = FillByTransferTask(
        receiver=scene.receiver,
        goal_value=GOAL_FILL,
        fill_level_tolerance=FILL_TOLERANCE,
        reference_velocity=TRANSFER_REFERENCE_VELOCITY,
        prediction_duration=prediction_window,
    )
    no_spill = KeepProjectileInReceiver(receiver=scene.receiver, source=scene.source)
    keep_above = KeepSourceRimAboveReceiverRim(
        receiver=scene.receiver,
        source=scene.source,
        minimum_clearance=MINIMUM_RIM_CLEARANCE,
        clearance_band=0.02,
        weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
    )
    keep_upright = AlignPlanes(
        root_link=scene.world.root,
        tip_link=scene.tool_frame,
        goal_normal=Vector3.X(reference_frame=scene.world.root),
        tip_normal=Vector3.Z(reference_frame=scene.tool_frame),
    )
    motion = Parallel([transfer, no_spill, keep_above, keep_upright])
    statechart = MotionStatechart()
    statechart.add_node(motion)
    statechart.add_node(EndMotion.when_true(motion))
    return statechart


# %% running the pour


def run(headless: bool, nominal: bool = False) -> None:
    """
    Carry the cup upright, fill it, pour it into the receiver under the pouring tasks,
    and report where the contents went.

    :param headless: Whether to run without the MuJoCo viewer.
    """
    scene = build_scene(repose_angle=0.0 if nominal else SOURCE_REPOSE_ANGLE)
    print(
        "the source's contents are modelled as holding "
        f"{math.degrees(0.0 if nominal else SOURCE_REPOSE_ANGLE):.1f} degrees "
        f"before they flow, and the fill is predicted "
        f"{FILL_PREDICTION_WINDOW if not nominal else 1.5} s ahead"
    )
    ParticleFill.settling_contact().apply_to(scene.world.bodies)

    simulation = MujocoSim(
        world=scene.world,
        headless=headless,
        step_size=STEP_SIZE,
        integrator=INTEGRATOR,
    )
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(milliseconds=20))
        _run_motion(build_carry_motion(scene), scene, simulation, "carry")
        _nudge_wrist(scene)

        fill = SOURCE.fill_with_particles(
            container=scene.source.root,
            world=scene.world,
            simulator=simulation.simulator,
            particle_radius=PARTICLE_RADIUS,
            count=SOURCE.particle_capacity(PARTICLE_RADIUS, fill_fraction=PACKED_SHARE),
        )
        simulation.step_simulation(SETTLE_TIME)
        fill.bulk_volume_per_particle = fill.bulk_volume_per_particle_in(
            scene.source.root, scene.source.capacity
        )
        _seed_fill_level_from_the_contents(scene, fill)
        perception = [
            MeasuredFillLevel(
                contents=fill,
                container=scene.source.root,
                capacity=scene.source.capacity,
                connection=scene.source.fill_connection,
                world=scene.world,
            ),
            MeasuredCommittedFillLevel(
                contents=fill,
                container=scene.receiver.root,
                capacity=scene.receiver.capacity,
                connection=scene.receiver.fill_connection,
                world=scene.world,
                source=scene.source.root,
                opening_height=_opening_height(scene.world, scene.receiver),
            ),
        ]
        for measurement in perception:
            measurement.report()
        print(
            f"{len(fill.names)} particles, "
            f"{fill.count_inside(scene.source.root)} of them in the source cup after "
            f"settling; both levels start from what is measured, source "
            f"{scene.source.fill_level:.2f} and receiver "
            f"{scene.receiver.fill_level:.2f}"
        )

        arriving = MeasuredInflowRate(level=perception[1])
        tilt = pour_start_tilt(scene.source, fill)
        print(f"carrying the cup pre-tilted to {tilt:.3f} rad, where its drain starts")
        _run_motion(
            build_carry_motion(scene, tilt=tilt), scene, simulation, "pre-tilt", fill
        )
        _run_motion(
            build_transfer_motion(
                scene, prediction_window=1.5 if nominal else FILL_PREDICTION_WINDOW
            ),
            scene,
            simulation,
            "transfer",
            fill=fill,
            perception=perception,
            arriving=arriving,
        )
        _summarize(scene, fill)
    finally:
        simulation.stop_simulation()


def _run_motion(
    statechart: MotionStatechart,
    scene: TransferScene,
    simulation: MujocoSim,
    name: str,
    fill: Optional[ParticleFill] = None,
    perception: Optional[List[MeasuredFillLevel]] = None,
    arriving: Optional[MeasuredInflowRate] = None,
) -> None:
    """
    Tick one motion to its end in lockstep with the physics.

    :param statechart: The motion to run.
    :param scene: The scene the motion runs in.
    :param simulation: The simulation the controller runs in lockstep with.
    :param name: What to call the motion in the report.
    :param fill: The contents to report on, if there are any yet.
    :param perception: What reports the containers' fill levels into the controller's
        model, if the run closes that loop.
    :param arriving: What measures how fast the contents reach the receiver, if the run
        watches that.
    """
    executor = Executor(
        context=MotionStatechartContext(
            world=scene.world,
            qp_controller_config=QPControllerConfig(
                target_frequency=CONTROL_FREQUENCY,
                prediction_horizon=PREDICTION_HORIZON,
            ),
            float_variable_data=scene.variables,
        ),
        pacer=SteppedSimulationPacer(simulation),
    )
    calibration = (
        CalibratedDrainScale(
            scale=scene.drain_scale,
            inflow=scene.receiver.fill_connection.inflow_equation.symbolic_velocity(
                scene.receiver.fill_connection
            ),
            variables=scene.variables,
            smoothing=CALIBRATION_SMOOTHING,
        )
        if arriving is not None
        else None
    )
    executor.compile(motion_statechart=statechart)
    if fill is not None:
        executor.tick = _reporting_tick(
            executor, scene, fill, perception, arriving, calibration
        )
    try:
        executor.tick_until_end(timeout=TICK_LIMIT)
        print(
            f"the {name} motion reached its goal after "
            f"{int(executor.control_cycles)} cycles"
        )
    except TimeoutError:
        print(f"the {name} motion did not end within {TICK_LIMIT} cycles")


def _reporting_tick(
    executor: Executor,
    scene: TransferScene,
    fill: ParticleFill,
    perception: Optional[List[MeasuredFillLevel]],
    arriving: Optional[MeasuredInflowRate],
    calibration: Optional[CalibratedDrainScale],
) -> Callable[[], None]:
    """
    Wrap an executor's tick so the receiver is measured and the pour reported as the
    motion runs.

    :param executor: The executor whose ticks are wrapped.
    :param scene: The scene the motion runs in.
    :param fill: The contents being poured.
    :param perception: What reports the containers' fill levels, if the run closes
        that loop.
    :param arriving: What measures how fast the contents reach the receiver.
    :param calibration: What holds the drain's factor at what that measurement says.
    :return: The wrapped tick.
    """
    tick = executor.tick
    cycles_between_measurements = max(
        1, round(CONTROL_FREQUENCY / PERCEPTION_FREQUENCY)
    )

    measured_rate = [0.0]

    def tick_and_report() -> None:
        tick()
        cycle = int(executor.control_cycles)
        if cycle % cycles_between_measurements == 0:
            if perception is not None:
                for measurement in perception:
                    measurement.report()
            if arriving is not None:
                measured_rate[0] = arriving.observe(at=cycle / CONTROL_FREQUENCY)
                if calibration is not None:
                    calibration.calibrate(measured_inflow=measured_rate[0])
        if cycle % REPORT_EVERY == 0:
            _report(cycle, scene, fill, measured_rate[0], calibration)

    return tick_and_report


def _seed_fill_level_from_the_contents(
    scene: TransferScene, contents: ParticleFill
) -> None:
    """
    Start the source's fill level where its contents actually stand.

    A packing is released looser than what it settles into, so how full a count of
    grains makes a cup is something to read off them once they have come to rest rather
    than something to predict from the count.

    :param scene: The scene whose source is to be seeded.
    :param contents: The grains standing in that source.
    """
    JointState.from_mapping(
        {scene.source.fill_connection: contents.filled_height_in(scene.source.root)}
    ).apply_to(scene.world)


def _nudge_wrist(scene: TransferScene) -> None:
    """
    Turn the wrist out of the carry pose, so the pour does not start from a wrist that
    can only tilt one way.

    :param scene: The scene whose robot is nudged.
    """
    wrist = scene.world.get_connection_by_name("left_wrist_3_joint")
    JointState.from_mapping({wrist: wrist.position + WRIST_NUDGE}).apply_to(scene.world)


def _report(
    tick: int,
    scene: TransferScene,
    fill: ParticleFill,
    measured_rate: float = 0.0,
    calibration: Optional[CalibratedDrainScale] = None,
) -> None:
    """
    Print one line comparing the commanded fill levels and inflow against the particles.

    :param tick: Which control cycle this is.
    :param scene: The scene being reported on.
    :param fill: The contents being poured.
    :param measured_rate: How fast the contents were last seen reaching the receiver.
    :param calibration: What holds the drain's factor, if the run corrects it.
    """
    in_source = fill.count_inside(scene.source.root)
    in_receiver = fill.count_inside(scene.receiver.root)
    print(
        f"cycle {tick:5d}  tilt {math.degrees(_cup_tilt(scene)):5.1f} deg  "
        f"depth {fill.filled_height_in(scene.source.root):4.2f}  "
        f"source: {in_source:3d} particles vs {scene.source.fill_level:4.2f} level  "
        f"receiver: {in_receiver:3d} particles vs "
        f"{scene.receiver.fill_level:4.2f} level (goal {GOAL_FILL})  "
        f"spilled: {len(fill.names) - in_source - in_receiver:3d}  "
        f"inflow {measured_rate:+5.2f} measured vs "
        f"{_predicted_inflow(scene):+5.2f} predicted /s"
        + (f"  drain scale {calibration.value:4.2f}" if calibration else "")
    )


def _predicted_inflow(scene: TransferScene) -> float:
    """
    How fast the drain model believes the receiver is filling.

    :param scene: The scene holding the coupling.
    :return: The predicted share of the receiver's cavity per second.
    """
    inflow = scene.receiver.fill_connection.inflow_equation
    if inflow is None:
        return 0.0
    return float(inflow.symbolic_velocity(scene.receiver.fill_connection).evaluate()[0])


def _summarize(scene: TransferScene, fill: ParticleFill) -> None:
    """
    Say plainly where the contents ended up, against what the controller believed.

    :param scene: The scene the pour ran in.
    :param fill: The contents that were poured.
    """
    in_source = fill.count_inside(scene.source.root)
    in_receiver = fill.count_inside(scene.receiver.root)
    print(
        f"the controller left its source at {scene.source.fill_level:.2f} and its "
        f"receiver at {scene.receiver.fill_level:.2f}, tilting the cup to "
        f"{math.degrees(_cup_tilt(scene)):.0f} degrees; of {len(fill.names)} grains, "
        f"{in_source} never left the source, {in_receiver} reached the receiver and "
        f"{len(fill.names) - in_source - in_receiver} went elsewhere"
    )


def _cup_tilt(scene: TransferScene) -> float:
    """
    How far the source cup is turned from upright, in radians.

    :param scene: The scene holding the cup.
    :return: The angle between the cup's own up and the world's.
    """
    cup_up = scene.world.compute_forward_kinematics_np(
        scene.world.root, scene.source.root
    )[:3, 2]
    return math.acos(min(1.0, max(-1.0, float(cup_up[2]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="run without the MuJoCo viewer"
    )
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="pour with no parameter of the contents estimated, for comparison",
    )
    arguments = parser.parse_args()
    run(headless=arguments.headless, nominal=arguments.nominal)
