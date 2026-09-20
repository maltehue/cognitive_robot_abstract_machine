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
from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Callable, Optional

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
from semantic_digital_twin.physics.particles import (
    HollowCylinder,
    MeasuredFillLevel,
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

INITIAL_FILL = 1.0
"""
How deep the source cup is packed, as a share of its cavity.

A loose packing settles to about half the depth it was packed to, so a cup packed to its
rim holds its contents at about half its height once they have come to rest.
"""

GOAL_FILL = 0.3
"""
The receiver's fill level the motion is commanded to reach.
"""

FILL_TOLERANCE = 0.05
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

MINIMUM_RIM_CLEARANCE = 0.2
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

SETTLE_TIME = timedelta(milliseconds=500)
"""
How long the contents settle in the cup before the pour starts.
"""

PERCEPTION_FREQUENCY = 10
"""
How often the receiver's fill level is measured from its contents and reported into the
controller's model, in hertz.

Without it the controller reasons about the pour its own drain model predicts. With it
the level it steers by is the one the grains actually produced.
"""

REPORT_EVERY = 200
"""
How many control cycles pass between two lines of the report.
"""

OUTFLOW_RATE_CONSTANT = 1.0
"""
Outflow rate constant of the source's drain.
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


def build_scene() -> TransferScene:
    """
    Build Tracy with both arms parked, a source cup in its left hand and a receiving
    cup on the table, coupled so what leaves one enters the other.

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
        initial_fill=INITIAL_FILL,
        outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
    )
    receiver.initialize_fill_level(world=world, initial_fill=0.0)
    receiver.receive_outflow_from(source=source, world=world)
    return TransferScene(
        world=world,
        robot=robot,
        source=source,
        receiver=receiver,
        tool_frame=tool_frame,
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


def pour_start_tilt(source: PourableContainer) -> float:
    """
    The tilt at which the source's contents reach its lip at its current fill, plus a
    margin, so the drain has flow and gradient from the start.

    :param source: The cup about to pour.
    :return: The tilt, in radians.
    """
    equation = source.fill_equation.ungated()
    dry_height = equation.container_height * (1.0 - source.fill_level)
    return math.atan2(dry_height, equation.lip_offset) + POUR_START_TILT_MARGIN


def build_transfer_motion(scene: TransferScene) -> MotionStatechart:
    """
    Build the motion that pours the source into the receiver.

    :param scene: The scene the motion runs in.
    :return: The statechart, ending when the receiver has reached its goal.
    """
    transfer = FillByTransferTask(
        receiver=scene.receiver,
        goal_value=GOAL_FILL,
        fill_level_tolerance=FILL_TOLERANCE,
        reference_velocity=0.03,
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


def run(headless: bool) -> None:
    """
    Carry the cup upright, fill it, pour it into the receiver under the pouring tasks,
    and report where the contents went.

    :param headless: Whether to run without the MuJoCo viewer.
    """
    scene = build_scene()
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
            count=SOURCE.particle_capacity(PARTICLE_RADIUS, fill_fraction=INITIAL_FILL),
        )
        simulation.step_simulation(SETTLE_TIME)
        settled = fill.filled_height_in(scene.source.root)
        JointState.from_mapping({scene.source.fill_connection: settled}).apply_to(
            scene.world
        )
        print(
            f"{len(fill.names)} particles in the source cup, "
            f"{fill.count_inside(scene.source.root)} of them still in it after "
            f"settling to a depth of {settled:.2f}, which the pouring equation "
            f"starts from"
        )

        perception = MeasuredFillLevel(
            contents=fill,
            container=scene.receiver.root,
            connection=scene.receiver.fill_connection,
            world=scene.world,
        )
        tilt = pour_start_tilt(scene.source)
        print(f"carrying the cup pre-tilted to {tilt:.3f} rad, where its drain starts")
        _run_motion(
            build_carry_motion(scene, tilt=tilt), scene, simulation, "pre-tilt", fill
        )
        _run_motion(
            build_transfer_motion(scene),
            scene,
            simulation,
            "transfer",
            fill=fill,
            perception=perception,
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
    perception: Optional[MeasuredFillLevel] = None,
) -> None:
    """
    Tick one motion to its end in lockstep with the physics.

    :param statechart: The motion to run.
    :param scene: The scene the motion runs in.
    :param simulation: The simulation the controller runs in lockstep with.
    :param name: What to call the motion in the report.
    :param fill: The contents to report on, if there are any yet.
    :param perception: What reports the receiver's fill level into the controller's
        model, if the run closes that loop.
    """
    executor = Executor(
        context=MotionStatechartContext(
            world=scene.world,
            qp_controller_config=QPControllerConfig(
                target_frequency=CONTROL_FREQUENCY,
                prediction_horizon=PREDICTION_HORIZON,
            ),
        ),
        pacer=SteppedSimulationPacer(simulation),
    )
    executor.compile(motion_statechart=statechart)
    if fill is not None:
        executor.tick = _reporting_tick(executor, scene, fill, perception)
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
    perception: Optional[MeasuredFillLevel],
) -> Callable[[], None]:
    """
    Wrap an executor's tick so the receiver is measured and the pour reported as the
    motion runs.

    :param executor: The executor whose ticks are wrapped.
    :param scene: The scene the motion runs in.
    :param fill: The contents being poured.
    :param perception: What reports the receiver's fill level, if the run closes that
        loop.
    :return: The wrapped tick.
    """
    tick = executor.tick
    cycles_between_measurements = max(
        1, round(CONTROL_FREQUENCY / PERCEPTION_FREQUENCY)
    )

    def tick_and_report() -> None:
        tick()
        cycle = int(executor.control_cycles)
        if perception is not None and cycle % cycles_between_measurements == 0:
            perception.report()
        if cycle % REPORT_EVERY == 0:
            _report(cycle, scene, fill)

    return tick_and_report


def _nudge_wrist(scene: TransferScene) -> None:
    """
    Turn the wrist out of the carry pose, so the pour does not start from a wrist that
    can only tilt one way.

    :param scene: The scene whose robot is nudged.
    """
    wrist = scene.world.get_connection_by_name("left_wrist_3_joint")
    JointState.from_mapping({wrist: wrist.position + WRIST_NUDGE}).apply_to(scene.world)


def _report(tick: int, scene: TransferScene, fill: ParticleFill) -> None:
    """
    Print one line comparing the commanded fill levels against the particles.

    :param tick: Which control cycle this is.
    :param scene: The scene being reported on.
    :param fill: The contents being poured.
    """
    in_source = fill.count_inside(scene.source.root)
    in_receiver = fill.count_inside(scene.receiver.root)
    print(
        f"cycle {tick:5d}  tilt {math.degrees(_cup_tilt(scene)):5.1f} deg  "
        f"source: {in_source:3d} particles vs {scene.source.fill_level:4.2f} level  "
        f"receiver: {in_receiver:3d} particles vs "
        f"{scene.receiver.fill_level:4.2f} level (goal {GOAL_FILL})  "
        f"spilled: {len(fill.names) - in_source - in_receiver:3d}"
    )


def _summarize(scene: TransferScene, fill: ParticleFill) -> None:
    """
    Say plainly where the contents ended up, against what the controller believed.

    :param scene: The scene the pour ran in.
    :param fill: The contents that were poured.
    """
    in_source = fill.count_inside(scene.source.root)
    in_receiver = fill.count_inside(scene.receiver.root)
    print(
        f"the controller emptied its source to {scene.source.fill_level:.2f} and "
        f"filled its receiver to {scene.receiver.fill_level:.2f}, tilting the cup to "
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
    run(headless=parser.parse_args().headless)
