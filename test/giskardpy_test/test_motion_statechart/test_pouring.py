"""
Tests of the tasks that pour from a held container or fill one container from another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from cramph.composites import Parallel
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
    PouringTask,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain
from semantic_digital_twin.world_description.liquid_connection import LiquidConnection
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from ...statechart_json_round_trip import StatechartSentThroughJSON
from ..motion_control_context import (
    add_to_statechart_with_motion_control,
    create_context_with_motion_control,
)
from .single_cup_world import PourableContainer, build_single_cup_world

# %% shared scene and controller settings


@dataclass(kw_only=True)
class PouringSetup:
    """
    The scene and controller settings the pouring tests share.
    """

    cup_mesh_path: str = field(
        default_factory=lambda: str(
            Path(files("semantic_digital_twin")).parent.parent
            / "resources"
            / "stl"
            / "jeroen_cup.stl"
        )
    )
    """
    The mesh of the Jeroen cup the robot pours with.
    """

    cup_mesh_scale: Scale = field(default_factory=lambda: Scale(1, 1, 1))
    """
    Scale of the Jeroen cup mesh.
    """

    table_surface_height: float = 0.9
    """
    Height of the table surface in the world, in metres.
    """

    target_frequency: int = 80
    """
    Control frequency the QP runs at, in Hz.
    """

    timeout: int = 4000
    """
    Most control cycles a pour may take in simulation.
    """

    carry_height: float = 0.3
    """
    Height of the left tool above the table when a pour starts, in metres.
    """

    carry_side: float = 0.2
    """
    Sideways position of the left tool in the world when a pour starts, in metres.
    """

    pre_tilt: float = 0.1
    """
    Tilt the held cup starts a pour at, in radians.
    """

    bottom_grasp: float = 0.0
    """
    Offset of the held cup's base along the left tool's y axis, in metres.
    """

    def controller_config(self) -> QPControllerConfig:
        """
        :return: A controller configuration running the QP at
            :attr:`target_frequency`, its horizon derived from the braking time.
        """
        return QPControllerConfig(target_frequency=self.target_frequency)

    def executor(self, *, world: World) -> StatechartExecutor:
        """
        :param world: The world the motion runs in.
        :return: An executor ticking a motion statechart with
            :meth:`controller_config`.
        """
        return StatechartExecutor(
            context=StatechartContext(world=world),
            extensions=[MotionControl(qp_controller_config=self.controller_config())],
        )

    def cup_body(self, *, name: str) -> Body:
        """
        :param name: The name of the body.
        :return: A body with the Jeroen cup mesh as geometry.
        """
        mesh = Mesh(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
            filename=self.cup_mesh_path,
            scale=self.cup_mesh_scale,
        )
        return Body.from_shape_collection(
            shape_collection=ShapeCollection([mesh]), name=PrefixedName(name)
        )

    def move_left_tool_upright(self, *, world: World, left_tool_frame: Body) -> None:
        """
        Move Tracy's left tool to the upright carry pose above the table.

        :param world: The world holding Tracy.
        :param left_tool_frame: Tracy's left tool frame.
        """
        upright_pose = Pose.from_xyz_quaternion(
            pos_x=1,
            pos_y=self.carry_side,
            pos_z=self.table_surface_height + self.carry_height,
            quat_z=0.5,
            quat_x=0.5,
            quat_y=0.5,
            quat_w=0.5,
            reference_frame=world.root,
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=world), extensions=[MotionControl()]
        )
        statechart = Statechart(context=executor.context)
        cartesian_task = CartesianPose(
            kinematic_chain=KinematicChain(root=world.root, tip=left_tool_frame),
            goal_pose=upright_pose,
        )
        statechart.add_node(node=cartesian_task)
        statechart.add_node(node=EndMotion.when_true(node=cartesian_task))
        executor.compile(statechart=statechart)
        executor.tick_until_end(timeout=self.timeout)

    def grasp_cup(
        self, *, world: World, left_tool_frame: Body, name: str
    ) -> PourableContainer:
        """
        :param world: The world holding Tracy.
        :param left_tool_frame: Tracy's left tool frame, which holds the cup.
        :param name: The name of the cup.
        :return: A full Jeroen cup fixed to the left tool, held at its bottom.
        """
        cup_body = self.cup_body(name=name)
        with world.modify_world():
            world.add_body(cup_body)
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world=world,
                    parent=left_tool_frame,
                    child=cup_body,
                    name=PrefixedName(f"l_gripper_T_{name}"),
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        roll=-math.pi / 2.0, y=self.bottom_grasp
                    ),
                )
            )
        cup = PourableContainer(name=PrefixedName(name), root=cup_body)
        with world.modify_world():
            world.add_semantic_annotation(cup)
        cup.initialize_fill_level(initial_fill=1.0, outflow_rate_constant=1.0)
        return cup

    def pre_tilt_left_wrist(self, *, world: World) -> None:
        """
        Turn Tracy's left wrist by :attr:`pre_tilt`.

        :param world: The world holding Tracy.
        """
        left_wrist_joint = world.get_connection_by_name("left_wrist_3_joint")
        JointState.from_mapping(
            {left_wrist_joint: left_wrist_joint.position + self.pre_tilt}
        ).apply_to(world)


@pytest.fixture
def pouring_setup() -> PouringSetup:
    """
    The scene and controller settings the pouring tests share.
    """
    return PouringSetup()


@pytest.fixture
def world_with_cup() -> tuple[World, PourableContainer]:
    """
    World containing a single pourable container with a tilt joint, filled to 100%.
    """
    return build_single_cup_world()


@pytest.fixture
def tracy_pouring_world(
    tracy_world_copy: World, pouring_setup: PouringSetup
) -> tuple[World, Tracy]:
    """
    Tracy world with both arms in park position and a Jeroen cup on the table.
    """
    world = tracy_world_copy
    [tracy] = world.get_semantic_annotations_by_type(Tracy)

    tracy.left_arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    tracy.right_arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)

    table_cup_body = pouring_setup.cup_body(name="table_cup")
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=table_cup_body,
                name=PrefixedName("table_T_table_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.5, 0.0, pouring_setup.table_surface_height
                ),
            )
        )

    return world, tracy


@pytest.fixture
def tracy_transfer_world(
    tracy_pouring_world: tuple[World, Tracy], pouring_setup: PouringSetup
) -> tuple[World, PourableContainer, PourableContainer, Body]:
    """
    World with a full source cup in Tracy's left gripper, pre-tilted, and an empty
    receiving cup on the table coupled to it.

    :returns:``(world, source_cup, receiving_cup, left_tool_frame)``
    """
    world, _tracy = tracy_pouring_world
    left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")
    pouring_setup.move_left_tool_upright(world=world, left_tool_frame=left_tool_frame)
    source_cup = pouring_setup.grasp_cup(
        world=world, left_tool_frame=left_tool_frame, name="source_cup"
    )

    receiving_cup_body = pouring_setup.cup_body(name="receiving_cup")
    with world.modify_world():
        world.add_body(receiving_cup_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=receiving_cup_body,
                name=PrefixedName("table_T_receiving_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 0.1, pouring_setup.table_surface_height
                ),
            )
        )
    receiving_cup = PourableContainer(
        name=PrefixedName("receiving_cup"), root=receiving_cup_body
    )
    with world.modify_world():
        world.add_semantic_annotation(receiving_cup)
    receiving_cup.initialize_fill_level(initial_fill=0.0, outflow_rate_constant=1.0)

    receiving_cup.receive_outflow_from(source=source_cup)
    pouring_setup.pre_tilt_left_wrist(world=world)

    return world, source_cup, receiving_cup, left_tool_frame


# %% the cup-to-cup transfer motion


@dataclass(kw_only=True)
class TransferMotion:
    """
    A built cup-to-cup transfer motion together with the entities its tasks act on, so
    the transfer tests share one construction path.
    """

    world: World
    """
    The world the motion is built against.
    """

    source_cup: PourableContainer
    """
    The cup being poured from.
    """

    receiving_cup: PourableContainer
    """
    The cup being poured into.
    """

    transfer_task: FillByTransferTask
    """
    The fill-driving task, whose tick hook the clearance recorder wraps.
    """

    aiming_task: KeepProjectileInReceiver
    """
    The task keeping the stream landing in the receiver, which the fill task waits for.
    """

    clearance_task: KeepSourceRimAboveReceiverRim
    """
    The task holding the source lip above the receiver rim.
    """

    executor: StatechartExecutor
    """
    The executor the motion is built for.
    """

    statechart: Statechart
    """
    The statechart executing the transfer.
    """

    setup: PouringSetup
    """
    The settings the motion was built with.
    """

    @classmethod
    def build(
        cls,
        *,
        transfer_world: tuple[World, PourableContainer, PourableContainer, Body],
        setup: PouringSetup,
        minimum_clearance: float = 0.07,
        clearance_band: float = 0.05,
        aiming_weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        aiming_cruise_velocity: float = 0.2,
        fill_level_tolerance: float = 0.05,
    ) -> TransferMotion:
        """
        :param transfer_world: The transfer world fixture value.
        :param setup: The settings to build the motion with.
        :param minimum_clearance: Rim-to-rim clearance floor of the clearance task.
        :param clearance_band: Width of the clearance band above the floor.
        :param aiming_weight: Weight of the pour-aiming task.
        :param aiming_cruise_velocity: Cruise velocity of the pour-aiming task.
        :param fill_level_tolerance: Tolerance around the fill goal.
        :return: The transfer motion, whose fill task starts once the pour is aimed.
        """
        world, source_cup, receiving_cup, left_tool_frame = transfer_world
        transfer_task = FillByTransferTask(
            receiver=receiving_cup,
            goal_value=0.7,
            fill_level_tolerance=fill_level_tolerance,
            cruise_velocity=0.03,
        )
        aiming_task = KeepProjectileInReceiver(
            receiver=receiving_cup,
            source=source_cup,
            weight=aiming_weight,
            cruise_velocity=aiming_cruise_velocity,
        )
        clearance_task = KeepSourceRimAboveReceiverRim(
            receiver=receiving_cup,
            source=source_cup,
            minimum_clearance=minimum_clearance,
            clearance_band=clearance_band,
        )
        keep_plane = AlignPlanes(
            kinematic_chain=KinematicChain(root=world.root, tip=left_tool_frame),
            reference_normal=Vector3.X(reference_frame=world.root),
            tip_normal=Vector3.Z(reference_frame=left_tool_frame),
        )
        motion = Parallel(
            nodes=[transfer_task, aiming_task, clearance_task, keep_plane]
        )
        executor = setup.executor(world=world)
        statechart = Statechart(context=executor.context)
        statechart.add_node(node=motion)
        statechart.add_node(node=EndMotion.when_true(node=motion))
        transfer_task.start_condition = aiming_task.observes_true
        return cls(
            world=world,
            source_cup=source_cup,
            receiving_cup=receiving_cup,
            transfer_task=transfer_task,
            aiming_task=aiming_task,
            clearance_task=clearance_task,
            executor=executor,
            statechart=statechart,
            setup=setup,
        )

    def record_rim_clearance(self) -> list[float]:
        """
        Start recording the source-lip-above-receiver-rim clearance on every transfer
        tick.

        :return: The list the clearance samples are appended to as the motion runs.
        """
        source_lip = self.source_cup.liquid_exit_point()
        receiver_rim = self.receiving_cup.opening_point()
        clearance_history: list[float] = []
        original_on_tick = self.transfer_task.on_tick

        def recording_on_tick() -> ObservationStateValue | None:
            """
            Record the clearance, then tick the task.
            """
            clearance_history.append(
                float(source_lip.z.evaluate()[0] - receiver_rim.z.evaluate()[0])
            )
            return original_on_tick()

        self.transfer_task.on_tick = recording_on_tick
        return clearance_history

    def execute(self) -> None:
        """
        Run the transfer to completion in simulation.
        """
        self.executor.compile(statechart=self.statechart)
        self.executor.tick_until_end(timeout=self.setup.timeout)


# %% fill-level perception


@dataclass(kw_only=True)
class NoisyFillPerception:
    """
    Replaces the controller's belief of a receiver's fill level with a noisy measurement
    of the true fill, at a rate below the control rate.
    """

    world: World
    """
    The world whose fill state is corrected.
    """

    fill_connection: LiquidConnection
    """
    The receiver's fill connection the belief lives on.
    """

    sigma: float
    """
    Standard deviation of the additive Gaussian measurement noise.
    """

    perception_rate: float
    """
    Rate the measurements arrive at, in Hz.
    """

    random_generator: np.random.Generator
    """
    Random number generator of the noise.
    """

    true_fill: float = field(init=False)
    """
    The receiver's true fill, advanced by the inflow the world integrates; the inflow
    does not depend on the receiver's own fill.
    """

    def run(self, *, executor: StatechartExecutor, setup: PouringSetup) -> None:
        """
        Tick `executor` until its statechart ends, correcting the belief on every
        perception tick.

        :param executor: The executor driving the compiled motion statechart.
        :param setup: The settings giving the control rate and the timeout.
        """
        ticks_per_perception = max(
            1, round(setup.target_frequency / self.perception_rate)
        )
        self.true_fill = float(self.fill_connection.position)
        for tick_index in range(setup.timeout):
            belief_before_tick = float(self.fill_connection.position)
            executor.tick()
            self.true_fill += float(self.fill_connection.position) - belief_before_tick
            if tick_index % ticks_per_perception == 0:
                measurement = self.true_fill + self.random_generator.normal(
                    0.0, self.sigma
                )
                JointState.from_mapping(
                    {self.fill_connection: float(np.clip(measurement, 0.0, 1.0))}
                ).apply_to(self.world)
            if executor.statechart.is_ended():
                break
        executor.finish_run()


# %% pouring from a held cup


class TestPouringTask:
    """
    Pouring from a held container down to a fill-level goal.
    """

    def test_pouring_task_achieves_goal(
        self, *, world_with_cup, pouring_setup: PouringSetup
    ) -> None:
        """
        PouringTask tilts the cup and brings its fill level to the goal.
        """
        world, cup = world_with_cup
        executor = pouring_setup.executor(world=world)
        motion_statechart = Statechart(context=executor.context)
        pouring_task = PouringTask(
            container=cup,
            goal_value=0.6,
            fill_level_tolerance=0.05,
            cruise_velocity=0.05,
        )
        motion_statechart.add_node(node=pouring_task)
        motion_statechart.add_node(node=EndMotion.when_true(node=pouring_task))
        executor.compile(statechart=motion_statechart)

        executor.tick_until_end(timeout=pouring_setup.timeout)

        assert pouring_task.observation_state == ObservationStateValue.TRUE
        assert cup.fill_level == pytest.approx(
            pouring_task.goal_value, abs=pouring_task.fill_level_tolerance
        )
        assert cup.root.parent_connection.position > 0.1
        assert cup.fill_connection.symbolic_fill_velocity().evaluate()[
            0
        ] == pytest.approx(0.0, abs=1e-2)

    def test_proactive_tilt_back(
        self, *, world_with_cup, pouring_setup: PouringSetup
    ) -> None:
        """The cup starts tilting back while the fill level is still above
        ``goal_value + 2 * fill_level_tolerance``, before it reaches the goal."""
        world, cup = world_with_cup
        executor = pouring_setup.executor(world=world)
        motion_statechart = Statechart(context=executor.context)
        pouring_task = PouringTask(
            container=cup,
            goal_value=0.6,
            fill_level_tolerance=0.05,
            cruise_velocity=0.05,
        )
        motion_statechart.add_node(node=pouring_task)
        motion_statechart.add_node(node=EndMotion.when_true(node=pouring_task))

        tilt_history: list[float] = []
        fill_history: list[float] = []
        original_on_tick = pouring_task.on_tick

        def recording_on_tick() -> ObservationStateValue | None:
            """
            Record tilt and fill, then tick the task.
            """
            tilt_history.append(float(cup.root.parent_connection.position))
            fill_history.append(float(cup.fill_level))
            return original_on_tick()

        pouring_task.on_tick = recording_on_tick

        executor.compile(statechart=motion_statechart)
        executor.tick_until_end(timeout=pouring_setup.timeout)

        assert pouring_task.observation_state == ObservationStateValue.TRUE

        tilt_near_goal_start: float | None = None
        max_tilt_before_threshold = 0.0
        threshold = pouring_task.goal_value + 2 * pouring_task.fill_level_tolerance
        for tilt, fill in zip(tilt_history, fill_history):
            if fill > threshold:
                max_tilt_before_threshold = max(max_tilt_before_threshold, tilt)
            elif tilt_near_goal_start is None:
                tilt_near_goal_start = tilt
                break

        assert tilt_near_goal_start is not None, "fill level never approached goal"
        assert tilt_near_goal_start < max_tilt_before_threshold, (
            f"Expected tilt to be decreasing when fill first reached goal region "
            f"(tilt={tilt_near_goal_start:.4f} should be < "
            f"max tilt before threshold={max_tilt_before_threshold:.4f})"
        )


# %% filling one cup from another


class TestCupToCupTransfer:
    """
    A fill-level goal on a receiving cup on the table makes the optimizer tilt the
    source cup in Tracy's gripper, the only one it controls, until the receiver reaches
    the goal.
    """

    def test_the_receiver_is_filled_to_the_goal(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """
        The receiver reaches the goal, the source loses what the receiver gains, once
        aimed the stream stays in the receiver, and the lip stays above the rim.
        """
        transfer = TransferMotion.build(
            transfer_world=tracy_transfer_world, setup=pouring_setup
        )
        source_cup, receiving_cup = transfer.source_cup, transfer.receiving_cup
        gate = receiving_cup.transfer_gate(source=source_cup)
        clearance_history = transfer.record_rim_clearance()
        gate_history: list[float] = []
        recording_on_tick = transfer.transfer_task.on_tick

        def recording_gate_on_tick() -> ObservationStateValue | None:
            """
            Record the transfer gate, then tick the task.
            """
            gate_history.append(float(gate.evaluate()[0]))
            return recording_on_tick()

        transfer.transfer_task.on_tick = recording_gate_on_tick

        transfer.execute()

        transfer_task = transfer.transfer_task
        tolerance = transfer_task.fill_level_tolerance
        assert transfer_task.observation_state == ObservationStateValue.TRUE
        assert receiving_cup.fill_level == pytest.approx(
            transfer_task.goal_value, abs=tolerance
        )
        assert 1.0 - source_cup.fill_level == pytest.approx(
            receiving_cup.fill_level, abs=tolerance
        )
        assert min(gate_history) > 0.5, "the stream left the receiver once aimed"
        assert min(clearance_history) > 0.0, "the lip dropped to the receiver's rim"


class TestRimClearanceDuringTransfer:
    """
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveRec eiverRim`
    keeps the pouring cup's lip above the receiving cup's rim throughout the transfer,
    and the motion statechart stays serializable.
    """

    def test_transfer_motion_is_json_serializable(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """
        The transfer motion statechart survives a JSON round trip, as it must to be sent
        to a Giskard server.
        """
        transfer = TransferMotion.build(
            transfer_world=tracy_transfer_world, setup=pouring_setup
        )

        round_trip = StatechartSentThroughJSON.received_in(
            transfer.statechart,
            create_context_with_motion_control(
                transfer.world, pouring_setup.controller_config()
            ),
        )

        assert [node.name for node in round_trip.received.nodes] == [
            node.name for node in transfer.statechart.nodes
        ]


class TestClearanceBandStaysAboveTheRim:
    """The clearance band of
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveReceiverRim`
    is enforced softly, so a band reaching down to the rim is rejected."""

    def test_rims_do_not_cross_when_aiming_outweighs_the_clearance(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """
        With the pour-aiming task at maximum weight the pouring lip still never reaches
        the receiver rim.
        """
        transfer = TransferMotion.build(
            transfer_world=tracy_transfer_world,
            setup=pouring_setup,
            aiming_weight=DefaultWeights.WEIGHT_MAXIMUM,
            aiming_cruise_velocity=0.1,
        )
        clearance_history = transfer.record_rim_clearance()

        transfer.execute()

        assert clearance_history, "transfer never ticked"
        assert min(clearance_history) > 0.0, (
            "the pouring lip reached the receiver rim: minimum clearance was "
            f"{min(clearance_history):.4f} m"
        )


class TestFillRowLeavesPlacementToTheOtherTasks:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.FillByTransferTask`
    predicts the fill through the transfer gate but does not steer by it:
    placing the source over the receiver is left to the aim and the
    clearance."""

    def test_the_fill_row_does_not_move_the_source_towards_the_receiver(
        self, *, tracy_transfer_world
    ) -> None:
        """
        The fill velocity has no gradient along the receiver's own degrees of freedom,
        while the tilt still drives it.
        """
        world, _source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        transfer_task = FillByTransferTask(
            receiver=receiving_cup, goal_value=0.7, fill_level_tolerance=0.05
        )
        receiver_height = receiving_cup.root.parent_connection.z.variables.position
        receiver_offset = receiving_cup.root.parent_connection.x.variables.position
        tilt_joint = world.get_connection_by_name(
            "left_wrist_3_joint"
        ).dof.variables.position
        add_to_statechart_with_motion_control(transfer_task, world)

        transfer_task.build()

        velocity = transfer_task.fill_velocity_expression
        assert velocity.jacobian(variables=[receiver_height])[0, 0].evaluate()[0] == 0.0
        assert velocity.jacobian(variables=[receiver_offset])[0, 0].evaluate()[0] == 0.0
        assert velocity.jacobian(variables=[tilt_joint])[0, 0].evaluate()[0] != 0.0


# %% filling under perception noise


class TestPerceptionCorrectedTransfer:
    """
    :class:`~giskardpy.motion_statechart.tasks.pouring.FillByTransferTask` converges
    when the receiver's fill belief is replaced by noisy measurements of the true fill
    at a rate below the control rate.
    """

    @pytest.mark.parametrize(
        "sigma,perception_rate",
        [(0.02, 10)],
        ids=["sigma=0.02_10Hz"],
    )
    def test_convergence_under_perception_noise(
        self,
        *,
        tracy_transfer_world,
        pouring_setup: PouringSetup,
        sigma: float,
        perception_rate: float,
    ) -> None:
        """
        The transfer ends with the receiver's true fill at the goal, within the
        tolerance widened by two noise deviations, as the task ends on the perceived
        fill.

        :param sigma: Standard deviation of the measurement noise.
        :param perception_rate: Rate of the measurements, in Hz.
        """
        transfer = TransferMotion.build(
            transfer_world=tracy_transfer_world,
            setup=pouring_setup,
            fill_level_tolerance=0.01,
        )
        transfer.executor.compile(statechart=transfer.statechart)
        perception = NoisyFillPerception(
            world=transfer.world,
            fill_connection=transfer.receiving_cup.fill_connection,
            sigma=sigma,
            perception_rate=perception_rate,
            random_generator=np.random.default_rng(seed=42),
        )

        perception.run(executor=transfer.executor, setup=pouring_setup)

        transfer_task = transfer.transfer_task
        assert transfer_task.observation_state == ObservationStateValue.TRUE, (
            f"Transfer task did not converge "
            f"(sigma={sigma}, perception_rate={perception_rate} Hz)"
        )
        source_loss = 1.0 - float(transfer.source_cup.fill_level)
        assert source_loss > transfer_task.fill_level_tolerance, (
            "source cup never poured"
        )
        assert perception.true_fill == pytest.approx(
            transfer_task.goal_value,
            abs=transfer_task.fill_level_tolerance + 2.0 * sigma,
        ), (
            f"true receiver fill {perception.true_fill:.3f} outside goal band around "
            f"{transfer_task.goal_value} (sigma={sigma}, perception_rate={perception_rate} Hz)"
        )
