"""Tests of the tasks that pour from a held container or fill one container
from another."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from cramph.composites import Parallel, Sequence
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion, MotionNodeArtifacts
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.exceptions import (
    MissingExitSpeedError,
    MissingInflowEquationError,
    MissingOutflowEquationError,
    NonPositiveClearanceError,
)
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    FillDebugValue,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
    PouringTask,
    StreamDebugValue,
    TerminalFillConstraintTask,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.terminal_state_prediction_strategy import (
    TerminalStatePredictionConstraint,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Vector3,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.equations.pouring_equations import InflowEquation
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    LiquidConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from ...statechart_json_round_trip import StatechartSentThroughJSON
from ..motion_control_context import (
    add_to_statechart_with_motion_control,
    create_context_with_motion_control,
)
from .debug_expression_helpers import debug_expression_by_name
from .single_cup_world import PourableContainer, build_single_cup_world

# %% shared scene and controller settings


@dataclass(kw_only=True)
class PouringSetup:
    """The scene and controller settings the pouring tests share."""

    cup_mesh_path: str = field(
        default_factory=lambda: str(
            Path(files("semantic_digital_twin")).parent.parent
            / "resources"
            / "stl"
            / "jeroen_cup.stl"
        )
    )
    """The mesh of the Jeroen cup the robot pours with."""

    cup_mesh_scale: Scale = field(default_factory=lambda: Scale(1, 1, 1))
    """Scale of the Jeroen cup mesh."""

    table_surface_height: float = 0.9
    """Height of the table surface in the world, in metres."""

    target_frequency: int = 80
    """Control frequency the QP runs at, in Hz."""

    timeout: int = 4000
    """Most control cycles a pour may take in simulation."""

    carry_height: float = 0.3
    """Height of the left tool above the table when a pour starts, in
    metres."""

    carry_side: float = 0.2
    """Sideways position of the left tool in the world when a pour starts, in
    metres."""

    pre_tilt: float = 0.1
    """Tilt the held cup starts a pour at, in radians."""

    bottom_grasp: float = 0.0
    """Offset of the held cup's base along the left tool's y axis, in
    metres."""

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

    def put_into_statechart(
        self, *, task: TerminalFillConstraintTask, world: World
    ) -> None:
        """Put `task` into a statechart of its own with
        :meth:`controller_config`, so it can be built outside of an executor.

        :param task: The task to build later.
        :param world: The world the task acts in.
        """
        add_to_statechart_with_motion_control(task, world, self.controller_config())

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
        """Move Tracy's left tool to the upright carry pose above the table.

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
        cup.initialize_fill_level(
            world=world, initial_fill=1.0, outflow_rate_constant=1.0
        )
        return cup

    def pre_tilt_left_wrist(self, *, world: World) -> None:
        """Turn Tracy's left wrist by :attr:`pre_tilt`.

        :param world: The world holding Tracy.
        """
        left_wrist_joint = world.get_connection_by_name("left_wrist_3_joint")
        JointState.from_mapping(
            {left_wrist_joint: left_wrist_joint.position + self.pre_tilt}
        ).apply_to(world)


@pytest.fixture
def pouring_setup() -> PouringSetup:
    """The scene and controller settings the pouring tests share."""
    return PouringSetup()


@pytest.fixture
def world_with_cup() -> tuple[World, PourableContainer]:
    """World containing a single pourable container with a tilt joint, filled
    to 100%."""
    return build_single_cup_world()


@pytest.fixture
def tracy_pouring_world(
    tracy_world_copy: World, pouring_setup: PouringSetup
) -> tuple[World, Tracy]:
    """Tracy world with both arms in park position and a Jeroen cup on the
    table."""
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
    """World with a full source cup in Tracy's left gripper, pre-tilted, and an
    empty receiving cup on the table coupled to it.

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
    receiving_cup.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )

    receiving_cup.receive_outflow_from(source=source_cup, world=world)
    pouring_setup.pre_tilt_left_wrist(world=world)

    return world, source_cup, receiving_cup, left_tool_frame


# %% the cup-to-cup transfer motion


@dataclass(kw_only=True)
class TransferMotion:
    """A built cup-to-cup transfer motion together with the entities its tasks
    act on, so the transfer tests share one construction path."""

    world: World
    """The world the motion is built against."""

    source_cup: PourableContainer
    """The cup being poured from."""

    receiving_cup: PourableContainer
    """The cup being poured into."""

    transfer_task: FillByTransferTask
    """The fill-driving task, whose tick hook the clearance recorder wraps."""

    aiming_task: KeepProjectileInReceiver
    """The task keeping the stream landing in the receiver, which the fill task
    waits for."""

    clearance_task: KeepSourceRimAboveReceiverRim
    """The task holding the source lip above the receiver rim."""

    executor: StatechartExecutor
    """The executor the motion is built for."""

    statechart: Statechart
    """The statechart executing the transfer."""

    setup: PouringSetup
    """The settings the motion was built with."""

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
        """Start recording the source-lip-above-receiver-rim clearance on every
        transfer tick.

        :return: The list the clearance samples are appended to as the
            motion runs.
        """
        source_lip = self.source_cup.liquid_exit_point(world=self.world)
        receiver_rim = (
            self.world.compose_forward_kinematics_expression(
                self.world.root, self.receiving_cup.root
            )
            @ self.receiving_cup.rim_point()
        )
        clearance_history: list[float] = []
        original_on_tick = self.transfer_task.on_tick

        def recording_on_tick() -> ObservationStateValue | None:
            """Record the clearance, then tick the task."""
            clearance_history.append(
                float(source_lip.z.evaluate()[0] - receiver_rim.z.evaluate()[0])
            )
            return original_on_tick()

        self.transfer_task.on_tick = recording_on_tick
        return clearance_history

    def execute(self) -> None:
        """Run the transfer to completion in simulation."""
        self.executor.compile(statechart=self.statechart)
        self.executor.tick_until_end(timeout=self.setup.timeout)


# %% fill-level perception


@dataclass(kw_only=True)
class NoisyFillPerception:
    """Replaces the controller's belief of a receiver's fill level with a noisy
    measurement of the true fill, at a rate below the control rate."""

    world: World
    """The world whose fill state is corrected."""

    fill_connection: LiquidConnection
    """The receiver's fill connection the belief lives on."""

    sigma: float
    """Standard deviation of the additive Gaussian measurement noise."""

    perception_hz: float
    """Rate the measurements arrive at, in Hz."""

    rng: np.random.Generator
    """Random number generator of the noise."""

    true_fill: float = field(init=False)
    """The receiver's true fill, advanced by the inflow the world integrates;
    the inflow does not depend on the receiver's own fill."""

    def run(self, *, executor: StatechartExecutor, setup: PouringSetup) -> None:
        """Tick `executor` until its statechart ends, correcting the belief on
        every perception tick.

        :param executor: The executor driving the compiled motion
            statechart.
        :param setup: The settings giving the control rate and the
            timeout.
        """
        ticks_per_perception = max(
            1, round(setup.target_frequency / self.perception_hz)
        )
        self.true_fill = float(self.fill_connection.position)
        for tick_index in range(setup.timeout):
            belief_before_tick = float(self.fill_connection.position)
            executor.tick()
            self.true_fill += float(self.fill_connection.position) - belief_before_tick
            if tick_index % ticks_per_perception == 0:
                measurement = self.true_fill + self.rng.normal(0.0, self.sigma)
                JointState.from_mapping(
                    {self.fill_connection: float(np.clip(measurement, 0.0, 1.0))}
                ).apply_to(self.world)
            if executor.statechart.is_ended():
                break
        executor.finish_run()


def _recording_fill_when_reached(
    *,
    task: PouringTask,
    container: PourableContainer,
    fill_when_reached: dict[str, float],
) -> Callable[[], ObservationStateValue | None]:
    """
    :param task: The pour whose tick is wrapped.
    :param container: The container the pour drains.
    :param fill_when_reached: Where the container's fill level is recorded under the
        task's name on the first tick the task observes its goal reached.
    :return: The task's tick, recording the fill level on reaching the goal.
    """
    original_on_tick = task.on_tick

    def recording_on_tick() -> ObservationStateValue | None:
        """Tick the task and record the fill level once it reaches its goal."""
        observation = original_on_tick()
        if observation == ObservationStateValue.TRUE:
            fill_when_reached.setdefault(task.name, container.fill_level)
        return observation

    return recording_on_tick


# %% pouring from a held cup


class TestPouringTask:
    """Pouring from a held container down to a fill-level goal."""

    def test_pouring_task_achieves_goal(
        self, *, world_with_cup, pouring_setup: PouringSetup
    ) -> None:
        """PouringTask tilts the cup and brings its fill level to the goal."""
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

    def test_two_consecutive_pours_reach_their_goals(
        self, *, world_with_cup, pouring_setup: PouringSetup
    ) -> None:
        """Two pours from the same cup, one after the other in one statechart,
        each reach their own goal."""
        world, cup = world_with_cup
        executor = pouring_setup.executor(world=world)
        motion_statechart = Statechart(context=executor.context)
        first_pour = PouringTask(
            container=cup, goal_value=0.8, fill_level_tolerance=0.05, name="first_pour"
        )
        second_pour = PouringTask(
            container=cup, goal_value=0.6, fill_level_tolerance=0.05, name="second_pour"
        )
        fill_when_reached: dict[str, float] = {}
        for pour in (first_pour, second_pour):
            pour.on_tick = _recording_fill_when_reached(
                task=pour, container=cup, fill_when_reached=fill_when_reached
            )
        pours = Sequence(nodes=[first_pour, second_pour])
        motion_statechart.add_node(node=pours)
        motion_statechart.add_node(node=EndMotion.when_true(node=pours))
        executor.compile(statechart=motion_statechart)

        executor.tick_until_end(timeout=2 * pouring_setup.timeout)

        for pour in (first_pour, second_pour):
            assert fill_when_reached[pour.name] == pytest.approx(
                pour.goal_value, abs=pour.fill_level_tolerance
            )
        assert cup.fill_level == pytest.approx(
            second_pour.goal_value, abs=second_pour.fill_level_tolerance
        )

    def test_build_hands_the_prediction_duration_to_the_constraint(
        self, *, world_with_cup, pouring_setup: PouringSetup
    ) -> None:
        """The QP row carries the task's own prediction window rather than the
        control horizon."""
        world, cup = world_with_cup
        pouring_task = PouringTask(
            container=cup,
            goal_value=0.6,
            fill_level_tolerance=0.05,
            prediction_duration=1.5,
        )
        pouring_setup.put_into_statechart(task=pouring_task, world=world)

        artifacts = pouring_task.build()

        [constraint] = artifacts.constraints.equality_constraints
        assert isinstance(constraint, TerminalStatePredictionConstraint)
        assert constraint.prediction_duration == pouring_task.prediction_duration

    def test_build_hands_the_stopping_velocity_to_the_constraint(
        self, *, world_with_cup, pouring_setup: PouringSetup
    ) -> None:
        """How fast the task may take back an overshoot is the task's to
        decide, apart from how fast it pours."""
        world, cup = world_with_cup
        pouring_task = PouringTask(
            container=cup,
            goal_value=0.6,
            fill_level_tolerance=0.05,
            cruise_velocity=0.03,
            stopping_velocity=0.09,
        )
        pouring_setup.put_into_statechart(task=pouring_task, world=world)

        artifacts = pouring_task.build()

        [constraint] = artifacts.constraints.equality_constraints
        assert constraint.stopping_velocity == pouring_task.stopping_velocity

    def test_the_fill_rate_is_the_drain_the_world_integrates(
        self, *, world_with_cup
    ) -> None:
        """The task predicts with the drain its container's fill connection
        integrates, at the container's own tilt."""
        world, cup = world_with_cup
        JointState.from_mapping({cup.root.parent_connection: 1.0}).apply_to(world)
        pouring_task = PouringTask(
            container=cup, goal_value=0.6, fill_level_tolerance=0.05
        )
        add_to_statechart_with_motion_control(pouring_task, world)

        pouring_task.build()

        assert pouring_task.fill_connection is cup.fill_connection
        assert pouring_task.fill_velocity_expression.evaluate()[0] == pytest.approx(
            cup.fill_connection.symbolic_fill_velocity().evaluate()[0]
        )

    def test_build_rejects_a_container_without_fill_level(
        self, *, world_with_cup
    ) -> None:
        """A container that was never given a fill level has nothing to
        pour."""
        world, _cup = world_with_cup
        empty_container = PourableContainer(
            name=PrefixedName("empty_container"),
            root=Body(name=PrefixedName("empty_container_body")),
        )
        pouring_task = PouringTask(
            container=empty_container, goal_value=0.6, fill_level_tolerance=0.05
        )
        add_to_statechart_with_motion_control(pouring_task, world)

        with pytest.raises(MissingOutflowEquationError) as error_info:
            pouring_task.build()

        assert error_info.value.node is pouring_task

    def test_on_tick_does_not_report_true_while_still_flowing(
        self, *, world_with_cup
    ) -> None:
        """A fill level inside the goal band does not count as done while
        liquid is still flowing."""
        world, cup = world_with_cup
        pouring_task = PouringTask(
            container=cup, goal_value=0.6, fill_level_tolerance=0.05
        )
        add_to_statechart_with_motion_control(pouring_task, world)
        pouring_task.build()

        JointState.from_mapping(
            {
                cup.root.parent_connection: 1.3,
                cup.fill_connection: pouring_task.goal_value,
            }
        ).apply_to(world)
        fill_rate = cup.fill_connection.symbolic_fill_velocity().evaluate()[0]
        assert abs(fill_rate) > pouring_task.fill_rate_tolerance

        assert pouring_task.on_tick() is None

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
            """Record tilt and fill, then tick the task."""
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


class TestPouringFromAGripper:
    """Pouring from a cup a robot's gripper holds."""

    def test_pouring_from_a_pr2_gripper(
        self, *, pr2_world_copy: World, pouring_setup: PouringSetup
    ) -> None:
        """PouringTask brings a cup held by the PR2 to its fill goal."""
        world = pr2_world_copy
        gripper_frame = world.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        cup_height = 0.12
        cup_width = 0.08

        with world.modify_world():
            cup_body = Body(name=PrefixedName("cup"))
            world.add_body(cup_body)
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world=world,
                    parent=gripper_frame,
                    child=cup_body,
                    name=PrefixedName("gripper_T_cup_tilt"),
                )
            )
            cup_shape = Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=cup_height / 2, reference_frame=cup_body
                ),
                scale=Scale(cup_width, cup_width, cup_height),
            )
            cup_body.visual = ShapeCollection(shapes=[cup_shape])
            cup_body.collision = ShapeCollection(shapes=[cup_shape])
            cup_body.collision.reference_frame = cup_body

        cup = PourableContainer(name=PrefixedName("cup"), root=cup_body)
        with world.modify_world():
            world.add_semantic_annotation(cup)
        cup.initialize_fill_level(
            world=world, initial_fill=1.0, outflow_rate_constant=1.0
        )

        executor = pouring_setup.executor(world=world)
        motion_statechart = Statechart(context=executor.context)
        pouring_task = PouringTask(
            container=cup, goal_value=0.6, fill_level_tolerance=0.05
        )
        motion_statechart.add_node(node=pouring_task)
        motion_statechart.add_node(node=EndMotion.when_true(node=pouring_task))
        executor.compile(statechart=motion_statechart)

        executor.tick_until_end(timeout=pouring_setup.timeout)

        assert pouring_task.observation_state == ObservationStateValue.TRUE
        assert cup.fill_level == pytest.approx(
            pouring_task.goal_value, abs=pouring_task.fill_level_tolerance
        )
        assert cup.fill_connection.symbolic_fill_velocity().evaluate()[
            0
        ] == pytest.approx(0.0, abs=1e-2)

    def test_pouring_from_a_tracy_gripper(
        self, *, tracy_pouring_world, pouring_setup: PouringSetup
    ) -> None:
        """PouringTask brings a Jeroen cup held in Tracy's left gripper from
        full to its fill goal while the gripper holds its position."""
        world, _tracy = tracy_pouring_world
        left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")
        pouring_setup.move_left_tool_upright(
            world=world, left_tool_frame=left_tool_frame
        )
        grasped_cup = pouring_setup.grasp_cup(
            world=world, left_tool_frame=left_tool_frame, name="grasped_cup"
        )
        pouring_setup.pre_tilt_left_wrist(world=world)
        assert grasped_cup.fill_level == pytest.approx(1.0)

        executor = pouring_setup.executor(world=world)
        pouring_statechart = Statechart(context=executor.context)
        pouring_task = PouringTask(
            container=grasped_cup, goal_value=0.8, fill_level_tolerance=0.05
        )
        keep_position = CartesianPosition(
            kinematic_chain=KinematicChain(root=world.root, tip=left_tool_frame),
            goal_point=Point3(reference_frame=left_tool_frame),
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        )
        motion = Parallel(nodes=[pouring_task, keep_position])
        pouring_statechart.add_node(node=motion)
        pouring_statechart.add_node(node=EndMotion.when_true(node=motion))
        executor.compile(statechart=pouring_statechart)
        executor.tick_until_end(timeout=pouring_setup.timeout)

        assert pouring_task.observation_state == ObservationStateValue.TRUE
        assert grasped_cup.fill_level == pytest.approx(
            pouring_task.goal_value, abs=pouring_task.fill_level_tolerance
        )
        assert grasped_cup.fill_connection.symbolic_fill_velocity().evaluate()[
            0
        ] == pytest.approx(0.0, abs=1e-2)


# %% filling one cup from another


class TestCupToCupTransfer:
    """A fill-level goal on a receiving cup on the table makes the optimizer
    tilt the source cup in Tracy's gripper, the only one it controls, until the
    receiver reaches the goal."""

    def test_the_receiver_is_filled_to_the_goal(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """The receiver reaches the goal, the source loses what the receiver
        gains, and once aimed the stream stays in the receiver."""
        world, source_cup, receiving_cup, left_tool_frame = tracy_transfer_world
        assert receiving_cup.fill_level == pytest.approx(0.0)
        assert source_cup.fill_level == pytest.approx(1.0)
        source_fill_before = source_cup.fill_level

        transfer_task = FillByTransferTask(
            receiver=receiving_cup,
            goal_value=0.7,
            fill_level_tolerance=0.05,
            cruise_velocity=0.03,
        )
        aiming_task = KeepProjectileInReceiver(
            receiver=receiving_cup, source=source_cup
        )
        # A narrow rim-to-rim band keeps the source at a stable height while the aim
        # repositions it, and keeps the lip from sinking towards the receiver's rim.
        clearance_task = KeepSourceRimAboveReceiverRim(
            receiver=receiving_cup,
            source=source_cup,
            minimum_clearance=0.2,
            clearance_band=0.02,
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        )
        keep_plane = AlignPlanes(
            kinematic_chain=KinematicChain(root=world.root, tip=left_tool_frame),
            reference_normal=Vector3.X(reference_frame=world.root),
            tip_normal=Vector3.Z(reference_frame=left_tool_frame),
        )
        executor = pouring_setup.executor(world=world)
        transfer_statechart = Statechart(context=executor.context)
        motion = Parallel(
            nodes=[transfer_task, aiming_task, clearance_task, keep_plane]
        )
        transfer_statechart.add_node(node=motion)
        transfer_statechart.add_node(node=EndMotion.when_true(node=motion))

        gate_history: list[float] = []
        tilt_history: list[float] = []
        clearance_history: list[float] = []
        original_on_tick = transfer_task.on_tick

        def recording_on_tick() -> ObservationStateValue | None:
            """Record gate, source tilt and cup-origin clearance, then tick."""
            inflow_equation = transfer_task.fill_connection.inflow_equation
            gate_history.append(float(inflow_equation.gate.evaluate()[0]))
            tilt_history.append(
                float(inflow_equation.source_tilt_expression.evaluate()[0])
            )
            source_z = world.compute_forward_kinematics_np(world.root, source_cup.root)[
                2, 3
            ]
            receiver_z = world.compute_forward_kinematics_np(
                world.root, receiving_cup.root
            )[2, 3]
            clearance_history.append(float(source_z - receiver_z))
            return original_on_tick()

        transfer_task.on_tick = recording_on_tick

        executor.compile(statechart=transfer_statechart)
        executor.tick_until_end(timeout=pouring_setup.timeout)

        tolerance = transfer_task.fill_level_tolerance
        assert transfer_task.observation_state == ObservationStateValue.TRUE
        assert receiving_cup.fill_level == pytest.approx(
            transfer_task.goal_value, abs=tolerance
        )
        receiver_gain = receiving_cup.fill_level
        source_loss = source_fill_before - source_cup.fill_level
        assert source_loss > tolerance, "source cup never poured"
        assert receiver_gain == pytest.approx(source_loss, abs=tolerance), (
            "transfer must be volume conserving (equal cups): "
            f"receiver gained {receiver_gain:.3f}, source lost {source_loss:.3f}"
        )
        assert max(tilt_history) > 0.5, "the source cup never tilted to pour"
        first_open_tick = next(
            (tick for tick, gate in enumerate(gate_history) if gate > 0.5), None
        )
        assert (
            first_open_tick is not None
        ), "the optimizer never aimed the pour into the receiver"
        assert min(gate_history[first_open_tick:]) > 0.5, (
            "the stream left the receiver after it was aimed: "
            f"the gate fell to {min(gate_history[first_open_tick:]):.3f}"
        )
        assert min(clearance_history) > 0.0, (
            "the source cup dropped to or below the receiver during the pour: "
            f"minimum clearance was {min(clearance_history):.3f}"
        )


class TestRimClearanceDuringTransfer:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveRec
    eiverRim` keeps the pouring cup's lip above the receiving cup's rim
    throughout the transfer, and the motion statechart stays serializable."""

    def test_transfer_motion_is_json_serializable(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """The transfer motion statechart survives a JSON round trip, as it
        must to be sent to a Giskard server."""
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

    def test_pouring_lip_stays_above_receiver_rim(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """The rim clearance stays positive for the whole pour."""
        transfer = TransferMotion.build(
            transfer_world=tracy_transfer_world, setup=pouring_setup
        )
        clearance_history = transfer.record_rim_clearance()

        transfer.execute()

        assert transfer.transfer_task.observation_state == ObservationStateValue.TRUE
        assert clearance_history, "transfer never ticked"
        assert min(clearance_history) > 0.0, (
            "the pouring lip dropped to or below the receiver rim: "
            f"minimum clearance was {min(clearance_history):.3f} m"
        )


class TestFillDriveStartsOnceAimed:
    """The fill-driving task of the transfer motion stays inactive until the
    pour-aiming task observes the stream landing in the receiver, since the
    transfer gate blocks all flow before."""

    def test_fill_task_waits_for_the_aim_observation(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """The fill task does not start while the pour is mis-aimed, and starts
        on the tick after the aiming task first observes the landing inside the
        receiver."""
        transfer = TransferMotion.build(
            transfer_world=tracy_transfer_world, setup=pouring_setup
        )
        transfer_task = transfer.transfer_task
        aiming_task = transfer.aiming_task
        executor = transfer.executor

        executor.compile(statechart=transfer.statechart)
        executor.tick()
        assert (
            aiming_task.observation_state != ObservationStateValue.TRUE
        ), "the pour must start mis-aimed for this test to be meaningful"
        assert transfer_task.life_cycle_state == LifeCycleValue.NOT_STARTED

        aimed_tick = None
        for tick in range(1000):
            if aiming_task.observation_state == ObservationStateValue.TRUE:
                aimed_tick = tick
                break
            assert (
                transfer_task.life_cycle_state == LifeCycleValue.NOT_STARTED
            ), f"fill task started at tick {tick} before the pour was aimed"
            executor.tick()
        assert aimed_tick is not None, "the pour never got aimed"

        executor.tick()
        assert transfer_task.life_cycle_state == LifeCycleValue.RUNNING
        executor.finish_run()


class TestClearanceBandStaysAboveTheRim:
    """The clearance band of
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveReceiverRim`
    is enforced softly, so a band reaching down to the rim is rejected."""

    def test_band_reaching_down_to_the_rim_is_rejected(self, *, world_with_cup) -> None:
        """A zero clearance floor asks for the rims to touch, so building the
        task raises."""
        world, cup = world_with_cup
        clearance_task = KeepSourceRimAboveReceiverRim(
            receiver=cup, source=cup, minimum_clearance=0.0
        )
        add_to_statechart_with_motion_control(clearance_task, world)

        with pytest.raises(NonPositiveClearanceError) as error_info:
            clearance_task.build()

        assert error_info.value.minimum_clearance == clearance_task.minimum_clearance

    def test_band_is_derived_from_the_configured_width(self, *, world_with_cup) -> None:
        """The band's upper end lies :attr:`clearance_band` above the floor."""
        _world, cup = world_with_cup
        clearance_task = KeepSourceRimAboveReceiverRim(
            receiver=cup,
            source=cup,
            minimum_clearance=0.1,
            clearance_band=0.04,
        )

        assert clearance_task.maximum_clearance == pytest.approx(
            clearance_task.minimum_clearance + clearance_task.clearance_band
        )

    def test_rims_do_not_cross_when_aiming_outweighs_the_clearance(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """With the pour-aiming task at maximum weight the pouring lip still
        never reaches the receiver rim."""
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


def _box_cup_body(*, name: str, height: float = 0.1, width: float = 0.06) -> Body:
    """
    :param name: The name of the body.
    :param height: Height of the cup, in metres.
    :param width: Width and depth of the cup, in metres.
    :return: A box-shaped cup body whose bounding box spans ``[0, height]`` in z.
    """
    body = Body(name=PrefixedName(name))
    cup_shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=height / 2, reference_frame=body
        ),
        scale=Scale(width, width, height),
    )
    body.visual = ShapeCollection(shapes=[cup_shape])
    body.collision = ShapeCollection(shapes=[cup_shape])
    body.collision.reference_frame = body
    return body


def _rim_band_world(
    *, source_height_offset: float
) -> tuple[World, PourableContainer, PourableContainer]:
    """
    :param source_height_offset: Height of the source cup's origin above the
        receiver's, which equals the rim-to-rim clearance.
    :return: ``(world, source, receiver)``: two identical upright box cups.
    """
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        world.add_body(map_body)
    receiver_body = _box_cup_body(name="receiver_cup")
    source_body = _box_cup_body(name="source_cup")
    with world.modify_world():
        world.add_body(receiver_body)
        world.add_body(source_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=map_body,
                child=receiver_body,
                name=PrefixedName("map_T_receiver_cup"),
            )
        )
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=map_body,
                child=source_body,
                name=PrefixedName("map_T_source_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=source_height_offset
                ),
            )
        )
    receiver = PourableContainer(name=PrefixedName("receiver_cup"), root=receiver_body)
    source = PourableContainer(name=PrefixedName("source_cup"), root=source_body)
    with world.modify_world():
        world.add_semantic_annotation(receiver)
        world.add_semantic_annotation(source)
    return world, source, receiver


@dataclass(kw_only=True)
class ClearanceBandProbe:
    """Evaluates the observation of a clearance task with a fixed band in
    worlds of varying rim-to-rim clearance."""

    minimum_clearance: float = 0.05
    """Clearance floor of the probed band, in metres."""

    clearance_band: float = 0.05
    """Width of the probed band above the floor, in metres."""

    def observation_at(self, *, clearance: float) -> float:
        """
        :param clearance: The rim-to-rim clearance of the world the task is built in.
        :return: The task's observation in that world.
        """
        world, source, receiver = _rim_band_world(source_height_offset=clearance)
        clearance_task = KeepSourceRimAboveReceiverRim(
            receiver=receiver,
            source=source,
            minimum_clearance=self.minimum_clearance,
            clearance_band=self.clearance_band,
        )
        add_to_statechart_with_motion_control(clearance_task, world)
        artifacts = clearance_task.build()
        return float(artifacts.observation.evaluate()[0])


@pytest.fixture
def clearance_band_probe() -> ClearanceBandProbe:
    """A clearance band probe with a 5 cm floor and a 5 cm band."""
    return ClearanceBandProbe()


class TestRimClearanceBandObservation:
    """The observation of
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveReceiverRim`
    reports whether the rim-to-rim clearance lies within the band."""

    def test_clearance_inside_the_band_observes_true(
        self, *, clearance_band_probe: ClearanceBandProbe
    ) -> None:
        """A clearance between floor and ceiling satisfies the band."""
        probe = clearance_band_probe
        clearance = probe.minimum_clearance + probe.clearance_band / 2
        assert probe.observation_at(clearance=clearance) == 1.0

    def test_clearance_above_the_band_observes_false(
        self, *, clearance_band_probe: ClearanceBandProbe
    ) -> None:
        """A clearance above the ceiling violates the band."""
        probe = clearance_band_probe
        clearance = probe.minimum_clearance + 2 * probe.clearance_band
        assert probe.observation_at(clearance=clearance) == 0.0

    def test_clearance_below_the_band_observes_false(
        self, *, clearance_band_probe: ClearanceBandProbe
    ) -> None:
        """A clearance below the floor violates the band."""
        probe = clearance_band_probe
        clearance = probe.minimum_clearance / 2
        assert probe.observation_at(clearance=clearance) == 0.0


class TestFillTaskInitializationErrors:
    """The transfer tasks validate their liquid coupling at build time and
    raise concrete initialization errors."""

    def test_fill_by_transfer_requires_inflow_coupling(self, *, world_with_cup) -> None:
        """A receiver never coupled via receive_outflow_from cannot drive a
        transfer."""
        world, cup = world_with_cup
        transfer_task = FillByTransferTask(
            receiver=cup, goal_value=0.7, fill_level_tolerance=0.05
        )
        add_to_statechart_with_motion_control(transfer_task, world)

        with pytest.raises(MissingInflowEquationError):
            transfer_task.build()

    def test_keep_projectile_requires_inflow_coupling(self, *, world_with_cup) -> None:
        """The aiming task needs the receiver's inflow equation to derive the
        landing point."""
        world, cup = world_with_cup
        aiming_task = KeepProjectileInReceiver(receiver=cup, source=cup)
        add_to_statechart_with_motion_control(aiming_task, world)

        with pytest.raises(MissingInflowEquationError):
            aiming_task.build()

    def test_keep_projectile_requires_an_exit_speed(self, *, world_with_cup) -> None:
        """A source with no live outflow model and an ungated inflow equation
        leave no exit speed to throw the stream at."""
        world, cup = world_with_cup
        cup.fill_connection.inflow_equation = InflowEquation(
            container_height=0.1, container_width=0.06
        )
        speedless_source = PourableContainer(
            name=PrefixedName("speedless_source"),
            root=Body(name=PrefixedName("speedless_source_body")),
        )
        assert speedless_source.current_exit_speed(world=world) is None
        aiming_task = KeepProjectileInReceiver(receiver=cup, source=speedless_source)
        add_to_statechart_with_motion_control(aiming_task, world)

        with pytest.raises(MissingExitSpeedError):
            aiming_task.build()


class TestFillRowLeavesPlacementToTheOtherTasks:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.FillByTransferTask`
    predicts the fill through the transfer gate but does not steer by it:
    placing the source over the receiver is left to the aim and the
    clearance."""

    def test_the_fill_row_does_not_move_the_source_towards_the_receiver(
        self, *, tracy_transfer_world
    ) -> None:
        """The fill velocity has no gradient along the receiver's own degrees
        of freedom, while the tilt still drives it."""
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

    def test_the_fill_rate_still_closes_with_the_gate(
        self, *, tracy_transfer_world
    ) -> None:
        """Moved out from under the pour, the receiver is predicted to receive
        nothing."""
        world, _source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        transfer_task = FillByTransferTask(
            receiver=receiving_cup, goal_value=0.7, fill_level_tolerance=0.05
        )
        add_to_statechart_with_motion_control(transfer_task, world)
        transfer_task.build()
        receiver_connection = receiving_cup.root.parent_connection

        transfer_task.on_tick()
        under_the_pour = float(transfer_task._compiled_fill_velocity.evaluate()[0])
        receiver_connection.origin = (
            receiver_connection.origin
            @ HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)
        )
        transfer_task.on_tick()
        moved_away = float(transfer_task._compiled_fill_velocity.evaluate()[0])

        assert under_the_pour > 0.0
        assert moved_away == pytest.approx(0.0, abs=1e-6)


class TestTransferPredictionDrainsTheSource:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.FillByTransferTask`
    lets the source's fill drain over the prediction window."""

    def test_the_source_fill_evolves_in_the_receivers_prediction(
        self, *, tracy_transfer_world
    ) -> None:
        """The fill row carries the source's fill level as a coupled state,
        moving at the source's own drain rate."""
        world, source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        transfer_task = FillByTransferTask(
            receiver=receiving_cup, goal_value=0.7, fill_level_tolerance=0.05
        )
        add_to_statechart_with_motion_control(transfer_task, world)

        artifacts = transfer_task.build()

        [row] = _prediction_rows(artifacts=artifacts)
        [coupled] = row.coupled_states
        source_connection = source_cup.fill_connection
        assert coupled.variable.id == source_connection.raw_dof.variables.position.id
        assert coupled.velocity.evaluate()[0] == pytest.approx(
            source_connection.symbolic_fill_velocity().evaluate()[0]
        )

    def test_the_source_fill_can_be_held_instead(self, *, tracy_transfer_world) -> None:
        """Without the coupling the prediction holds every other fill level."""
        world, _source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        transfer_task = FillByTransferTask(
            receiver=receiving_cup,
            goal_value=0.7,
            fill_level_tolerance=0.05,
            couple_source_fill=False,
        )
        add_to_statechart_with_motion_control(transfer_task, world)

        artifacts = transfer_task.build()

        [row] = _prediction_rows(artifacts=artifacts)
        assert row.coupled_states == []


def _prediction_rows(
    *, artifacts: MotionNodeArtifacts
) -> list[TerminalStatePredictionConstraint]:
    """
    :param artifacts: The artifacts a fill task built.
    :return: The terminal-state prediction rows among its equality constraints.
    """
    return [
        constraint
        for constraint in artifacts.constraints.equality_constraints
        if isinstance(constraint, TerminalStatePredictionConstraint)
    ]


class TestTransferPredictionIsExposed:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.FillByTransferTask`
    exposes what its row predicts and the fill rate its model gives as debug
    expressions."""

    def test_the_rate_is_the_receivers_own_fill_rate(
        self, *, tracy_transfer_world, pouring_setup: PouringSetup
    ) -> None:
        """The fill-rate debug expression gives the rate the receiver's fill
        connection integrates, and a predicted fill is exposed next to it."""
        world, _source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        transfer_task = FillByTransferTask(
            receiver=receiving_cup, goal_value=0.7, fill_level_tolerance=0.05
        )
        pouring_setup.put_into_statechart(task=transfer_task, world=world)

        artifacts = transfer_task.build()

        expression_names = {
            expression.name for expression in artifacts.debug_expressions
        }
        rate = debug_expression_by_name(
            artifacts.debug_expressions,
            f"{transfer_task.name}/{FillDebugValue.FILL_RATE}",
        )
        assert f"{transfer_task.name}/{FillDebugValue.PREDICTED_FILL}" in (
            expression_names
        )
        assert rate.expression.evaluate()[0] == pytest.approx(
            receiving_cup.fill_connection.symbolic_fill_velocity().evaluate()[0]
        )


# %% aiming the stream


class TestProjectileAimingErrorIsHorizontal:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.KeepProjectileInRecei
    ver` measures only where the stream lands within the receiver's opening
    plane, so the pour can be reported as aimed."""

    def test_task_reports_aimed_once_the_landing_is_within_the_threshold(
        self, *, tracy_transfer_world
    ) -> None:
        """A threshold above the horizontal landing distance to the opening
        centre is satisfied, one below it is not."""
        world, source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        landing = receiving_cup.projectile_landing_point(
            source=source_cup,
            world=world,
            exit_speed=source_cup.current_exit_speed(world=world),
        )
        opening = receiving_cup.opening_point(world=world)
        horizontal_distance = math.hypot(
            landing.x.evaluate()[0] - opening.x.evaluate()[0],
            landing.y.evaluate()[0] - opening.y.evaluate()[0],
        )

        def is_aimed(threshold: float) -> bool:
            """
            :param threshold: The aiming task's threshold.
            :return: Whether the aiming task observes the pour as aimed.
            """
            task = KeepProjectileInReceiver(
                receiver=receiving_cup, source=source_cup, threshold=threshold
            )
            add_to_statechart_with_motion_control(task, world)
            artifacts = task.build()
            return bool(artifacts.observation.evaluate()[0])

        assert is_aimed(horizontal_distance * 1.5)
        assert not is_aimed(horizontal_distance * 0.5)


class TestKeepProjectileInReceiverDebugExpressions:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.KeepProjectileInRecei
    ver` exposes the stream's exit point and landing point as debug
    expressions."""

    def test_registers_exit_and_landing_points(self, *, tracy_transfer_world) -> None:
        """Building the task exposes the exit and landing points as colored
        point markers, the exit marker on the source's exit point."""
        world, source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        aiming_task = KeepProjectileInReceiver(
            receiver=receiving_cup, source=source_cup
        )
        add_to_statechart_with_motion_control(aiming_task, world)

        artifacts = aiming_task.build()

        exit_point = debug_expression_by_name(
            artifacts.debug_expressions,
            f"{aiming_task.name}/{StreamDebugValue.EXIT_POINT}",
        )
        landing_point = debug_expression_by_name(
            artifacts.debug_expressions,
            f"{aiming_task.name}/{StreamDebugValue.LANDING_POINT}",
        )
        assert isinstance(exit_point.expression, Point3)
        assert isinstance(landing_point.expression, Point3)
        assert exit_point.color == KeepProjectileInReceiver.exit_point_color
        assert landing_point.color == KeepProjectileInReceiver.landing_point_color
        np.testing.assert_allclose(
            exit_point.expression.evaluate(),
            source_cup.liquid_exit_point(world=world).evaluate(),
        )

    def test_landing_uses_current_exit_speed(self, *, tracy_transfer_world) -> None:
        """The landing point is thrown at the source's live exit speed, not the
        nominal speed stored on the inflow coupling."""
        world, source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        aiming_task = KeepProjectileInReceiver(
            receiver=receiving_cup, source=source_cup
        )
        add_to_statechart_with_motion_control(aiming_task, world)

        artifacts = aiming_task.build()

        landing = debug_expression_by_name(
            artifacts.debug_expressions,
            f"{aiming_task.name}/{StreamDebugValue.LANDING_POINT}",
        )
        live_speed = source_cup.current_exit_speed(world=world)
        nominal_exit_speed = receiving_cup.fill_connection.inflow_equation.exit_speed
        assert live_speed.evaluate()[0] != pytest.approx(nominal_exit_speed)
        expected = receiving_cup.projectile_landing_point(
            source=source_cup, world=world, exit_speed=live_speed
        )
        assert landing.expression.x.evaluate()[0] == pytest.approx(
            expected.x.evaluate()[0]
        )
        assert landing.expression.y.evaluate()[0] == pytest.approx(
            expected.y.evaluate()[0]
        )


# %% filling under perception noise


class TestPerceptionCorrectedTransfer:
    """:class:`~giskardpy.motion_statechart.tasks.pouring.FillByTransferTask`
    converges when the receiver's fill belief is replaced by noisy measurements
    of the true fill at a rate below the control rate."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "sigma,perception_hz",
        [(0.01, 10), (0.02, 10), (0.01, 20), (0.01, 30)],
        ids=[
            "sigma=0.01_10Hz",
            "sigma=0.02_10Hz",
            "sigma=0.01_20Hz",
            "sigma=0.01_30Hz",
        ],
    )
    def test_convergence_under_perception_noise(
        self,
        *,
        tracy_transfer_world,
        pouring_setup: PouringSetup,
        sigma: float,
        perception_hz: float,
    ) -> None:
        """The transfer ends with the receiver's true fill at the goal, within
        the tolerance widened by two noise deviations, as the task ends on the
        perceived fill.

        :param sigma: Standard deviation of the measurement noise.
        :param perception_hz: Rate of the measurements, in Hz.
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
            perception_hz=perception_hz,
            rng=np.random.default_rng(seed=42),
        )

        perception.run(executor=transfer.executor, setup=pouring_setup)

        transfer_task = transfer.transfer_task
        assert transfer_task.observation_state == ObservationStateValue.TRUE, (
            f"Transfer task did not converge "
            f"(sigma={sigma}, perception_hz={perception_hz} Hz)"
        )
        source_loss = 1.0 - float(transfer.source_cup.fill_level)
        assert (
            source_loss > transfer_task.fill_level_tolerance
        ), "source cup never poured"
        assert perception.true_fill == pytest.approx(
            transfer_task.goal_value,
            abs=transfer_task.fill_level_tolerance + 2.0 * sigma,
        ), (
            f"true receiver fill {perception.true_fill:.3f} outside goal band around "
            f"{transfer_task.goal_value} (sigma={sigma}, perception_hz={perception_hz} Hz)"
        )
