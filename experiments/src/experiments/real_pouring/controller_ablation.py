"""Disconnected full-arm QP probes for paired pouring-model comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
import math

import numpy

from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights, LifeCycleValues
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
    TiltPacing,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Cylinder, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% disconnected controller


class ProbeEntity(StrEnum):
    """World entities required by the pouring experiment."""

    SOURCE = "source_cup"
    """Held pouring vessel."""
    RECEIVER = "receiving_cup"
    """Stationary receiving vessel."""
    TOOL = "l_gripper_tool_frame"
    """Source attachment frame."""
    SOURCE_MESH = "resources/stl/jeroen_cup.stl"
    """Cup mesh used by the physical demo."""


@dataclass(frozen=True)
class ProbeSettings:
    """Snapshot of the demo's control settings for reproducible offline solves."""

    fill_tolerance: float = 0.01
    """Receiver-fill completion tolerance."""
    fill_velocity: float = 0.03
    """Reference filling speed in receiver fill per second."""
    stopping_velocity: float = 1.0
    """Maximum corrective fill speed."""
    settled_inflow: float = 0.005
    """Measured inflow threshold for completion."""
    approach_rate: float = math.radians(3)
    """Approach preference and maximum forward tilt speed."""
    maximum_tilt: float = math.radians(110)
    """Maximum permitted source angle."""
    return_rate: float = math.radians(10)
    """Maximum tilt-back speed."""
    ceiling_width: float = math.radians(5)
    """Angular slowdown band below the ceiling."""
    minimum_clearance: float = 0.07
    """Minimum source-to-receiver rim clearance in metres."""
    clearance_band: float = 0.05
    """Permitted clearance band in metres."""
    aiming_velocity: float = 0.1
    """Reference speed of the aiming task."""
    initial_clearance: float = 0.2
    """Setup clearance before applying the recorded receiver position."""
    frequency: int = 80
    """Controller cycles per second."""
    horizon: int = 30
    """Optimization horizon in controller steps."""
    drain_rate: float = 0.8
    """Initial model's outflow constant."""
    discharge: float = 0.2
    """Initial model's discharge coefficient."""
    prediction_duration: float = 1.5
    """Fill forecast window in seconds, independent of the arm's control horizon."""


@dataclass(frozen=True)
class CommandProbeResult:
    """One controller solve at a supplied robot state."""

    velocities: numpy.ndarray
    """Next commanded joint velocities, in world state order."""
    tilt_velocity: float
    """Source tilt velocity induced by the command, in radians per second."""
    inflow: float
    """Production model's gated inflow before applying the command."""
    planned_tilt_velocities: tuple[float, ...]
    """Linearized tilt velocities over the optimized control horizon."""


@dataclass
class ControllerProbe:
    """Evaluate the demo's task set without a ROS executor or hardware connection."""

    world: World
    """Local robot world for snapshot and consecutive solves."""
    source: HasFillLevel
    """Held source with mutable drain corrections."""
    receiver: HasFillLevel
    """Receiver with externally supplied measured fill."""
    receiver_joint: Connection6DoF
    """Pose of the stationary receiving vessel."""
    executor: Executor
    """Compiled controller without a hardware interface."""
    fill_task: FillByTransferTask
    """Production terminal-fill objective."""

    @classmethod
    def build(
        cls,
        world: World,
        goal: float,
        *,
        prediction_duration: float | None = None,
        configuration: QPControllerConfig | None = None,
    ) -> ControllerProbe:
        """Attach cups and compile the current demo's five simultaneous tasks."""
        settings = ProbeSettings()
        if prediction_duration is not None:
            settings = replace(settings, prediction_duration=prediction_duration)
        tool = world.get_body_by_name(ProbeEntity.TOOL)
        mesh_path = (
            Path(files("semantic_digital_twin")).parent.parent / ProbeEntity.SOURCE_MESH
        )
        source_body = Body.from_shape_collection(
            shape_collection=ShapeCollection(
                [Mesh(filename=str(mesh_path), scale=Scale(1, 1, 1))]
            ),
            name=PrefixedName(ProbeEntity.SOURCE),
        )
        cup = ReceivingCup.TRANSLUCENT_CUP
        receiver_body = Body.from_shape_collection(
            shape_collection=ShapeCollection(
                [
                    Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=cup.height / 2
                        ),
                        width=2 * cup.inner_radius,
                        height=cup.height,
                    )
                ]
            ),
            name=PrefixedName(ProbeEntity.RECEIVER),
        )
        with world.modify_world():
            world.add_body(source_body)
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world=world,
                    parent=tool,
                    child=source_body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        roll=-math.pi / 2
                    ),
                )
            )
            world.add_body(receiver_body)
            receiver_joint = Connection6DoF.create_with_dofs(
                world, world.root, receiver_body
            )
            world.add_connection(receiver_joint)
            source = HasFillLevel(
                name=PrefixedName(ProbeEntity.SOURCE), root=source_body
            )
            world.add_semantic_annotation(source)
            receiver = HasFillLevel(
                name=PrefixedName(ProbeEntity.RECEIVER),
                root=receiver_body,
                measured_capacity=source.capacity
                * cup.source_cups_held(PouredMaterial.CORN),
            )
            world.add_semantic_annotation(receiver)
        source.initialize_fill_level(
            world=world,
            initial_fill=1.0,
            outflow_rate_constant=settings.drain_rate,
            discharge_coefficient=settings.discharge,
            repose_angle=0.0,
        )
        receiver.initialize_fill_level(world=world, initial_fill=0.0)
        source_position = world.compute_forward_kinematics_np(world.root, source_body)[
            :3, 3
        ]
        receiver_joint.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=source_position[0],
            y=source_position[1],
            z=source_position[2] - cup.height - settings.initial_clearance,
            reference_frame=world.root,
        )
        receiver.receive_outflow_from(source=source, world=world)
        fill_task = FillByTransferTask(
            receiver=receiver,
            goal_value=goal,
            fill_level_tolerance=settings.fill_tolerance,
            reference_velocity=settings.fill_velocity,
            stopping_velocity=settings.stopping_velocity,
            outflow_tolerance=settings.settled_inflow,
            prediction_duration=settings.prediction_duration,
        )
        motion = Parallel(
            [
                fill_task,
                KeepProjectileInReceiver(
                    receiver=receiver,
                    source=source,
                    weight=DefaultWeights.WEIGHT_MAXIMUM,
                    reference_velocity=settings.aiming_velocity,
                ),
                KeepSourceRimAboveReceiverRim(
                    receiver=receiver,
                    source=source,
                    minimum_clearance=settings.minimum_clearance,
                    clearance_band=settings.clearance_band,
                    weight=DefaultWeights.WEIGHT_MAXIMUM,
                ),
                AlignPlanes(
                    root_link=world.root,
                    tip_link=tool,
                    goal_normal=Vector3.X(reference_frame=world.root),
                    tip_normal=Vector3.Z(reference_frame=tool),
                ),
                TiltPacing(
                    source=source,
                    approach_rate=settings.approach_rate,
                    maximum_rate=settings.approach_rate,
                    maximum_tilt=settings.maximum_tilt,
                    maximum_return_rate=settings.return_rate,
                    ceiling_slowdown_width=settings.ceiling_width,
                    receiver=receiver,
                    approach_until_fill=goal - settings.fill_tolerance,
                ),
            ]
        )
        chart = MotionStatechart()
        chart.add_node(motion)
        executor = Executor(
            MotionStatechartContext(
                world=world,
                qp_controller_config=(
                    configuration
                    if configuration is not None
                    else QPControllerConfig(
                        target_frequency=settings.frequency,
                        prediction_horizon=settings.horizon,
                        verbose=False,
                    )
                ),
            )
        )
        executor.compile(chart)
        return cls(world, source, receiver, receiver_joint, executor, fill_task)

    def command(
        self, corrections: DrainCorrectionValues, use_fill_constraint: bool = True
    ) -> CommandProbeResult:
        """Solve once, returning commanded velocities without advancing the world."""
        before = self.world.state._data.copy()
        result = self.advance(corrections, use_fill_constraint)
        self.world.state._data[:] = before
        self.world.notify_state_change()
        return result

    def advance(
        self, corrections: DrainCorrectionValues, use_fill_constraint: bool = True
    ) -> CommandProbeResult:
        """Integrate one command, retaining motion derivatives for consecutive solves."""
        lifecycle = self.executor.motion_statechart.life_cycle_state
        previous_lifecycle = lifecycle.data.copy()
        if not use_fill_constraint:
            lifecycle[self.fill_task] = LifeCycleValues.PAUSED
        corrections.write_to(self.world, self.source.fill_connection.drain_correction)
        self.fill_task.on_tick(self.executor.context)
        inflow = float(self.fill_task.fill_velocity_expression.evaluate()[0])
        controller = self.executor.qp_controller
        solution = controller.compute_command(
            world_state=self.world.state._data,
            life_cycle_state=self.executor.motion_statechart.life_cycle_state.data,
            float_variables=self.executor.context.float_variable_data.data,
        )
        jacobian = self.source.pour_tilt_expression.jacobian(
            self.world.state.position_float_variables
        )
        gradient = numpy.asarray(jacobian.evaluate(), dtype=float).reshape(-1)
        count = len(controller.active_dofs)
        horizon = controller.config.control_horizon
        planned_velocities = solution.decision_variables[: count * horizon].reshape(
            horizon, count
        )
        planned_tilts = tuple(
            float(value)
            for value in planned_velocities @ gradient[controller.dof_filter]
        )
        self.world.apply_control_commands(
            solution.control_commands,
            controller.config.control_dt,
            controller.config.max_derivative,
        )
        velocities = self.world.state.velocities.copy()
        lifecycle.data[:] = previous_lifecycle
        return CommandProbeResult(
            velocities, float(gradient @ velocities), inflow, planned_tilts
        )
