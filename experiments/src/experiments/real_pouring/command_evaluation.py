"""Counterfactual full-arm commands along recorded robot states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path

import numpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState as JointStateMessage

from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.controller_ablation import ControllerProbe
from experiments.real_pouring.pour_replay import (
    PourRecording,
    ReplayStep,
    GRAINS_PER_FILL,
)
from experiments.real_pouring.pouring_ablation import ForecastModel
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

# %% recorded joint state


class BagField(StrEnum):
    """Recorded ROS topic and transport identifiers."""

    JOINT_STATES = "/left_arm/joint_states"
    """Measured arm positions and velocities."""
    STORAGE = "mcap"
    """Recorded storage format."""
    SERIALIZATION = "cdr"
    """ROS message serialization format."""
    TIME = "time"
    """Original live-bank timestamp."""


@dataclass(frozen=True)
class ArmSample:
    """One measured joint state, before any counterfactual solve."""

    time: float
    """ROS header timestamp in seconds."""
    names: tuple[str, ...]
    """Names matching the URDF joint connections."""
    positions: tuple[float, ...]
    """Measured joint angles in radians."""
    velocities: tuple[float, ...]
    """Measured angular velocities in radians per second."""

    def apply(self, world: World) -> None:
        """Reset the local robot to a measured state with unobserved acceleration zero."""
        world.state.velocities[:] = 0
        world.state.accelerations[:] = 0
        world.state.jerks[:] = 0
        for name, position, velocity in zip(
            self.names, self.positions, self.velocities, strict=True
        ):
            connection = world.get_connection_by_name(name)
            connection.position = position
            connection.velocity = velocity

    @classmethod
    def read(cls, bag: Path) -> list[ArmSample]:
        """Read arm measurements without initializing ROS or opening a publisher."""
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag), storage_id=BagField.STORAGE),
            rosbag2_py.ConverterOptions(BagField.SERIALIZATION, BagField.SERIALIZATION),
        )
        reader.set_filter(rosbag2_py.StorageFilter(topics=[BagField.JOINT_STATES]))
        samples = []
        while reader.has_next():
            _, data, _ = reader.read_next()
            message = deserialize_message(data, JointStateMessage)
            samples.append(
                cls(
                    message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                    tuple(message.name),
                    tuple(message.position),
                    tuple(message.velocity),
                )
            )
        return sorted(samples, key=lambda sample: sample.time)


@dataclass(frozen=True)
class CommandComparison:
    """A model-specific command evaluated against an identical robot snapshot."""

    time: float
    """Seconds from the pour start."""
    model: ForecastModel
    """Alternative drain parameters supplied to the optimizer."""
    tilt: float
    """Source tilt reconstructed from the measured arm joints."""
    tilt_velocity: float
    """Next commanded tilt speed in radians per second."""
    joint_velocities: tuple[float, ...]
    """Next commands in the arm sample's joint order."""
    inflow: float
    """Current modeled receiver inflow in fill per second."""
    predicted_fill: float
    """Production zero-motion terminal prediction over the fill task's window."""
    reported_fill: float
    """Identical measured receiver state used for every alternative."""
    source_fill: float
    """Identical posterior source state used for every alternative."""
    pose_error: float
    """Source position discrepancy against the recorded transform, in metres."""
    angle_error: float
    """Source tilt discrepancy against the recorded transform, in radians."""
    joint_age: float
    """Age of the most recent joint sample, in seconds."""
    flowing: bool
    """Whether the camera detected a stream at this origin."""
    planned_tilt_velocities: tuple[float, ...]
    """Optimized tilt velocities over the configured motion horizon."""
    prediction_duration: float = 1.5
    """Fill forecast window; older saved probes used 1.5 seconds."""


@dataclass
class CommandEvaluation:
    """Reconstruct task commands, keeping measured states fixed across alternatives."""

    probe: ControllerProbe
    """Disconnected production controller."""
    recording: PourRecording
    """Recorded camera-aligned cup transforms."""
    arms: list[ArmSample]
    """Timestamped arm feedback."""

    def evaluate(
        self, steps: list[ReplayStep], stride: int = 5
    ) -> list[CommandComparison]:
        """Compare commands during the original transfer phase only."""
        world = self.probe.world
        times = numpy.array([sample.time for sample in self.arms])
        results = []
        # The robot URDF and the bag may use different names for the fixed root.
        initial = steps[0]
        absolute = initial.time + self.recording.pour_start
        first_arm = max(0, int(numpy.searchsorted(times, absolute, side="right")) - 1)
        self.arms[first_arm].apply(world)
        local_source = world.compute_forward_kinematics_np(
            world.root, self.probe.source.root
        )
        recorded_pose = self.recording.cup_poses[initial.frame_index]
        recorded_source = numpy.eye(4)
        recorded_source[:3, :3] = recorded_pose.rotation
        recorded_source[:3, 3] = recorded_pose.position
        local_from_recording = local_source @ numpy.linalg.inv(recorded_source)
        receiver = local_from_recording @ numpy.append(
            self.recording.receiver_position, 1.0
        )
        self.probe.receiver_joint.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            *receiver[:3], reference_frame=world.root
        )
        last_time = (
            self.recording.live_trace[-1][BagField.TIME] - self.recording.pour_start
        )
        for step in steps[::stride]:
            if step.time > last_time:
                break
            absolute = step.time + self.recording.pour_start
            index = int(numpy.searchsorted(times, absolute, side="right")) - 1
            if index < 0:
                continue
            arm = self.arms[index]
            arm.apply(world)
            self.probe.source.fill_connection.position = min(
                1.0, max(0.0, step.bank.source_grains / GRAINS_PER_FILL)
            )
            conversion = (
                GRAINS_PER_FILL
                * self.probe.receiver.capacity
                / self.probe.source.capacity
            )
            reported = min(1.0, step.told_grains / conversion)
            self.probe.receiver.fill_connection.position = reported
            reconstructed = world.compute_forward_kinematics_np(
                world.root, self.probe.source.root
            )
            pose = self.recording.cup_poses[step.frame_index]
            expected = local_from_recording @ numpy.append(pose.position, 1.0)
            pose_error = float(numpy.linalg.norm(reconstructed[:3, 3] - expected[:3]))
            tilt = float(self.probe.source.pour_tilt_expression.evaluate()[0])
            corrections = {
                ForecastModel.ADAPTED: step.bank.corrections,
                ForecastModel.INITIAL: DrainCorrectionValues(0.0, 0.0, 0.0),
                ForecastModel.FASTER: DrainCorrectionValues(
                    step.bank.corrections.repose_correction,
                    step.bank.corrections.log_drain_scale + math.log(2.0),
                    step.bank.corrections.log_discharge_scale,
                ),
                ForecastModel.PACING_ONLY: step.bank.corrections,
            }
            for model, correction in corrections.items():
                command = self.probe.command(
                    correction,
                    use_fill_constraint=model is not ForecastModel.PACING_ONLY,
                )
                joint_velocities = tuple(
                    float(
                        command.velocities[
                            world.state._index[
                                world.get_connection_by_name(name).dof.id
                            ]
                        ]
                    )
                    for name in arm.names
                )
                results.append(
                    CommandComparison(
                        step.time,
                        model,
                        tilt,
                        command.tilt_velocity,
                        joint_velocities,
                        command.inflow,
                        reported
                        + self.probe.fill_task.prediction_duration * command.inflow,
                        reported,
                        float(self.probe.source.fill_connection.position),
                        pose_error,
                        abs(tilt - pose.tilt),
                        absolute - arm.time,
                        step.reading.stream_seen,
                        command.planned_tilt_velocities,
                        self.probe.fill_task.prediction_duration,
                    )
                )
        return results
