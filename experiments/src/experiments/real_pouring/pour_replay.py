"""
A recorded pour replayed offline: the camera's frames through the stream detector and
its readings through the live bank, as the robot would have run them, so what the
perception saw and what the bank concluded can be looked at frame by frame.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy
import rosbag2_py
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from tf2_ros import BufferCore
from typing_extensions import List, Optional

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    FreeParameter,
    ModelPoint,
)
from experiments.pouring_model_verification.sensor import CameraCondition
from experiments.real_pouring.live_bank import LiveBank, LiveBankStep
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.stream_detection import (
    BLUE_CUP_OUTLINE,
    CameraView,
    ReceiverOutline,
    OpeningRegion,
    StreamDetector,
    StreamReading,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the setup the demo pours in

CUP_SIZE = Scale(0.07, 0.07, 0.16)
"""
The Jeroen cup's outer size, in metres.
"""

OPENING_HEIGHT = 0.16
"""
How far the receiver's opening lies above its frame, in metres.
"""

OPENING_RADIUS = 0.035
"""
The receiver's opening radius, in metres.
"""

FRAME_PERIOD = 1 / 30
"""
How long apart the pole camera takes its images, in seconds.
"""

BACKGROUND_FRAMES = 15
"""
How many frames before the pour make up the empty view.
"""

RECEIVER_OUTLINE = BLUE_CUP_OUTLINE
"""
The receiving cup's outline the opening is moved onto, as the perception node does.
"""

READING_EVERY = 3
"""
Every how many frames the live bank weighs a reading: ten times a second.
"""

GRAINS_PER_FILL = 100.0
"""
How many of the bank's grains fill the source cup.
"""

GOAL_FILL = 0.7
"""
The fill the demo pours the receiver to.
"""

PREDICTION_WINDOW = 1.5
"""
How far ahead the demo's fill task predicts, in seconds.
"""

START_TILT = 0.035
"""
The cup's tilt from upright, in radians, from which on a recording without a bank trace
counts as pouring.
"""

WORLD_FRAME = "map"
CAMERA_FRAME = "camera_color_optical_frame"
SOURCE_FRAME = "source_cup"
RECEIVER_FRAME = "receiving_cup"
IMAGE_TOPIC = "/camera/color/image_raw/compressed"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"
LIVE_BANK_TRACE_FILE_NAME = "live_bank_trace.json"
WEIGHED_PARTICLES_FILE_NAME = "weighed_particles.json"
RECORDING_SUFFIX = ".mcap"
ARCHIVE_SUFFIX = ".zip"


@dataclass
class NoRecordingFoundError(Exception):
    """
    Raised when a folder holds no bag recording.
    """

    folder: Path
    """
    The folder searched.
    """

    def __str__(self) -> str:
        return f"No {RECORDING_SUFFIX} recording anywhere in {self.folder}."


def bag_folder_in(folder: Path) -> Path:
    """
    :param folder: A folder holding a bag, at any depth, such as an unpacked download.
    :return: The bag's folder: the one holding its recording.
    :raises NoRecordingFoundError: If no recording lies anywhere in it.
    """
    recordings = sorted(folder.rglob(f"*{RECORDING_SUFFIX}"))
    if not recordings:
        raise NoRecordingFoundError(folder=folder)
    return recordings[0].parent


# %% what a bag holds


@dataclass(frozen=True)
class CameraFrame:
    """
    One image of the pole camera.
    """

    stamp: float
    """
    When it was taken, in seconds.
    """

    encoded: bytes
    """
    The image as the camera published it, compressed.
    """

    def image(self) -> numpy.ndarray:
        """
        :return: The image, BGR.
        """
        return cv2.imdecode(
            numpy.frombuffer(self.encoded, numpy.uint8), cv2.IMREAD_COLOR
        )


@dataclass(frozen=True)
class CupPose:
    """
    Where the source cup was at one time.
    """

    stamp: float
    """
    When, in seconds.
    """

    position: numpy.ndarray
    """
    Its origin in the world, in metres.
    """

    rotation: numpy.ndarray
    """
    Its orientation in the world, a rotation matrix.
    """

    @property
    def tilt(self) -> float:
        """
        :return: How far its axis leans from upright, in radians.
        """
        return float(numpy.arccos(numpy.clip(self.rotation[2, 2], -1.0, 1.0)))


@dataclass
class PourRecording:
    """
    What a demo bag recorded of one pour.
    """

    frames: List[CameraFrame]
    """
    The pole camera's images, in order.
    """

    intrinsics: numpy.ndarray
    """
    The camera's intrinsic matrix.
    """

    world_T_camera: numpy.ndarray
    """
    The camera's pose in the world.
    """

    receiver_position: numpy.ndarray
    """
    The receiver's frame in the world, in metres.
    """

    cup_poses: List[CupPose]
    """
    The source cup's pose at each frame.
    """

    pour_start: float
    """
    When the pour started, in seconds: the live bank's first reading, or the cup first
    leaning by :data:`START_TILT` beyond where it started.
    """

    live_trace: Optional[List[dict]] = None
    """
    What the live bank concluded on the robot, as recorded, or ``None`` without one.
    """

    weighed: Optional[dict] = None
    """
    What was weighed after the pour, as recorded, or ``None`` without it.
    """

    @classmethod
    def read(cls, bag: Path) -> PourRecording:
        """
        :param bag: A demo bag's folder, or a zip archive holding one.
        :return: What it recorded of the pour.
        """
        if bag.suffix == ARCHIVE_SUFFIX:
            with tempfile.TemporaryDirectory() as unpacked:
                zipfile.ZipFile(bag).extractall(unpacked)
                return cls._read_folder(bag_folder_in(Path(unpacked)))
        return cls._read_folder(bag)

    @classmethod
    def _read_folder(cls, bag: Path) -> PourRecording:
        """
        :param bag: A demo bag's folder.
        :return: What it recorded of the pour.
        """
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
        transforms = BufferCore(Duration(seconds=3600))
        frames, intrinsics = [], None
        while reader.has_next():
            topic, data, _ = reader.read_next()
            message = deserialize_message(data, types[topic])
            if topic in ("/tf", "/tf_static"):
                for transform in message.transforms:
                    if topic == "/tf_static":
                        transforms.set_transform_static(transform, "bag")
                    else:
                        transforms.set_transform(transform, "bag")
            elif topic == IMAGE_TOPIC:
                stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
                frames.append(CameraFrame(stamp=stamp, encoded=bytes(message.data)))
            elif topic == CAMERA_INFO_TOPIC and intrinsics is None:
                intrinsics = numpy.array(message.k).reshape(3, 3)

        def pose(target: str, source: str, stamp: float) -> numpy.ndarray:
            transform = transforms.lookup_transform_core(
                target, source, Time(nanoseconds=int(stamp * 1e9))
            ).transform
            matrix = numpy.eye(4)
            rotation = transform.rotation
            matrix[:3, :3] = Rotation.from_quat(
                [rotation.x, rotation.y, rotation.z, rotation.w]
            ).as_matrix()
            translation = transform.translation
            matrix[:3, 3] = [translation.x, translation.y, translation.z]
            return matrix

        def can_pose(target: str, source: str, stamp: float) -> bool:
            return transforms.can_transform_core(
                target, source, Time(nanoseconds=int(stamp * 1e9))
            )[0]

        frames = [
            frame
            for frame in frames
            if can_pose(WORLD_FRAME, SOURCE_FRAME, frame.stamp)
            and can_pose(WORLD_FRAME, RECEIVER_FRAME, frame.stamp)
            and can_pose(WORLD_FRAME, CAMERA_FRAME, frame.stamp)
        ]
        cup_poses = []
        for frame in frames:
            world_T_cup = pose(WORLD_FRAME, SOURCE_FRAME, frame.stamp)
            cup_poses.append(
                CupPose(
                    stamp=frame.stamp,
                    position=world_T_cup[:3, 3],
                    rotation=world_T_cup[:3, :3],
                )
            )
        trace_file, weighed_file = (
            bag / LIVE_BANK_TRACE_FILE_NAME,
            bag / WEIGHED_PARTICLES_FILE_NAME,
        )
        live_trace = json.loads(trace_file.read_text()) if trace_file.exists() else None
        first = frames[0].stamp
        if live_trace is not None:
            pour_start = live_trace[0]["time"]
        else:
            upright = cup_poses[0].tilt
            pour_start = next(
                (p.stamp for p in cup_poses if p.tilt > upright + START_TILT), first
            )
        return cls(
            frames=frames,
            intrinsics=intrinsics,
            world_T_camera=pose(WORLD_FRAME, CAMERA_FRAME, first),
            receiver_position=pose(WORLD_FRAME, RECEIVER_FRAME, first)[:3, 3],
            cup_poses=cup_poses,
            pour_start=pour_start,
            live_trace=live_trace,
            weighed=(
                json.loads(weighed_file.read_text()) if weighed_file.exists() else None
            ),
        )

    @property
    def first_pour_frame(self) -> int:
        """
        :return: The index of the first frame taken once the pour started.
        """
        return next(
            index
            for index, frame in enumerate(self.frames)
            if frame.stamp >= self.pour_start
        )


# %% the replay


@dataclass(frozen=True)
class ReplayStep:
    """
    One reading the replayed bank weighed.
    """

    frame_index: int
    """
    The frame it was read from.
    """

    time: float
    """
    Seconds since the pour started.
    """

    tilt: float
    """
    The source cup's tilt, in radians.
    """

    reading: StreamReading
    """
    What the detector saw of the stream.
    """

    bank: LiveBankStep
    """
    What the bank concluded and wrote.
    """

    told_grains: float
    """
    What the controller was told the receiver holds, margin included, in grains.
    """

    onset: Optional[float]
    """
    The repose angle the hypothesis in charge estimates, in radians, or ``None`` where
    it keeps the model's.
    """

    onset_spread: Optional[float]
    """
    How uncertain that estimate is, a standard deviation in radians.
    """


@dataclass
class PourReplay:
    """
    A recorded pour run through the detector and the live bank again.
    """

    recording: PourRecording
    """
    The pour replayed.
    """

    detector: StreamDetector
    """
    The detector, with the empty view learned before the pour.
    """

    steps: List[ReplayStep] = field(default_factory=list)
    """
    Every reading the bank weighed, in order.
    """

    @classmethod
    def of(
        cls,
        recording: PourRecording,
        material: PouredMaterial,
        drain: ModelPoint,
        settings: BankSettings,
        receiver_from_count: bool = False,
        stop_lead_time: float = 0.0,
        outline: Optional[ReceiverOutline] = RECEIVER_OUTLINE,
    ) -> PourReplay:
        """
        :param recording: The pour to replay.
        :param material: What was poured, which sets how the camera counts.
        :param drain: The drain the controller's model was built with.
        :param settings: How the bank weighs the pour.
        :param receiver_from_count: Whether the receiver is reported from the count.
        :param stop_lead_time: How long grains keep arriving after a stop, in seconds.
        :param outline: The receiving cup's outline the opening is moved onto, or
            ``None`` to count where the opening projects.
        :return: The replay, every reading weighed.
        """
        world, source, source_joint = _replay_world(recording, drain)
        detector = StreamDetector(
            view=CameraView(
                intrinsics=recording.intrinsics,
                world_T_camera=recording.world_T_camera,
            ),
            opening=OpeningRegion(
                centre=recording.receiver_position
                + numpy.array([0, 0, OPENING_HEIGHT]),
                radius=OPENING_RADIUS,
            ),
            segmentation=material.segmentation,
            pixel_seconds_per_grain=material.pixel_seconds_per_grain,
            frame_period=FRAME_PERIOD,
            outline=outline,
        )
        first = recording.first_pour_frame
        detector.learn_background(
            [
                frame.image()
                for frame in recording.frames[max(0, first - BACKGROUND_FRAMES) : first]
            ]
        )
        bank = LiveBank.for_cups(
            source=source,
            receiver=world.get_semantic_annotation_by_name("receiving_cup"),
            world=world,
            built=drain,
            settings=settings,
            camera=CameraCondition.POLE_COUNTING.model,
            grains_per_fill=GRAINS_PER_FILL,
            loaded_grains=GRAINS_PER_FILL,
            goal_grains=GOAL_FILL * GRAINS_PER_FILL,
            opening_radius=OPENING_RADIUS,
            prediction_window=PREDICTION_WINDOW,
            receiver_from_count=receiver_from_count,
            stop_lead_time=stop_lead_time,
        )
        replay = cls(recording=recording, detector=detector)
        for index in range(first, len(recording.frames)):
            frame = recording.frames[index]
            reading = detector.observe(
                frame.image(), stamp=frame.stamp - recording.pour_start
            )
            if (index - first) % READING_EVERY:
                continue
            pose = recording.cup_poses[index]
            source_joint.origin = _transform(pose, world)
            step = bank.observe(reading)
            active = bank.coupling.bank.filters[bank.coupling.active]
            estimates_onset = FreeParameter.REPOSE_ANGLE in active.free
            replay.steps.append(
                ReplayStep(
                    frame_index=index,
                    time=frame.stamp - recording.pour_start,
                    tilt=pose.tilt,
                    reading=reading,
                    bank=step,
                    told_grains=step.told_grains,
                    onset=active.mean().repose_angle if estimates_onset else None,
                    onset_spread=(
                        active.spread(FreeParameter.REPOSE_ANGLE)
                        if estimates_onset
                        else None
                    ),
                )
            )
        return replay

    def grain_mask(self, frame_index: int) -> numpy.ndarray:
        """
        :param frame_index: A frame of the recording.
        :return: Which of its pixels the detector takes for grains in the opening.
        """
        frames = self.recording.frames
        previous = frames[frame_index - 1].image() if frame_index > 0 else None
        return self.detector.grain_mask(frames[frame_index].image(), previous)

    def opening_mask(self) -> numpy.ndarray:
        """
        :return: Which pixels of a frame the detector watches as the receiver's opening.
        """
        return self.detector.opening_mask(self.recording.frames[0].image().shape[:2])


def _transform(pose: CupPose, world: World) -> HomogeneousTransformationMatrix:
    """
    :return: A cup pose as the world's transform.
    """
    x, y, z, w = Rotation.from_matrix(pose.rotation).as_quat()
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=pose.position[0],
        pos_y=pose.position[1],
        pos_z=pose.position[2],
        quat_x=x,
        quat_y=y,
        quat_z=z,
        quat_w=w,
        reference_frame=world.root,
    )


def _replay_world(
    recording: PourRecording, drain: ModelPoint
) -> tuple[World, HasFillLevel, Connection6DoF]:
    """
    :return: A world with the source cup, filled and draining as ``drain`` says, and the
        receiver where the recording had them, the source's joint to pose it by.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName(WORLD_FRAME)))
    cups = {}
    for name in ("source_cup", "receiving_cup"):
        body = Body(
            name=PrefixedName(name),
            collision=ShapeCollection(
                [
                    Box(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=CUP_SIZE.z / 2
                        ),
                        scale=CUP_SIZE,
                    )
                ]
            ),
        )
        with world.modify_world():
            world.add_body(body)
            joint = Connection6DoF.create_with_dofs(world, world.root, body)
            world.add_connection(joint)
        cup = HasFillLevel(name=PrefixedName(name), root=body)
        with world.modify_world():
            world.add_semantic_annotation(cup)
        cups[name] = (cup, joint)
    source, source_joint = cups["source_cup"]
    receiver, receiver_joint = cups["receiving_cup"]
    receiver_joint.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *recording.receiver_position, reference_frame=world.root
    )
    source_joint.origin = _transform(recording.cup_poses[0], world)
    source.initialize_fill_level(
        world=world,
        initial_fill=1.0,
        outflow_rate_constant=drain.outflow_rate_constant,
        discharge_coefficient=drain.discharge_coefficient,
        repose_angle=drain.repose_angle,
    )
    receiver.initialize_fill_level(world=world, initial_fill=0.0)
    receiver.receive_outflow_from(source=source, world=world)
    return world, source, source_joint
