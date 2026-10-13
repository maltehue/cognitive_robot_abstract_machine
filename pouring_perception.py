"""
Watch the receiver's opening with the pole camera and publish, for every image, whether
a stream crosses it, how many grains have crossed so far, and where.

Start it once, after the robot and camera drivers and next to ``tracy_velocity.py``, and
leave it running: ``demo_pouring_transfer.py`` with ``USE_LIVE_BANK = True`` asks it to
learn the empty view before every pour, which also restarts its count, and its live bank
reads what it publishes. No arguments are needed; the settings below are the defaults.

The camera's pose and the receiver's opening come from the transforms, so the camera has
to be calibrated. Readings go out as JSON on ``/stream_detector/reading``.
"""

from __future__ import annotations

import json
from enum import StrEnum

import cv2
import numpy
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from experiments.real_pouring.stream_detection import (
    CameraView,
    OpeningRegion,
    StreamDetector,
    WhiteSegmentation,
)
from krrood.adapters.json_serializer import to_json

# %% settings


class Parameter(StrEnum):
    """
    The node's ROS parameters.
    """

    IMAGE_TOPIC = "image_topic"
    CAMERA_INFO_TOPIC = "camera_info_topic"
    WORLD_FRAME = "world_frame"
    RECEIVER_FRAME = "receiver_frame"
    OPENING_HEIGHT = "opening_height"
    OPENING_RADIUS = "opening_radius"
    PIXEL_SECONDS_PER_GRAIN = "pixel_seconds_per_grain"
    FRAME_PERIOD = "frame_period"
    BACKGROUND_FRAMES = "background_frames"


STYROFOAM_PIXEL_SECONDS_PER_GRAIN = 29.15
"""
How much stream area, times the time it shows, one of the live bank's grains (a percent
of a full Jeroen cup) adds for the styrofoam balls: 1100 pixel-seconds per gram in the
five demo pours of 2026-10-12, times 2.65 g per full cup over 100 grains.

Another substance needs its own value: the demo prints the factor to correct it by after
every weighed pour.
"""

PIXEL_SECONDS_PER_GRAIN = STYROFOAM_PIXEL_SECONDS_PER_GRAIN
"""
The count calibration in use: change it when pouring another substance, by the factor
the demo prints after a weighed pour.
"""

DEFAULTS = {
    Parameter.IMAGE_TOPIC: "/camera/color/image_raw/compressed",
    Parameter.CAMERA_INFO_TOPIC: "/camera/color/camera_info",
    Parameter.WORLD_FRAME: "map",
    Parameter.RECEIVER_FRAME: "receiving_cup",
    Parameter.OPENING_HEIGHT: 0.16,
    Parameter.OPENING_RADIUS: 0.035,
    Parameter.PIXEL_SECONDS_PER_GRAIN: PIXEL_SECONDS_PER_GRAIN,
    Parameter.FRAME_PERIOD: 1 / 30,
    Parameter.BACKGROUND_FRAMES: 15,
}
"""
The parameters' defaults, for the Jeroen cup under the Tracy pole camera.
"""


def _matrix(transform) -> numpy.ndarray:
    """
    :param transform: A ``geometry_msgs/Transform``.
    :return: The same transform as a homogeneous matrix.
    """
    matrix = numpy.eye(4)
    rotation = transform.rotation
    matrix[:3, :3] = Rotation.from_quat(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    ).as_matrix()
    translation = transform.translation
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


# %% the node


class StreamDetectorNode(Node):
    """
    Feeds the camera's images to a :class:`StreamDetector` and publishes its readings.
    """

    def __init__(self) -> None:
        super().__init__("stream_detector")
        for parameter, default in DEFAULTS.items():
            self.declare_parameter(parameter.value, default)
        self.transforms = Buffer()
        self.listener = TransformListener(self.transforms, self)
        self.camera_info = None
        self.waiting_reported = False
        self.detector = None
        self.background = []
        self.publisher = self.create_publisher(String, "~/reading", 10)
        self.create_subscription(
            CameraInfo,
            self._value(Parameter.CAMERA_INFO_TOPIC),
            self._on_camera_info,
            1,
        )
        self.create_subscription(
            CompressedImage, self._value(Parameter.IMAGE_TOPIC), self._on_image, 10
        )
        self.create_service(Trigger, "~/learn_background", self._on_learn_background)

    def _value(self, parameter: Parameter):
        return self.get_parameter(parameter.value).value

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _on_learn_background(self, request, response):
        self.get_logger().info("learning the empty view for the next pour")
        self.detector = None
        self.background = []
        response.success = True
        response.message = "learning the background from the next images"
        return response

    def _on_image(self, message: CompressedImage) -> None:
        if self.camera_info is None:
            return
        image = cv2.imdecode(
            numpy.frombuffer(message.data, numpy.uint8), cv2.IMREAD_COLOR
        )
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if self.detector is None:
            self._learn(image)
            return
        reading = self.detector.observe(image, stamp=stamp)
        self.publisher.publish(String(data=json.dumps(to_json(reading))))

    def _learn(self, image: numpy.ndarray) -> None:
        """
        Collect the images before the pour, and build the detector once there are enough
        and the transforms are known.
        """
        self.background.append(image)
        if len(self.background) < self._value(Parameter.BACKGROUND_FRAMES):
            return
        world = self._value(Parameter.WORLD_FRAME)
        latest = Time()
        camera_frame = self.camera_info.header.frame_id
        receiver_frame = self._value(Parameter.RECEIVER_FRAME)
        if not (
            self.transforms.can_transform(world, camera_frame, latest)
            and self.transforms.can_transform(world, receiver_frame, latest)
        ):
            self.background = self.background[1:]
            if not self.waiting_reported:
                self.get_logger().info(
                    f"waiting for '{receiver_frame}' in the transforms: it appears once "
                    "the demo has created the cups"
                )
                self.waiting_reported = True
            return
        world_T_camera = _matrix(
            self.transforms.lookup_transform(world, camera_frame, latest).transform
        )
        world_T_receiver = _matrix(
            self.transforms.lookup_transform(world, receiver_frame, latest).transform
        )
        opening_centre = world_T_receiver[:3, 3] + world_T_receiver[
            :3, 2
        ] * self._value(Parameter.OPENING_HEIGHT)
        detector = StreamDetector(
            view=CameraView(
                intrinsics=numpy.array(self.camera_info.k).reshape(3, 3),
                world_T_camera=world_T_camera,
            ),
            opening=OpeningRegion(
                centre=opening_centre, radius=self._value(Parameter.OPENING_RADIUS)
            ),
            segmentation=WhiteSegmentation(),
            pixel_seconds_per_grain=self._value(Parameter.PIXEL_SECONDS_PER_GRAIN),
            frame_period=self._value(Parameter.FRAME_PERIOD),
        )
        detector.learn_background(self.background)
        self.detector = detector
        self.background = []
        self.get_logger().info(
            f"watching the opening at {numpy.round(opening_centre, 3).tolist()}"
        )


def main() -> None:
    rclpy.init()
    node = StreamDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
