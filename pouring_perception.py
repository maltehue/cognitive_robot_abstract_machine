"""
Watch the receiver's opening with the pole camera and publish, for every image, whether
a stream crosses it, how many grains have crossed so far, and where.

Start it once, after the robot and camera drivers and next to ``tracy_velocity.py``, and
leave it running: ``demo_pouring_transfer.py`` with ``USE_LIVE_BANK = True`` asks it to
learn the empty view before every pour, which also restarts its count, and its live bank
reads what it publishes. No arguments are needed; the settings below are the defaults.
Set :data:`MATERIAL` to what the demo pours and :data:`RECEIVER` to the cup it pours
into, or start it with ``--ros-args -p material:=corn -p receiver:=translucent_cup``.

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

from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.pour_perception import PourPerception
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import (
    CameraView,
    OBSERVATION_QUEUE_DEPTH,
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
    RECEIVER = "receiver"
    MATERIAL = "material"
    FRAME_PERIOD = "frame_period"
    BACKGROUND_FRAMES = "background_frames"


MATERIAL = PouredMaterial.CORN
"""
What the demo pours, which sets how grains are told from the empty view and how much
image area one counts for.
"""

RECEIVER = ReceivingCup.TRANSLUCENT_CUP
"""
The cup the demo pours into, which sets where its opening is, how the opening is placed
in the image despite the camera's pose being a few centimetres off, and how its fill is
read.
"""

DEFAULTS = {
    Parameter.IMAGE_TOPIC: "/camera/color/image_raw/compressed",
    Parameter.CAMERA_INFO_TOPIC: "/camera/color/camera_info",
    Parameter.WORLD_FRAME: "map",
    Parameter.RECEIVER_FRAME: "receiving_cup",
    Parameter.RECEIVER: RECEIVER.value,
    Parameter.MATERIAL: MATERIAL.value,
    Parameter.FRAME_PERIOD: 1 / 30,
    Parameter.BACKGROUND_FRAMES: 15,
}
"""
The parameters' defaults, for the Tracy pole camera.
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
        self.publisher = self.create_publisher(
            String, "~/reading", OBSERVATION_QUEUE_DEPTH
        )
        self.create_subscription(
            CameraInfo,
            self._value(Parameter.CAMERA_INFO_TOPIC),
            self._on_camera_info,
            1,
        )
        self.create_subscription(
            CompressedImage,
            self._value(Parameter.IMAGE_TOPIC),
            self._on_image,
            OBSERVATION_QUEUE_DEPTH,
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
        response.message = self._value(Parameter.MATERIAL)
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
        receiver = ReceivingCup(self._value(Parameter.RECEIVER))
        opening_centre = world_T_receiver[:3, 3] + world_T_receiver[:3, 2] * (
            receiver.height
        )
        material = PouredMaterial(self._value(Parameter.MATERIAL))
        detector = receiver.detector_for(
            material=material,
            view=CameraView(
                intrinsics=numpy.array(self.camera_info.k).reshape(3, 3),
                world_T_camera=world_T_camera,
            ),
            opening_centre=opening_centre,
            frame_period=self._value(Parameter.FRAME_PERIOD),
            wall_calibration=PourPerception.for_cup(
                receiver, material
            ).wall_calibration,
        )
        detector.learn_background(self.background)
        self.detector = detector
        self.background = []
        self.get_logger().info(
            f"watching the opening at {numpy.round(opening_centre, 3).tolist()} "
            f"of the {receiver} for {material}"
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
