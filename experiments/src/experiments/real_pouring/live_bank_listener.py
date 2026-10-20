"""
Hands the stream detector's readings to a :class:`LiveBank` while a pour runs on the
robot.
"""

# %% imports

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum

from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from typing_extensions import Optional

from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.stream_detection import OBSERVATION_QUEUE_DEPTH
from krrood.adapters.json_serializer import from_json

# %% the detector's interface


class DetectorInterface(StrEnum):
    """
    Where the stream detector node publishes and listens.
    """

    READING_TOPIC = "/stream_detector/reading"
    LEARN_BACKGROUND_SERVICE = "/stream_detector/learn_background"


READING_PERIOD = 0.1
"""
How long apart the bank weighs readings, in seconds: the detector reports every image,
the bank was tuned for ten readings a second.
"""

SERVICE_TIMEOUT = 5.0
"""
How long to wait for the detector's service, in seconds.
"""

FIRST_READING_TIMEOUT = 10.0
"""
How long to wait for the detector's first reading after asking it to learn the view, in
seconds: it learns from a few images, once the receiver shows up in the transforms.
"""

WAIT_POLL_INTERVAL = 0.05
"""
How often the first reading is looked for while waiting, in seconds.
"""


class DetectorNotRunningError(Exception):
    """
    Raised when the stream detector node does not answer.
    """

    def __init__(self) -> None:
        super().__init__(
            "The stream detector does not answer: start pouring_perception.py before "
            "the pour."
        )


@dataclass
class MaterialMismatchError(Exception):
    """
    Raised when the detector counts for another material than the one poured.
    """

    expected: PouredMaterial
    """
    The material poured.
    """

    reported: str
    """
    The material the detector reported counting for.
    """

    def __str__(self) -> str:
        return (
            f"The demo pours {self.expected} but pouring_perception.py counts for "
            f"'{self.reported}': restart it with --ros-args -p material:={self.expected}."
        )


class DetectorSendsNothingError(Exception):
    """
    Raised when the stream detector answers but sends no readings.
    """

    def __init__(self, timeout: float) -> None:
        super().__init__(
            f"The stream detector answered but sent no reading within {timeout} s: check "
            "that the camera topic is published and that the receiving cup is in the "
            "transforms."
        )


# %% the listener


@dataclass
class LiveBankListener:
    """
    Subscribes to the stream detector while a pour runs and lets the bank weigh one
    reading every :data:`READING_PERIOD`.
    """

    node: Node
    """
    The node the subscription and the service client live on.
    """

    bank: LiveBank
    """
    The bank the readings are handed to.
    """

    service_timeout: float = SERVICE_TIMEOUT
    """
    How long to wait for the detector's service, in seconds.
    """

    first_reading_timeout: float = FIRST_READING_TIMEOUT
    """
    How long to wait for the detector's first reading, in seconds.
    """

    material: Optional[PouredMaterial] = None
    """
    The material poured, which the detector has to count for, or ``None`` to not check.
    """

    _subscription: Optional[object] = field(init=False, default=None, repr=False)
    """
    The subscription to the detector's readings while listening.
    """

    _last_stamp: Optional[float] = field(init=False, default=None, repr=False)
    """
    When the reading weighed last was taken, in seconds.
    """

    def start(self) -> None:
        """
        Have the detector learn the empty view and restart its count, listen, and return
        once the bank has weighed the first reading.

        :raises DetectorNotRunningError: If the detector does not answer.
        :raises MaterialMismatchError: If it counts for another material than
            :attr:`material`.
        :raises DetectorSendsNothingError: If it answers but sends no reading.
        """
        client = self.node.create_client(
            Trigger, DetectorInterface.LEARN_BACKGROUND_SERVICE.value
        )
        if not client.wait_for_service(timeout_sec=self.service_timeout):
            raise DetectorNotRunningError()
        relearned = client.call_async(Trigger.Request())
        deadline = time.monotonic() + self.service_timeout
        while not relearned.done():
            if time.monotonic() > deadline:
                raise DetectorNotRunningError()
            time.sleep(WAIT_POLL_INTERVAL)
        reported = relearned.result().message
        if self.material is not None and reported != self.material.value:
            raise MaterialMismatchError(expected=self.material, reported=reported)
        self._subscription = self.node.create_subscription(
            String,
            DetectorInterface.READING_TOPIC.value,
            self._on_reading,
            OBSERVATION_QUEUE_DEPTH,
        )
        deadline = time.monotonic() + self.first_reading_timeout
        while not self.bank.trace:
            if time.monotonic() > deadline:
                self.stop()
                raise DetectorSendsNothingError(self.first_reading_timeout)
            time.sleep(WAIT_POLL_INTERVAL)

    def stop(self) -> None:
        """
        Stop listening.
        """
        self.node.destroy_subscription(self._subscription)
        self._subscription = None

    def _on_reading(self, message: String) -> None:
        reading = from_json(json.loads(message.data))
        if (
            self._last_stamp is not None
            and reading.stamp - self._last_stamp < READING_PERIOD
        ):
            return
        self._last_stamp = reading.stamp
        self.bank.observe(reading)
