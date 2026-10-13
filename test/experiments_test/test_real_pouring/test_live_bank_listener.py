import json
import threading
import time

import pytest

from std_msgs.msg import String
from std_srvs.srv import Trigger

from experiments.real_pouring.live_bank_listener import (
    DetectorInterface,
    DetectorNotRunningError,
    LiveBankListener,
    READING_PERIOD,
)
from experiments.real_pouring.stream_detection import StreamReading
from krrood.adapters.json_serializer import to_json

from .test_live_bank import live_bank_tilted


def _wait_until(condition, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        time.sleep(0.02)


def test_the_listener_learns_the_background_and_weighs_a_reading_per_period(
    rclpy_node,
):
    background_requests = []

    def learn_background(request, response):
        background_requests.append(request)
        response.success = True
        return response

    rclpy_node.create_service(
        Trigger, DetectorInterface.LEARN_BACKGROUND_SERVICE.value, learn_background
    )
    publisher = rclpy_node.create_publisher(
        String, DetectorInterface.READING_TOPIC.value, 10
    )

    def publish_readings():
        time.sleep(0.3)
        for stamp in (0.0, 0.5 * READING_PERIOD, READING_PERIOD, 2 * READING_PERIOD):
            reading = StreamReading(
                stamp=stamp, stream_seen=False, counted_grains=0.0, crossing=None
            )
            publisher.publish(String(data=json.dumps(to_json(reading))))
            time.sleep(0.05)

    threading.Thread(target=publish_readings).start()
    listener = LiveBankListener(node=rclpy_node, bank=live_bank_tilted(tilt=1.0))
    listener.start()
    assert listener.bank.trace, "start returns once the detector is watching"
    _wait_until(lambda: len(listener.bank.trace) == 3)
    listener.stop()

    assert len(background_requests) == 1
    assert [step.time for step in listener.bank.trace] == [
        0.0,
        READING_PERIOD,
        2 * READING_PERIOD,
    ]


def test_starting_without_a_detector_says_so(rclpy_node):
    listener = LiveBankListener(
        node=rclpy_node, bank=live_bank_tilted(tilt=1.0), service_timeout=0.2
    )
    with pytest.raises(DetectorNotRunningError):
        listener.start()
