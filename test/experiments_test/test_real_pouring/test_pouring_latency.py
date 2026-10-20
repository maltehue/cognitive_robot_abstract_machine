"""
Fresh observations and consistent world updates during pouring.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from unittest.mock import PropertyMock, patch

import pytest
from rclpy.node import Node
from rclpy.qos import HistoryPolicy

from experiments.pouring_model_verification.hypothesis_bank import (
    FreeParameter,
)
from experiments.pouring_model_verification.recovery import BankCoupling
from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.live_bank_listener import LiveBankListener
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.stream_detection import StreamReading
from pouring_perception import Parameter, StreamDetectorNode
from semantic_digital_twin.callbacks.callback import StateChangeCallback

from .test_live_bank import live_bank_from_count
from .test_live_bank_listener import _detector_reporting

# %% consistent publication


@dataclass(eq=False)
class PourStateObserver(StateChangeCallback):
    """
    Capture the receiver contents visible at each world notification.
    """

    bank: LiveBank
    """
    Pour whose receiver is observed.
    """

    fills: list[float] = field(default_factory=list)
    """
    Receiver fill at each notification.
    """

    def on_state_change(self, **kwargs: object) -> None:
        """
        Record the receiver contents exposed to downstream consumers.
        """
        self.fills.append(self.bank.receiver.fill_level)


def test_bank_publishes_corrections_and_fill_together() -> None:
    """
    A new model must reach observers with its corresponding measured fill.
    """
    bank = live_bank_from_count(tilt=1.0)
    bank.observe(StreamReading(0.0, False, 0.0, None))
    observer = PourStateObserver(_world=bank.world, bank=bank)
    bank.observe(StreamReading(0.1, True, 19.0, None))
    assert observer.fills == [bank.receiver.fill_level]


# %% model response


MODEL_RESPONSE_DEADLINE = 0.3
"""
Seconds available for the live model to follow an updated onset estimate.
"""

REMAINING_MODEL_ERROR = 0.05
"""
Maximum fraction of an onset correction still unapplied at the deadline.
"""


def test_live_model_applies_onset_corrections_within_response_deadline() -> None:
    """
    A corrected onset must affect control promptly during a short burst.
    """
    bank = live_bank_from_count(tilt=1.0)
    initial = bank._blended(0.0)
    target = initial.with_values(
        {
            FreeParameter.REPOSE_ANGLE: initial.repose_angle
            + float(bank.source.pour_tilt_expression.evaluate()[0])
        }
    )
    with patch.object(BankCoupling, "point", new_callable=PropertyMock) as point:
        point.return_value = target
        moved = bank._blended(MODEL_RESPONSE_DEADLINE)
    remaining = (target.repose_angle - moved.repose_angle) / (
        target.repose_angle - initial.repose_angle
    )
    assert 0 < remaining <= REMAINING_MODEL_ERROR


# %% bounded observation queues


@pytest.fixture
def detector_node(rclpy_node: Node) -> Iterator[StreamDetectorNode]:
    """
    Create a detector in the existing ROS test context.
    """
    node = StreamDetectorNode()
    yield node
    node.destroy_node()


def test_camera_subscription_keeps_only_the_latest_image(
    detector_node: StreamDetectorNode,
) -> None:
    """
    An overloaded detector must not work through stale queued camera images.
    """
    topic = detector_node.get_parameter(Parameter.IMAGE_TOPIC).value
    subscription = next(
        subscription
        for subscription in detector_node.subscriptions
        if subscription.topic_name == topic
    )
    assert subscription.qos_profile.history is HistoryPolicy.KEEP_LAST
    assert subscription.qos_profile.depth == 1


def test_detector_publisher_keeps_only_the_latest_reading(
    detector_node: StreamDetectorNode,
) -> None:
    """
    A reader catching up must receive the current camera observation.
    """
    assert detector_node.publisher.qos_profile.history is HistoryPolicy.KEEP_LAST
    assert detector_node.publisher.qos_profile.depth == 1


def test_bank_subscription_keeps_only_the_latest_reading(rclpy_node: Node) -> None:
    """
    Bank computation must not build a queue of obsolete observations.
    """
    _detector_reporting(rclpy_node, PouredMaterial.CORN)
    listener = LiveBankListener(node=rclpy_node, bank=live_bank_from_count(tilt=1.0))
    listener.start()
    subscription = listener._subscription
    listener.stop()
    assert subscription.qos_profile.history is HistoryPolicy.KEEP_LAST
    assert subscription.qos_profile.depth == 1
