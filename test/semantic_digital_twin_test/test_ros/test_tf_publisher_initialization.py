"""
State notifications can reach a newly registered TF publisher immediately.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from rclpy.node import Node
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.callbacks.callback import StateChangeCallback

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% callback registration


def test_publisher_accepts_state_notification_during_registration(
    mini_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Registration exposes a publisher whose transform dependencies are ready.
    """
    update_previous_world_state = StateChangeCallback.update_previous_world_state

    def notify_and_update(callback: StateChangeCallback) -> None:
        """
        Deliver a state change as soon as the callback has been registered.
        """
        callback.notify_state_change()
        update_previous_world_state(callback)

    monkeypatch.setattr(
        StateChangeCallback, "update_previous_world_state", notify_and_update
    )
    node = Mock(spec=Node)
    publisher = TFPublisher(node=node, _world=mini_world)

    assert node.create_publisher.return_value.publish.call_count == 2
    publisher.stop()
