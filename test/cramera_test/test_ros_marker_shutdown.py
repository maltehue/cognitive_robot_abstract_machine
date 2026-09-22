"""
Shared ROS context shutdown is a normal end of marker delivery.
"""

from unittest.mock import Mock

import pytest

from cramera.live import ros_markers


@pytest.mark.skipif(not ros_markers.ROS_AVAILABLE, reason="ROS is not installed")
def test_external_ros_shutdown_ends_marker_delivery() -> None:
    """
    Ctrl+C must not leave an uncaught exception on the marker thread.
    """
    listener = ros_markers.RosMarkerListener(bridge=Mock())
    listener._executor = Mock()
    listener._executor.spin.side_effect = ros_markers.ExternalShutdownException()
    listener.spin_until_context_ends()


@pytest.mark.skipif(not ros_markers.ROS_AVAILABLE, reason="ROS is not installed")
def test_unexpected_marker_executor_errors_remain_visible() -> None:
    """
    Normal shutdown handling must not hide an actual executor failure.
    """
    listener = ros_markers.RosMarkerListener(bridge=Mock())
    listener._executor = Mock()
    failure = RuntimeError("executor failed")
    listener._executor.spin.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        listener.spin_until_context_ends()
    assert caught.value is failure
