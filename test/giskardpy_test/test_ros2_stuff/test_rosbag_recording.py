"""
How a pour's bag is named.
"""

import re
from datetime import datetime

from giskardpy.middleware.ros2.rosbag_recording import (
    BAG_NAME_FORMAT,
    RosbagRecording,
)

# %% bag names

MOMENT = datetime(2026, 10, 24, 13, 39, 18)
"""
When the example bags start recording.
"""

ROS_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""
What a ROS node name may consist of.
"""


def test_an_unlabelled_bag_is_named_after_its_start() -> None:
    assert RosbagRecording().bag_name(MOMENT) == MOMENT.strftime(BAG_NAME_FORMAT)


def test_a_label_follows_the_start(tmp_path) -> None:
    recording = RosbagRecording(label="corn_goal30")

    assert recording.bag_name(MOMENT) == (
        f"{MOMENT.strftime(BAG_NAME_FORMAT)}_corn_goal30"
    )


def test_the_recorder_node_name_is_a_valid_ros_name(tmp_path) -> None:
    """
    Whatever a label holds, the recorder's node, named after the bag, stays a name ROS
    accepts.
    """
    recording = RosbagRecording(directory=tmp_path, label="corn-1.5 deg")
    recording.bag_path = tmp_path / recording.bag_name(MOMENT)

    assert ROS_NAME_PATTERN.fullmatch(recording.recorder_node_name)
