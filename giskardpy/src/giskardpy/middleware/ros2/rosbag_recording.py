"""
Recording of a demo run into a timestamped rosbag.
"""

from __future__ import annotations

import atexit
import signal
import subprocess
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import rclpy
from rclpy.context import Context
from rclpy.node import Node

from giskardpy.data_types.exceptions import GiskardException

# %% what is recorded and where

DEFAULT_BAG_DIRECTORY = Path.home() / "pouring_bags"
"""
Directory the pouring demos write their bags into.
"""

BAG_NAME_FORMAT = "%Y-%m-%d_%H-%M-%S"
"""
Timestamp format naming one bag, ordering the bags of a session by name.
"""

RECORDER_NODE_NAME_PREFIX = "pouring_bag_recorder_"
"""
Prefix of the recorder's node name, completed by the bag's timestamp.
"""

GRAPH_WATCHER_NODE_NAME = "pouring_bag_recorder_watcher"
"""
Name of the node that reads the recorder's subscriptions off the ROS graph.
"""

RECORDER_STARTUP_TIMEOUT = 15.0
"""
How long to wait for the recorder to subscribe to every topic, in seconds.
"""

GRAPH_POLL_INTERVAL = 0.1
"""
How often the recorder's subscriptions are looked up while waiting, in seconds.
"""

RECORDER_SHUTDOWN_TIMEOUT = 30.0
"""
How long to wait for the recorder to close its bag after being asked to stop, in
seconds.
"""


class RecordedTopic(StrEnum):
    """
    The topics a pouring run is recorded from.
    """

    LEFT_ARM_JOINT_STATES = "/left_arm/joint_states"
    """
    Measured positions of the arm that carries the source cup.
    """

    TRANSFORMS = "/tf"
    """
    Moving frames, including the ones the pour is evaluated in.
    """

    STATIC_TRANSFORMS = "/tf_static"
    """
    Fixed frames, needed to resolve the moving ones.
    """

    CAMERA_COLOR_IMAGE = "/camera/color/image_raw/compressed"
    """
    Compressed color images of the scene.
    """

    CAMERA_COLOR_INFO = "/camera/color/camera_info"
    """
    Intrinsics of the camera the images come from.
    """


class RosbagCommand(StrEnum):
    """
    The ``ros2 bag record`` invocation and the options the recording sets.
    """

    EXECUTABLE = "ros2"
    """
    The ros2 command line tool.
    """

    SUBCOMMAND = "bag"
    """
    Its bag subcommand.
    """

    ACTION = "record"
    """
    The recording action of the bag subcommand.
    """

    OUTPUT = "--output"
    """
    Option naming the bag to write.
    """

    TOPICS = "--topics"
    """
    Option introducing the topics to record.
    """

    DISABLE_KEYBOARD_CONTROLS = "--disable-keyboard-controls"
    """
    Option that keeps the recorder from reading the terminal the demo prints to.
    """

    NODE_NAME = "--node-name"
    """
    Option naming the recorder's node, under which its subscriptions appear in the ROS
    graph.
    """


@dataclass
class RecordingAlreadyStartedError(GiskardException):
    """
    Raised when a recording that is already writing a bag is started again.
    """

    bag_path: Path
    """
    The bag the recording is writing.
    """

    def error_message(self) -> str:
        return f"Already recording into '{self.bag_path}'."

    def suggest_correction(self) -> str:
        return "Stop the recording before starting a new one."


@dataclass
class RecorderNeverSubscribedError(GiskardException):
    """
    Raised when the recorder has not subscribed to every topic within the startup
    timeout.
    """

    unsubscribed_topics: list[RecordedTopic]
    """
    The topics the recorder was still not subscribed to.
    """

    def error_message(self) -> str:
        return (
            f"The recorder did not subscribe to {self.unsubscribed_topics} within "
            f"{RECORDER_STARTUP_TIMEOUT} seconds."
        )

    def suggest_correction(self) -> str:
        return "Check that these topics are published."


@dataclass
class RecorderExitedError(GiskardException):
    """
    Raised when the recorder process ends before it has subscribed to every topic.
    """

    bag_path: Path
    """
    The bag the recorder was asked to write.
    """

    exit_code: int
    """
    The code the recorder exited with.
    """

    def error_message(self) -> str:
        return (
            f"The recorder for '{self.bag_path}' exited with {self.exit_code} while "
            f"starting up."
        )

    def suggest_correction(self) -> str:
        return "Read the recorder's output to see what it rejected."


@dataclass
class NothingRecordedError(GiskardException):
    """
    Raised when a file is added to a recording that has no bag yet.
    """

    file_name: str
    """
    The file that was to be added.
    """

    def error_message(self) -> str:
        return f"There is no bag to add '{self.file_name}' to."

    def suggest_correction(self) -> str:
        return "Start the recording before adding files to its bag."


# %% reading the ROS graph


@dataclass
class RosGraphWatcher(AbstractContextManager):
    """
    Reads subscriptions off the ROS graph through a node of its own.

    The node lives in its own context, so watching the graph leaves the process's
    default context untouched for whoever initializes ROS next.
    """

    node_name: str = GRAPH_WATCHER_NODE_NAME
    """
    Name of the node that reads the graph.
    """

    _context: Context | None = field(init=False, default=None)
    """
    The context the node lives in, while watching.
    """

    _node: Node | None = field(init=False, default=None)
    """
    The node reading the graph, while watching.
    """

    def __enter__(self) -> RosGraphWatcher:
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(self.node_name, context=self._context)
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self._node.destroy_node()
        rclpy.shutdown(context=self._context)
        self._node = None
        self._context = None

    def subscriber_names(self, topic: str) -> set[str]:
        """
        :return: Names of the nodes currently subscribed to the topic.
        """
        return {
            endpoint.node_name
            for endpoint in self._node.get_subscriptions_info_by_topic(topic)
        }


# %% the recording itself


@dataclass
class RosbagRecording:
    """
    One rosbag written by a ``ros2 bag record`` process for the duration of a demo.

    Starting registers the stop at interpreter exit, so the bag is closed properly even
    when the demo ends on an exception or a keyboard interrupt.
    """

    directory: Path = DEFAULT_BAG_DIRECTORY
    """
    Directory the bag is written into; created if it does not exist.
    """

    topics: tuple[RecordedTopic, ...] = tuple(RecordedTopic)
    """
    The topics to record.
    """

    graph_watcher: RosGraphWatcher = field(default_factory=RosGraphWatcher)
    """
    Tells which topics the recorder has subscribed to.
    """

    bag_path: Path | None = field(init=False, default=None)
    """
    The bag being written, set once the recording has started.
    """

    _recorder: subprocess.Popen | None = field(init=False, default=None)
    """
    The running ``ros2 bag record`` process, or ``None`` while not recording.
    """

    @property
    def recorder_node_name(self) -> str:
        """
        :return: Name of the recorder's node, unique to this bag.
        """
        return RECORDER_NODE_NAME_PREFIX + self.bag_path.name.replace("-", "_")

    @property
    def command(self) -> list[str]:
        """
        :return: The ``ros2 bag record`` invocation writing this bag.
        """
        return [
            RosbagCommand.EXECUTABLE,
            RosbagCommand.SUBCOMMAND,
            RosbagCommand.ACTION,
            RosbagCommand.DISABLE_KEYBOARD_CONTROLS,
            RosbagCommand.NODE_NAME,
            self.recorder_node_name,
            RosbagCommand.OUTPUT,
            str(self.bag_path),
            RosbagCommand.TOPICS,
            *self.topics,
        ]

    def start(self) -> Path:
        """
        Start recording into a bag named after the current time.

        Returns once the recorder is subscribed to every topic, so the messages of
        whatever runs next are captured from the first one.

        :return: The bag being written.
        """
        if self._recorder is not None:
            raise RecordingAlreadyStartedError(self.bag_path)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bag_path = self.directory / datetime.now().strftime(BAG_NAME_FORMAT)
        # A session of its own keeps the recorder from receiving the terminal's Ctrl-C,
        # so the bag is closed by the stop below instead of dying mid-write.
        self._recorder = subprocess.Popen(self.command, start_new_session=True)
        atexit.register(self.stop)
        self._wait_for_subscriptions()
        return self.bag_path

    def stop(self) -> None:
        """
        Ask the recorder to close the bag and wait until it has.

        Does nothing when no recording is running, so stopping explicitly and stopping
        at interpreter exit are the same call.
        """
        if self._recorder is None:
            return
        recorder = self._recorder
        self._recorder = None
        atexit.unregister(self.stop)
        recorder.send_signal(signal.SIGINT)
        recorder.wait(timeout=RECORDER_SHUTDOWN_TIMEOUT)

    def add_file(self, file_name: str, content: str) -> Path:
        """
        Write a file next to the recorded messages, inside the bag.

        :param file_name: Name the file gets in the bag.
        :param content: What to write into it.
        :return: The file written.
        """
        if self.bag_path is None:
            raise NothingRecordedError(file_name)
        file_path = self.bag_path / file_name
        file_path.write_text(content)
        return file_path

    def _wait_for_subscriptions(self) -> None:
        """
        Block until the recorder is subscribed to every topic, or raise if it never
        gets there.
        """
        deadline = time.time() + RECORDER_STARTUP_TIMEOUT
        unsubscribed = list(self.topics)
        with self.graph_watcher as graph:
            while time.time() < deadline:
                exit_code = self._recorder.poll()
                if exit_code is not None:
                    raise RecorderExitedError(self.bag_path, exit_code)
                unsubscribed = [
                    topic
                    for topic in self.topics
                    if self.recorder_node_name not in graph.subscriber_names(topic)
                ]
                if not unsubscribed:
                    return
                time.sleep(GRAPH_POLL_INTERVAL)
        raise RecorderNeverSubscribedError(unsubscribed)
