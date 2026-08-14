"""
Publishing of a recorded world state, so a running Giskard shows the pose a control
cycle was solved in again.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

import numpy as np
import std_msgs.msg
from rclpy.node import Node

from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    WorldUpdate,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

REPLAY_NODE_NAME = "constraint_inspector"
"""
Name the replayed states are published under, so their origin is obvious on the topic.
"""


class MessagePublisher(Protocol):
    """
    The part of a ROS publisher this module needs.
    """

    def publish(self, message: std_msgs.msg.String) -> None:
        """
        Send one message.
        """


@dataclass
class RecordedWorldStatePublisher:
    """
    Sends the world state of a recorded cycle onto the world synchronization topic.

    Whoever else is on that topic applies the state as if it came from any other
    process, so the world of a running Giskard follows the recording and its
    visualization shows the robot where it was.

    .. warning::
        This overwrites the world state of every process on the topic.  Replay a
        recording against a standalone Giskard, not against one commanding a real robot.
    """

    publisher: MessagePublisher
    """
    Sends the serialized world updates.
    """

    world_id: UUID = field(default_factory=uuid4)
    """
    Identifies this replay as its own publisher, so its updates are not mistaken for
    those of the world that recorded them.
    """

    _sequence_number: int = field(init=False, default=0)
    """
    Counts the published updates, so receivers can tell how far they caught up.
    """

    @classmethod
    def for_node(cls, node: Node) -> RecordedWorldStatePublisher:
        """
        Create a publisher on the topic the world synchronizers share.
        """
        return cls(
            publisher=node.create_publisher(
                std_msgs.msg.String,
                topic=WorldSynchronizer.topic_name,
                qos_profile=10,
            )
        )

    @property
    def meta_data(self) -> MetaData:
        """
        Describes this replay as the origin of the states it sends.
        """
        return MetaData(
            node_name=REPLAY_NODE_NAME,
            process_id=os.getpid(),
            world_id=self.world_id,
        )

    def create_message(
        self, degree_of_freedom_ids: list[str], positions: np.ndarray
    ) -> str:
        """
        Build the serialized world update that sets the given positions.

        :param degree_of_freedom_ids: Identifier of every degree of freedom.
        :param positions: Position of every degree of freedom, in the same order.
        """
        self._sequence_number += 1
        state_update = WorldStateUpdate(
            meta_data=self.meta_data,
            ids=[UUID(identifier) for identifier in degree_of_freedom_ids],
            states=[float(position) for position in positions],
            sequence_number=self._sequence_number,
        )
        update = WorldUpdate(
            meta_data=self.meta_data,
            state_update=state_update,
            sequence_number=self._sequence_number,
        )
        return json.dumps(to_json(update))

    def publish(self, degree_of_freedom_ids: list[str], positions: np.ndarray) -> None:
        """
        Send the given world state to everyone on the synchronization topic.
        """
        self.publisher.publish(
            std_msgs.msg.String(
                data=self.create_message(degree_of_freedom_ids, positions)
            )
        )
