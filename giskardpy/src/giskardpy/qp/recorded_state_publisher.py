"""
Publishing of a recorded world state, so a running Giskard shows the pose a control
cycle was solved in again.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import std_msgs.msg
from rclpy.node import Node
from typing_extensions import Self

from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    WorldUpdate,
)
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedom,
    )

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
class ReplayedWorldState:
    """
    The positions of one recorded cycle, under the identifiers of the world they are
    replayed into.
    """

    ids: List[UUID]
    """
    Identifier of every degree of freedom a position is replayed onto.
    """

    positions: List[float]
    """
    The position replayed onto each of those degrees of freedom.
    """

    @property
    def is_empty(self) -> bool:
        """
        Whether the replayed world shares no degree of freedom with the recording, so
        there is nothing to put back.
        """
        return not self.ids


@dataclass
class ReplayTargetDegreesOfFreedom:
    """
    The degrees of freedom of the world a recording is replayed into, by name.

    A world hands out a fresh identifier to every degree of freedom it builds, so the
    identifiers a recording was made under mean nothing to a world started since. The
    name is what both worlds agree on, and is what a recorded position is matched by.
    """

    ids_by_name: Dict[str, UUID]
    """
    The identifier this world knows each degree of freedom by, for the names that belong
    to exactly one of them.
    """

    @classmethod
    def of_world(cls, world: World) -> Self:
        """
        Read the degrees of freedom a recording may be replayed onto off a world.
        """
        return cls.of_degrees_of_freedom(world.degrees_of_freedom)

    @classmethod
    def of_degrees_of_freedom(
        cls, degrees_of_freedom: Iterable[DegreeOfFreedom]
    ) -> Self:
        """
        Index the given degrees of freedom by name, dropping every name that more than
        one of them goes by.
        """
        ids_by_name: Dict[str, UUID] = {}
        ambiguous_names = set()
        for degree_of_freedom in degrees_of_freedom:
            name = str(degree_of_freedom.name)
            if name in ids_by_name:
                ambiguous_names.add(name)
            ids_by_name[name] = degree_of_freedom.id
        return cls(
            ids_by_name={
                name: identifier
                for name, identifier in ids_by_name.items()
                if name not in ambiguous_names
            }
        )

    def identify(self, name: str) -> Optional[UUID]:
        """
        :param name: The name a recorded degree of freedom goes by.
        :return: The identifier this world knows it by, or ``None`` if this world has no
            such degree of freedom or several that share the name.
        """
        return self.ids_by_name.get(name)

    def identify_state(
        self, degree_of_freedom_names: List[str], positions: np.ndarray
    ) -> ReplayedWorldState:
        """
        Keep the recorded positions this world can be put into, in its own terms.

        :param degree_of_freedom_names: Name of every recorded degree of freedom.
        :param positions: Position of each of them, in the same order.
        """
        state = ReplayedWorldState(ids=[], positions=[])
        for name, position in zip(degree_of_freedom_names, positions):
            identifier = self.identify(name)
            if identifier is None:
                continue
            state.ids.append(identifier)
            state.positions.append(float(position))
        return state


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

    target_degrees_of_freedom: ReplayTargetDegreesOfFreedom
    """
    Says which degree of freedom of the replayed world each recorded name stands for.
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
        Create a publisher on the topic the world synchronizers share, matched to the
        world that is currently being served.

        :raises NoServiceFoundError: If no world is being served, leaving nothing to
            replay the recording onto.
        """
        return cls(
            publisher=node.create_publisher(
                std_msgs.msg.String,
                topic=WorldSynchronizer.topic_name,
                qos_profile=10,
            ),
            target_degrees_of_freedom=ReplayTargetDegreesOfFreedom.of_world(
                fetch_world_from_service(node)
            ),
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

    def create_message(self, state: ReplayedWorldState) -> str:
        """
        Build the serialized world update that puts the replayed world into the given
        state.
        """
        self._sequence_number += 1
        state_update = WorldStateUpdate(
            meta_data=self.meta_data,
            ids=state.ids,
            states=state.positions,
            sequence_number=self._sequence_number,
        )
        update = WorldUpdate(
            meta_data=self.meta_data,
            state_update=state_update,
            sequence_number=self._sequence_number,
        )
        return json.dumps(to_json(update))

    def publish(
        self, degree_of_freedom_names: list[str], positions: np.ndarray
    ) -> None:
        """
        Send the recorded positions to everyone on the synchronization topic, as far as
        the replayed world has the degrees of freedom they were recorded for.

        :param degree_of_freedom_names: Name of every recorded degree of freedom.
        :param positions: Position of each of them, in the same order.
        """
        state = self.target_degrees_of_freedom.identify_state(
            degree_of_freedom_names, positions
        )
        if state.is_empty:
            return
        self.publisher.publish(std_msgs.msg.String(data=self.create_message(state)))
