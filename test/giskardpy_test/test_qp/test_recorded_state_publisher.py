import json
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

pytest.importorskip("rclpy", reason="replaying a world state needs ros")

from krrood.adapters.json_serializer import from_json  # noqa: E402
from semantic_digital_twin.adapters.ros.messages import WorldUpdate  # noqa: E402

from giskardpy.qp.recorded_state_publisher import (  # noqa: E402
    REPLAY_NODE_NAME,
    RecordedWorldStatePublisher,
)

FIRST_IDENTIFIER = "6f1d0e6e-0000-4000-8000-000000000001"
SECOND_IDENTIFIER = "6f1d0e6e-0000-4000-8000-000000000002"


@dataclass
class CollectingPublisher:
    """
    Stands in for a ros publisher and keeps what would have been sent.
    """

    messages: List[str] = field(default_factory=list)

    def publish(self, message) -> None:
        self.messages.append(message.data)


@pytest.fixture()
def state_publisher() -> RecordedWorldStatePublisher:
    return RecordedWorldStatePublisher(publisher=CollectingPublisher())


def _published_update(state_publisher: RecordedWorldStatePublisher) -> WorldUpdate:
    return from_json(json.loads(state_publisher.publisher.messages[-1]))


# %% what is sent


def test_a_replayed_cycle_is_sent_as_a_state_update(state_publisher):
    state_publisher.publish(
        [FIRST_IDENTIFIER, SECOND_IDENTIFIER], np.array([0.25, 0.5])
    )

    update = _published_update(state_publisher)
    assert update.modification_block is None
    assert update.state_update.states == [0.25, 0.5]


def test_the_positions_keep_the_identifiers_of_their_degrees_of_freedom(
    state_publisher,
):
    state_publisher.publish(
        [FIRST_IDENTIFIER, SECOND_IDENTIFIER], np.array([0.25, 0.5])
    )

    update = _published_update(state_publisher)
    assert [str(identifier) for identifier in update.state_update.ids] == [
        FIRST_IDENTIFIER,
        SECOND_IDENTIFIER,
    ]


def test_a_replay_names_itself_as_the_origin(state_publisher):
    """
    A receiver drops the updates it published itself, so a replay has to be a publisher
    of its own for its states to be applied at all.
    """
    state_publisher.publish([FIRST_IDENTIFIER], np.array([0.25]))

    assert _published_update(state_publisher).meta_data.node_name == REPLAY_NODE_NAME


def test_replayed_cycles_are_numbered_in_order(state_publisher):
    """
    Receivers track how far they caught up with a publisher by these numbers, so they
    have to count up rather than repeat.
    """
    for position in (0.1, 0.2, 0.3):
        state_publisher.publish([FIRST_IDENTIFIER], np.array([position]))

    sequence_numbers = [
        from_json(json.loads(message)).sequence_number
        for message in state_publisher.publisher.messages
    ]
    assert sequence_numbers == [1, 2, 3]
