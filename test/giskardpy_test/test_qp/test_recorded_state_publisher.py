import json
from dataclasses import dataclass, field
from typing import List
from uuid import UUID

import numpy as np
import pytest

pytest.importorskip("rclpy", reason="replaying a world state needs ros")

from krrood.adapters.json_serializer import from_json  # noqa: E402
from semantic_digital_twin.adapters.ros.messages import WorldUpdate  # noqa: E402
from semantic_digital_twin.datastructures.prefixed_name import (  # noqa: E402
    PrefixedName,
)
from semantic_digital_twin.spatial_types import Vector3  # noqa: E402
from semantic_digital_twin.world import World  # noqa: E402
from semantic_digital_twin.world_description.connections import (  # noqa: E402
    PrismaticConnection,
)
from semantic_digital_twin.world_description.world_entity import Body  # noqa: E402

from giskardpy.qp.recorded_state_publisher import (  # noqa: E402
    REPLAY_NODE_NAME,
    RecordedWorldStatePublisher,
    ReplayTargetDegreesOfFreedom,
)

FIRST_NAME = "robot/first_joint"
SECOND_NAME = "robot/second_joint"
FIRST_IDENTIFIER = UUID("6f1d0e6e-0000-4000-8000-000000000001")
SECOND_IDENTIFIER = UUID("6f1d0e6e-0000-4000-8000-000000000002")


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
    return RecordedWorldStatePublisher(
        publisher=CollectingPublisher(),
        target_degrees_of_freedom=ReplayTargetDegreesOfFreedom(
            ids_by_name={FIRST_NAME: FIRST_IDENTIFIER, SECOND_NAME: SECOND_IDENTIFIER}
        ),
    )


def _published_update(state_publisher: RecordedWorldStatePublisher) -> WorldUpdate:
    return from_json(json.loads(state_publisher.publisher.messages[-1]))


# %% what is sent


def test_a_replayed_cycle_is_sent_as_a_state_update(state_publisher):
    state_publisher.publish([FIRST_NAME, SECOND_NAME], np.array([0.25, 0.5]))

    update = _published_update(state_publisher)
    assert update.modification_block is None
    assert update.state_update.states == [0.25, 0.5]


def test_a_replay_names_itself_as_the_origin(state_publisher):
    """
    A receiver drops the updates it published itself, so a replay has to be a publisher
    of its own for its states to be applied at all.
    """
    state_publisher.publish([FIRST_NAME], np.array([0.25]))

    assert _published_update(state_publisher).meta_data.node_name == REPLAY_NODE_NAME


def test_replayed_cycles_are_numbered_in_order(state_publisher):
    """
    Receivers track how far they caught up with a publisher by these numbers, so they
    have to count up rather than repeat.
    """
    for position in (0.1, 0.2, 0.3):
        state_publisher.publish([FIRST_NAME], np.array([position]))

    sequence_numbers = [
        from_json(json.loads(message)).sequence_number
        for message in state_publisher.publisher.messages
    ]
    assert sequence_numbers == [1, 2, 3]


# %% matching the recording to the world it is replayed into


def test_a_position_is_sent_under_the_identifier_the_target_world_uses(
    state_publisher,
):
    """
    The recording stores names because the identifiers it was made under are gone the
    moment the world that held them was rebuilt.
    """
    state_publisher.publish([FIRST_NAME, SECOND_NAME], np.array([0.25, 0.5]))

    assert _published_update(state_publisher).state_update.ids == [
        FIRST_IDENTIFIER,
        SECOND_IDENTIFIER,
    ]


def test_a_degree_of_freedom_the_target_world_lacks_is_left_out(state_publisher):
    """
    A world started without the objects the recording was made with keeps working; only
    what it still has is put back.
    """
    state_publisher.publish(
        [FIRST_NAME, "robot/vanished_joint", SECOND_NAME],
        np.array([0.25, 9.9, 0.5]),
    )

    update = _published_update(state_publisher)
    assert update.state_update.ids == [FIRST_IDENTIFIER, SECOND_IDENTIFIER]
    assert update.state_update.states == [0.25, 0.5]


def test_nothing_is_sent_when_the_target_world_shares_no_degree_of_freedom(
    state_publisher,
):
    state_publisher.publish(["other/joint"], np.array([0.25]))

    assert state_publisher.publisher.messages == []


# %% the degrees of freedom of the target world


def test_the_degrees_of_freedom_of_a_world_are_found_by_name():
    world = World(name="replay_target")
    parent_body = Body(name=PrefixedName("parent"))
    child_body = Body(name=PrefixedName("child"))
    with world.modify_world():
        world.add_body(parent_body)
        world.add_body(child_body)
        connection = PrismaticConnection.create_with_dofs(
            world=world,
            parent=parent_body,
            child=child_body,
            axis=Vector3.X(),
            name=PrefixedName("slider"),
        )
        world.add_connection(connection)

    target = ReplayTargetDegreesOfFreedom.of_world(world)

    assert target.identify(str(connection.dof.name)) == connection.dof.id


def test_a_name_the_world_does_not_have_is_not_identified():
    target = ReplayTargetDegreesOfFreedom(ids_by_name={FIRST_NAME: FIRST_IDENTIFIER})

    assert target.identify("robot/missing_joint") is None


@dataclass
class DegreeOfFreedomWithNameAndId:
    """
    Stands in for a degree of freedom, which is looked up by nothing but its name and
    identifier.
    """

    name: str
    id: UUID


def test_a_name_more_than_one_degree_of_freedom_shares_is_not_identified():
    """
    Which of them a recorded position belongs to cannot be told, and putting it on the
    wrong one would move the robot somewhere it never was.
    """
    target = ReplayTargetDegreesOfFreedom.of_degrees_of_freedom(
        [
            DegreeOfFreedomWithNameAndId(FIRST_NAME, FIRST_IDENTIFIER),
            DegreeOfFreedomWithNameAndId(FIRST_NAME, SECOND_IDENTIFIER),
            DegreeOfFreedomWithNameAndId(SECOND_NAME, SECOND_IDENTIFIER),
        ]
    )

    assert target.identify(FIRST_NAME) is None
    assert target.identify(SECOND_NAME) == SECOND_IDENTIFIER
