"""
Blocking action waits remain independent across clients.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from unittest.mock import AsyncMock, Mock

import pytest
from typing_extensions import TYPE_CHECKING

from giskardpy.middleware.ros2 import ros2_interface
from giskardpy.middleware.ros2.exceptions import ExecutionAbortedException
from giskardpy.middleware.ros2.ros2_interface import MyActionClient

if TYPE_CHECKING:
    from typing_extensions import Callable

# %% mocked action transport


@pytest.fixture
def action_client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], MyActionClient]:
    """
    Create clients whose transport requires no ROS node or action server.
    """
    monkeypatch.setattr(ros2_interface, "ActionClient", Mock())
    monkeypatch.setattr(ros2_interface.rospy, "wait_for_future_to_complete", Mock())
    return lambda: MyActionClient(
        node_handle=Mock(), action_type=Mock(), action_name=__name__
    )


@dataclass
class CoordinatedResult:
    """
    Hold a result until both clients are waiting for their server.
    """

    barrier: Barrier
    """
    Meeting point shared by the independently waiting clients.
    """

    result: object
    """
    Result belonging to this client.
    """

    timeout: float = 5.0
    """
    Maximum wait for the other client, in seconds.
    """

    async def receive(self) -> object:
        """
        Return this client's result once both waits have started.
        """
        await asyncio.to_thread(self.barrier.wait, self.timeout)
        return self.result


# %% concurrent waits


def test_independent_clients_can_wait_concurrently(
    action_client_factory: Callable[[], MyActionClient],
) -> None:
    """
    Each caller receives its own result while another caller is waiting.
    """
    barrier = Barrier(2)
    results = [object(), object()]
    clients = [action_client_factory() for result in results]
    for client, result in zip(clients, results):
        client.get_result = AsyncMock(
            side_effect=CoordinatedResult(barrier, result).receive
        )

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        waits = [executor.submit(client.send_goal, None) for client in clients]
        assert [wait.result() for wait in waits] == results


# %% completed waits


def test_goal_can_be_sent_after_a_failed_wait(
    action_client_factory: Callable[[], MyActionClient],
) -> None:
    """
    A server failure propagates without preventing the next blocking wait.
    """
    client = action_client_factory()
    result = object()
    client.get_result = AsyncMock(side_effect=[ExecutionAbortedException(), result])

    with pytest.raises(ExecutionAbortedException):
        client.send_goal(None)

    assert client.send_goal(None) is result
