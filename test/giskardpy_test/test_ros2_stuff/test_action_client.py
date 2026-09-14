import threading
from dataclasses import dataclass, field
from typing import Any, List

from action_msgs.msg import GoalStatus

from giskardpy.middleware.ros2.ros2_interface import MyActionClient

# %% mimics

MEETING_TIMEOUT = 3.0
"""
How long a thread waits for the other one to send its goal too.
"""


@dataclass
class CompletedFutureMimic:
    """
    Stands in for the future of a goal that the server accepted right away.
    """

    def done(self) -> bool:
        """
        Report the goal as accepted.
        """
        return True


@dataclass
class SucceededResultMimic:
    """
    Stands in for the result message of a goal that finished successfully.
    """

    status: int = GoalStatus.STATUS_SUCCEEDED
    """
    The status the client reads to decide whether the goal succeeded.
    """


@dataclass
class ClientAnsweredOnlyWhenAnotherSentAGoalToo(MyActionClient):
    """
    An action client whose goal stays in flight until a second client sent one, so that
    two calls of :meth:`MyActionClient.send_goal` overlap.
    """

    meeting: threading.Barrier
    """
    Met inside the goal by every thread that sends one.
    """

    answer: SucceededResultMimic = field(default_factory=SucceededResultMimic)
    """
    The result delivered once every thread has sent its goal.
    """

    def __post_init__(self) -> None:
        self.result = None
        self._current_goal_id = 0

    def send_goal_async(self, goal: Any) -> CompletedFutureMimic:
        """
        Accept the goal and answer it once every other thread has sent one.
        """
        threading.Thread(target=self._answer_after_meeting, daemon=True).start()
        return CompletedFutureMimic()

    def _answer_after_meeting(self) -> None:
        """
        Deliver the result as soon as every thread has reached the meeting.
        """
        self.meeting.wait(timeout=MEETING_TIMEOUT)
        self.result = self.answer


# %% several threads sending goals


def test_two_threads_can_send_a_goal_at_the_same_time():
    meeting = threading.Barrier(parties=2)
    clients = [
        ClientAnsweredOnlyWhenAnotherSentAGoalToo(meeting=meeting) for _ in range(2)
    ]
    results: List[Any] = []
    errors: List[BaseException] = []

    def send_goal(client: ClientAnsweredOnlyWhenAnotherSentAGoalToo) -> None:
        try:
            results.append(client.send_goal(goal=None))
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=send_goal, args=(client,), daemon=True)
        for client in clients
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=MEETING_TIMEOUT * 2)

    assert errors == []
    assert results == [client.answer for client in clients]
