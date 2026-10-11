from dataclasses import dataclass

import pytest

from experiments.pouring_model_verification.transfer_trial import (
    MotionEnding,
    MotionResult,
)
from giskardpy.qp.exceptions import InfeasibleException

# %% how a motion ends


@dataclass
class ExecutorEndingWith:
    """
    Stands in for an executor whose motion ends one way, after a known number of cycles.
    """

    failure: Exception | None
    control_cycles: int = 17

    def tick_until_end(self, timeout: int) -> None:
        if self.failure is not None:
            raise self.failure


@pytest.mark.parametrize(
    "failure, ending",
    [
        (None, MotionEnding.REACHED_GOAL),
        (TimeoutError(), MotionEnding.TIMED_OUT),
        (InfeasibleException(solver_status="-1"), MotionEnding.INFEASIBLE),
    ],
)
def test_a_motion_ending_is_recorded_rather_than_raised(failure, ending):
    executor = ExecutorEndingWith(failure=failure)
    result = MotionResult.of(executor, timeout=10)
    assert result.ending is ending
    assert result.cycles == executor.control_cycles


def test_only_a_motion_that_reached_its_goal_counts_as_reaching_it():
    assert [ending.reached_goal for ending in MotionEnding] == [
        ending is MotionEnding.REACHED_GOAL for ending in MotionEnding
    ]
