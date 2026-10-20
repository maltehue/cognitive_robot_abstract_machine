"""
Paired stopping metrics distinguish isolated and sustained reversal requests.
"""

from dataclasses import replace
import math

import pytest

from experiments.real_pouring.command_evaluation import CommandComparison
from experiments.real_pouring.pouring_ablation import ForecastModel
from experiments.real_pouring.stopping_ablation import (
    StoppingAnalysis,
    UnpairedCommandsError,
)

# %% paired optimizer plans


@pytest.fixture
def plans() -> list[CommandComparison]:
    """
    Three equally spaced states whose model progressively requests reversal.
    """
    base = CommandComparison(
        0.0,
        ForecastModel.ADAPTED,
        1.0,
        0.0,
        (),
        0.01,
        0.2,
        0.1,
        0.8,
        0.0,
        0.0,
        0.0,
        True,
        (),
    )
    rows = []
    for index, speed in enumerate((1.0, -1.0, -1.0)):
        rows.append(
            replace(
                base,
                time=index * 0.5,
                planned_tilt_velocities=(math.radians(speed),) * 2,
            )
        )
        rows.append(
            replace(
                base,
                time=index * 0.5,
                model=ForecastModel.PACING_ONLY,
                planned_tilt_velocities=(math.radians(3.0),) * 2,
            )
        )
    return rows


def test_stopping_counts_only_model_suppressed_forward_motion(plans):
    """
    The fill constraint must change forward pacing into a hold or return.
    """
    result = StoppingAnalysis(plans).summarize(ForecastModel.ADAPTED)
    assert result.suppressed_forward == 2
    assert result.first_sustained_reversal == 0.5


def test_an_isolated_reversal_is_not_sustained(plans):
    """
    One negative plan is not evidence of a sustained stopping request.
    """
    plans[-2] = replace(plans[-2], planned_tilt_velocities=(math.radians(1.0),) * 2)
    assert (
        StoppingAnalysis(plans)
        .summarize(ForecastModel.ADAPTED)
        .first_sustained_reversal
        is None
    )


def test_unmatched_states_cannot_be_compared(plans):
    """
    A missing reference state must fail rather than silently bias pairing.
    """
    with pytest.raises(UnpairedCommandsError):
        StoppingAnalysis(plans[:-1]).summarize(ForecastModel.ADAPTED)
