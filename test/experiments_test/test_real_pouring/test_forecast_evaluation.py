"""
Future observations are scoring data, never forecast inputs.
"""

from dataclasses import replace

import pytest

from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.forecast_evaluation import ForecastEvaluation
from experiments.real_pouring.live_bank import LiveBankStep
from experiments.real_pouring.pour_replay import ReplayStep
from experiments.real_pouring.pouring_ablation import ForecastModel
from experiments.real_pouring.stream_detection import StreamReading

# %% causal evaluation


@pytest.fixture
def evaluation():
    """
    Uniform recorded motion with continuously calibrated observations.
    """
    steps = []
    for index in range(41):
        time = index / 10
        reading = StreamReading(time, True, time, None, time / 100, True)
        bank = LiveBankStep(
            time,
            1.5,
            True,
            time,
            "H0",
            {},
            DrainCorrectionValues(0.0, 0.0, 0.0),
            time,
            source_grains=50.0,
        )
        steps.append(ReplayStep(index, time, 1.5, reading, bank, time, None, None))
    return ForecastEvaluation(steps, 200.0)


def test_future_measurements_cannot_change_frozen_prediction(evaluation):
    """
    Changing later perception affects targets but cannot alter an earlier model.
    """
    index = 10
    first = evaluation.forecast(index, ForecastModel.ADAPTED).predict(
        evaluation.trajectory, 1.0, 1.5
    )
    evaluation.steps[index + 1 :] = [
        replace(
            step,
            reading=replace(step.reading, heap_fill=0.9),
            bank=replace(
                step.bank,
                source_grains=0.0,
                corrections=DrainCorrectionValues(2.0, 2.0, 2.0),
            ),
        )
        for step in evaluation.steps[index + 1 :]
    ]
    assert (
        evaluation.forecast(index, ForecastModel.ADAPTED).predict(
            evaluation.trajectory, 1.0, 1.5
        )
        == first
    )
    assert evaluation.recent_rate(index) == pytest.approx(0.01)


def test_fallback_boundary_is_not_scored_as_physical_flow(evaluation):
    """
    Estimator switches invalidate the target interval.
    """
    step = evaluation.steps[15]
    evaluation.steps[15] = replace(
        step, reading=replace(step.reading, wall_calibrated=False)
    )
    assert evaluation.target(1.0, 2.0) is None


def test_missing_source_snapshot_is_rejected(evaluation):
    """
    Old replay formats cannot silently supply a made-up source state.
    """
    step = evaluation.steps[10]
    evaluation.steps[10] = replace(step, bank=replace(step.bank, source_grains=None))
    with pytest.raises(ValueError):
        evaluation.forecast(10, ForecastModel.ADAPTED)
