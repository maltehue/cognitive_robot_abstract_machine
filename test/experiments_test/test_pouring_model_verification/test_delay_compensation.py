import math

import numpy
import pandas
import pytest

from experiments.pouring_model_verification.inflow_model import (
    FirstOrderInflowModel,
    InflowModels,
    identify_inflow_model,
)
from experiments.pouring_model_verification.pid import (
    NoCompensation,
    PidFillController,
    PidGains,
    RateProjection,
    SmithPrediction,
)
from experiments.pouring_model_verification.records import (
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)

MODEL = FirstOrderInflowModel(gain=20.0, onset_tilt=0.8, time_constant=0.4)
STEP = 0.1

# %% the first-order inflow model


def test_nothing_flows_below_the_onset_tilt():
    assert MODEL.steady_rate(MODEL.onset_tilt - 0.1) == 0.0


def test_held_above_its_onset_the_inflow_settles_at_its_steady_rate():
    times = numpy.arange(0.0, 20.0, STEP)
    tilt = MODEL.onset_tilt + 0.5
    fill = MODEL.simulate(times, numpy.full_like(times, tilt))
    late_rate = (fill[-1] - fill[-11]) / (times[-1] - times[-11])
    assert late_rate == pytest.approx(MODEL.steady_rate(tilt), rel=1e-3)


def _record_from(model: FirstOrderInflowModel, tilts: numpy.ndarray) -> TrialRecord:
    times = numpy.arange(len(tilts)) * 0.05
    grains = model.simulate(times, tilts)
    share_per_grain = 0.001
    samples = pandas.DataFrame(
        {
            Sample.TIME: times,
            Sample.PHASE: Phase.TRANSFER,
            Sample.TILT: tilts,
            Sample.COMMITTED_RECEIVER_FILL: grains * share_per_grain,
            Sample.MEASURED_RECEIVER_FILL: numpy.round(grains) * share_per_grain,
            Sample.GRAINS_IN_RECEIVER: numpy.round(grains),
        }
    )
    return TrialRecord(description={TrialKey.SOURCE: "cup"}, samples=samples)


def test_identification_recovers_the_model_that_produced_the_readings():
    ramps = [
        numpy.concatenate([numpy.linspace(0.3, peak, 60), numpy.full(80, peak)])
        for peak in (1.0, 1.2, 1.4)
    ]
    identified = identify_inflow_model([_record_from(MODEL, tilts) for tilts in ramps])
    assert (
        identified.gain,
        identified.onset_tilt,
        identified.time_constant,
    ) == pytest.approx((MODEL.gain, MODEL.onset_tilt, MODEL.time_constant), rel=0.05)


def test_inflow_models_survive_a_json_round_trip(tmp_path):
    models = InflowModels(models={"cup": MODEL})
    models.write(tmp_path / "models.json")
    assert InflowModels.read(tmp_path / "models.json") == models


# %% compensating a late reading


def test_without_compensation_the_controller_acts_and_stops_on_the_reading():
    compensation = NoCompensation()
    compensation.observe(12.0, at=0.0, tilt=1.0)
    assert (compensation.control_reading(12.0), compensation.stop_reading(12.0)) == (
        12.0,
        12.0,
    )


def test_stopping_early_projects_the_reading_by_its_rate_over_the_horizon():
    horizon, rate = 0.5, 10.0
    compensation = RateProjection(horizon=horizon)
    for index in range(6):
        compensation.observe(rate * index * STEP, at=index * STEP, tilt=1.0)
    reading = rate * 5 * STEP
    assert compensation.control_reading(reading) == reading
    assert compensation.stop_reading(reading) == pytest.approx(reading + rate * horizon)


def test_a_smith_predictor_with_an_exact_model_sees_through_the_delay():
    horizon = 0.5
    compensation = SmithPrediction(model=MODEL, horizon=horizon)
    times = numpy.arange(0.0, 6.0, STEP)
    tilts = numpy.full_like(times, MODEL.onset_tilt + 0.4)
    truth = MODEL.simulate(times, tilts)
    delay_steps = round(horizon / STEP)
    for index, (at, tilt) in enumerate(zip(times, tilts)):
        late = truth[max(0, index - delay_steps)]
        compensation.observe(late, at=at, tilt=tilt)
    assert compensation.control_reading(late) == pytest.approx(truth[-1], rel=1e-2)
    assert compensation.stop_reading(late) == compensation.control_reading(late)


def test_a_compensated_reading_stops_the_pour_when_the_raw_one_would_not():
    gains = PidGains(proportional=0.01, integral=0.0, derivative=0.0, start_tilt=0.5)
    controller = PidFillController(
        gains=gains,
        goal=40.0,
        tolerance=5.0,
        maximum_tilt=2.0,
        compensation=RateProjection(horizon=1.0),
    )
    controller.update(20.0, at=0.0, tilt=1.0)
    controller.update(30.0, at=0.5, tilt=1.0)
    assert controller.reached
