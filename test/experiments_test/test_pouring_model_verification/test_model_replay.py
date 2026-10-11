import math

import numpy
import pytest

from experiments.pouring_model_verification.model_replay import (
    FillModelReplay,
    RecordedPour,
)
from experiments.pouring_model_verification.parameters import (
    DEMO_PARAMETERS,
    ModelParameter,
)
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.tilt_rig import (
    SAMPLE_PERIOD,
    StepAndHold,
    TiltRig,
)

# %% the replay integrates the drain the way the world does


@pytest.fixture
def rig() -> TiltRig:
    return TiltRig.build(DEMO_PARAMETERS, SourceCup.CUP)


def _tip_in_the_world(rig: TiltRig, profile: StepAndHold) -> RecordedPour:
    """
    Turn the rig's cup along a profile with no physics, only the world integrating its
    fill level, and record that level as if it had been measured.
    """
    period = SAMPLE_PERIOD.total_seconds()
    times, tilts, fills = [0.0], [0.0], [rig.source.fill_level]
    for step in range(1, math.ceil(profile.duration / period) + 1):
        rig.set_tilt(profile.tilt_at(step * period))
        rig.world.step_physics(period)
        times.append(step * period)
        tilts.append(rig.tilt.position)
        fills.append(rig.source.fill_level)
    return RecordedPour(
        time=numpy.array(times),
        tilt=numpy.array(tilts),
        measured_fill=numpy.array(fills),
    )


def test_replay_reproduces_the_fill_the_world_integrates(rig):
    pour = _tip_in_the_world(rig, StepAndHold(tilt=math.radians(85), hold=10.0))
    replay = FillModelReplay(
        equation=rig.source.fill_equation.ungated(), parameters=DEMO_PARAMETERS
    )
    assert replay.predict(pour) == pytest.approx(pour.measured_fill, abs=1e-3)


def test_replay_with_a_scaled_parameter_departs_from_the_world(rig):
    pour = _tip_in_the_world(rig, StepAndHold(tilt=math.radians(85), hold=10.0))
    faster = FillModelReplay(
        equation=rig.source.fill_equation.ungated(),
        parameters=DEMO_PARAMETERS.scaled(ModelParameter.OUTFLOW_RATE_CONSTANT, 4.0),
    )
    assert faster.root_mean_square_error([pour]) > 0.01


# %% a held tilt drains to the fill whose onset it is


def test_held_tilt_settles_where_the_equation_starts_to_pour(rig):
    held_tilt = math.radians(85)
    pour = RecordedPour(
        time=numpy.arange(0.0, 120.0, 0.02),
        tilt=numpy.full(6000, held_tilt),
        measured_fill=numpy.full(6000, rig.source.fill_level),
    )
    equation = DEMO_PARAMETERS.applied_to(rig.source.fill_equation.ungated())
    settled = FillModelReplay(equation=equation, parameters=DEMO_PARAMETERS).predict(
        pour
    )[-1]
    assert float(equation.onset_tilt(settled).evaluate()[0]) == pytest.approx(
        held_tilt, abs=1e-3
    )


# %% onset read off a fill


def test_onset_is_the_tilt_at_which_the_fill_first_falls():
    pour = RecordedPour(
        time=numpy.arange(5.0),
        tilt=numpy.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        measured_fill=numpy.array([0.7, 0.7, 0.7, 0.6, 0.4]),
    )
    assert pour.onset_tilt(pour.measured_fill) == pour.tilt[3]


def test_a_fill_that_never_falls_has_no_onset():
    pour = RecordedPour(
        time=numpy.arange(3.0),
        tilt=numpy.array([0.0, 0.5, 1.0]),
        measured_fill=numpy.full(3, 0.7),
    )
    assert math.isnan(pour.onset_tilt(pour.measured_fill))
