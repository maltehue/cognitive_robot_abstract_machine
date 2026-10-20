"""
Stream evidence can recover flow within the physical hysteresis interval.
"""

from __future__ import annotations

from dataclasses import replace

import numpy
import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    FlowStateEvidence,
    Hypothesis,
    HypothesisFilter,
    InvalidFlowTransitionRateError,
    Reading,
)
from .test_hypothesis_bank import FLOWING_CONTENTS, GEOMETRY, HYSTERESIS, _filter_of

# %% flow-state evidence


@pytest.fixture
def stopped_filter() -> HypothesisFilter:
    """
    Return identical stopped particles with a stream observation model.
    """
    model = _filter_of(FLOWING_CONTENTS)
    model.sensors = replace(model.sensors, stream_dropout=0.1, conditions_receiver=True)
    tilt = (
        GEOMETRY.onset_tilt(GEOMETRY.grains_per_fill, FLOWING_CONTENTS) - HYSTERESIS / 2
    )
    model.start(Reading(0.0, tilt, None, GEOMETRY.grains_per_fill))
    model.source[:] = GEOMETRY.grains_per_fill
    model._loaded[:] = GEOMETRY.grains_per_fill
    return model


def test_stream_evidence_recovers_flow_without_preexisting_flowing_particles(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    A visible stream can activate the continuing branch below the starting onset.
    """
    tilt = stopped_filter._last_tilt
    stopped_filter.update(Reading(0.1, tilt, None, None, stream_seen=True))
    assert numpy.mean(stopped_filter.flowing) > 0.5
    assert stopped_filter.controller_point(tilt, 0.0).repose_angle == pytest.approx(
        FLOWING_CONTENTS.repose_angle - HYSTERESIS
    )


def test_stream_evidence_cannot_create_flow_below_the_continuing_onset(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    Physically impossible flow stays off even with a false detection.
    """
    stopped_filter.update(Reading(0.1, 0.0, None, None, stream_seen=True))
    numpy.testing.assert_array_equal(stopped_filter.flowing, False)


def test_absent_stream_measurement_preserves_physical_state_transition(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    No observation leaves stopped contents stopped below their starting onset.
    """
    stopped_filter.update(Reading(0.1, stopped_filter._last_tilt, None, None))
    numpy.testing.assert_array_equal(stopped_filter.flowing, False)


def test_state_inference_does_not_retroactively_pour_material(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    Inferring current flow changes future dynamics without fabricating past transfer.
    """
    before = stopped_filter.source.copy()
    stopped_filter.update(
        Reading(0.1, stopped_filter._last_tilt, None, None, stream_seen=True)
    )
    numpy.testing.assert_array_equal(stopped_filter.source, before)
    numpy.testing.assert_array_equal(stopped_filter.committed, 0.0)


def test_flow_state_probability_obeys_bayes_rule() -> None:
    """
    A two-state observation uses the marginal likelihood exactly once.
    """
    evidence = FlowStateEvidence(
        numpy.array([True]),
        numpy.array([0.25]),
        numpy.array([0.8]),
        numpy.array([0.2]),
    )
    numpy.testing.assert_allclose(evidence.likelihood, 0.35)
    numpy.testing.assert_allclose(evidence.posterior, 4 / 7)


def test_stream_likelihood_marginalizes_the_flow_state(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    Particle weights use the same state mixture as conditional mode inference.
    """
    reading = Reading(0.1, stopped_filter._last_tilt, None, None, stream_seen=True)
    stopped_filter.predict(reading)
    evidence = stopped_filter.flow_state_evidence(reading)
    assert isinstance(evidence, FlowStateEvidence)
    numpy.testing.assert_allclose(
        numpy.exp(stopped_filter.stream_log_likelihoods(reading)),
        evidence.likelihood,
    )


def test_no_stream_detection_reduces_flow_probability(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    Missing visible grains favor the stopped alternative without certainty.
    """
    reading = Reading(0.1, stopped_filter._last_tilt, None, None, stream_seen=False)
    stopped_filter.predict(reading)
    evidence = stopped_filter.flow_state_evidence(reading)
    assert isinstance(evidence, FlowStateEvidence)
    assert numpy.all(evidence.posterior < evidence.prior)
    assert numpy.all(evidence.posterior > 0.0)


@pytest.mark.parametrize("transition_rate", (-1.0, numpy.inf, numpy.nan))
def test_invalid_flow_transition_rates_are_rejected(transition_rate: float) -> None:
    """
    Transition probabilities require a finite nonnegative rate.
    """
    with pytest.raises(InvalidFlowTransitionRateError):
        BankSettings(flow_transition_rate=transition_rate)


def test_exported_mode_does_not_confuse_a_flowing_minority_with_onset_uncertainty(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    The low onset quantile is conditional on the more probable physical mode.
    """
    tilt = stopped_filter._last_tilt
    stopped_filter.update(Reading(0.1, tilt, None, None, stream_seen=True))
    stopped_filter.flowing[:] = False
    minority = len(stopped_filter.flowing) // 4
    stopped_filter.flowing[:minority] = True
    assert stopped_filter.controller_point(tilt, 0.0).repose_angle == pytest.approx(
        FLOWING_CONTENTS.repose_angle
    )


def test_stopped_mode_retains_the_existing_catch_all_stream_likelihood(
    stopped_filter: HypothesisFilter,
) -> None:
    """
    State alternatives preserve the catch-all process leakage observation model.
    """
    stopped_filter.hypothesis = Hypothesis.CATCH_ALL
    stopped_filter.settings = replace(stopped_filter.settings, stall_probability=0.2)
    reading = Reading(0.1, stopped_filter._last_tilt, None, None, stream_seen=True)
    stopped_filter.predict(reading)
    evidence = stopped_filter.flow_state_evidence(reading)
    assert isinstance(evidence, FlowStateEvidence)
    stopped_filter.settings = replace(stopped_filter.settings, flow_transition_rate=0.0)
    numpy.testing.assert_allclose(
        evidence.stopped_likelihood,
        numpy.exp(stopped_filter.stream_log_likelihoods(reading)),
    )
