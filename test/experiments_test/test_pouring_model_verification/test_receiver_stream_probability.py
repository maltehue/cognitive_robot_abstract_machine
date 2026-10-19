"""
Analytical flow uncertainty retains the probability of a stalled stream.
"""

from dataclasses import replace

import numpy

from experiments.pouring_model_verification.hypothesis_bank import Reading

from .test_hypothesis_bank import GIVEN, _filter_of

# %% stream likelihood under uncertain outflow


def test_conditional_flow_likelihood_retains_stalled_frames() -> None:
    """
    The marginalized stream likelihood equals its flowing and stalled mixture.
    """
    hypothesis_filter = _filter_of(GIVEN)
    hypothesis_filter.sensors = replace(
        hypothesis_filter.sensors, stream_dropout=0.1, conditions_receiver=True
    )
    hypothesis_filter.settings = replace(
        hypothesis_filter.settings, stall_probability=0.2
    )
    hypothesis_filter.start(Reading(0.0, 0.0, 0.0, 100.0))
    observation = Reading(0.1, 0.0, None, None, stream_seen=False)
    hypothesis_filter._last_interval = observation.time
    conditional_rate = numpy.full_like(hypothesis_filter.source, 10.0)
    flowing_share = 1.0 - hypothesis_filter.settings.stall_probability
    hypothesis_filter._last_rate = conditional_rate * flowing_share
    actual = hypothesis_filter.stream_log_likelihoods(observation)
    hypothesis_filter.sensors = replace(
        hypothesis_filter.sensors, conditions_receiver=False
    )
    hypothesis_filter._last_rate = conditional_rate
    flowing = hypothesis_filter.stream_log_likelihoods(observation)
    hypothesis_filter._last_rate = numpy.zeros_like(conditional_rate)
    stalled = hypothesis_filter.stream_log_likelihoods(observation)
    expected = numpy.logaddexp(
        numpy.log(flowing_share) + flowing,
        numpy.log(hypothesis_filter.settings.stall_probability) + stalled,
    )
    numpy.testing.assert_allclose(actual, expected)
