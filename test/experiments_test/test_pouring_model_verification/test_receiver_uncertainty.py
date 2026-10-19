"""
Receiver observation quality controls the particle likelihood.
"""

from dataclasses import replace

import numpy

from experiments.pouring_model_verification.hypothesis_bank import Reading

from .test_hypothesis_bank import GIVEN, _filter_of

# %% observation-specific uncertainty


def test_observation_noise_matches_the_equivalent_sensor_configuration() -> None:
    """
    A per-observation uncertainty overrides the fixed receiver uncertainty.
    """
    hypothesis_filter = _filter_of(GIVEN)
    initial = Reading(0.0, 0.0, 0.0, 100.0)
    hypothesis_filter.start(initial)
    observation = Reading(0.0, 0.0, 10.0, None, receiver_noise=8.0)
    actual = hypothesis_filter.log_likelihoods(observation)
    hypothesis_filter.sensors = replace(
        hypothesis_filter.sensors, receiver_noise=observation.receiver_noise
    )
    expected = hypothesis_filter.log_likelihoods(
        replace(observation, receiver_noise=None)
    )
    numpy.testing.assert_array_equal(actual, expected)
