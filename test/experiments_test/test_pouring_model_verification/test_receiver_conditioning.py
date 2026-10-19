"""
Receiver state correction preserves mass and accounts for delayed uncertainty.
"""

from dataclasses import replace

import numpy
import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    HypothesisFilter,
    Reading,
)

from .test_hypothesis_bank import GIVEN, _filter_of

# %% conditional receiver state


@pytest.fixture
def conditioned_filter() -> HypothesisFilter:
    """
    Particles sharing an uncertain receiver state and conserved source mass.
    """
    hypothesis_filter = _filter_of(GIVEN)
    hypothesis_filter.sensors = replace(
        hypothesis_filter.sensors,
        conditions_receiver=True,
        receiver_delay=0.0,
        fall_time=0.0,
    )
    hypothesis_filter.start(Reading(0.0, 0.0, 20.0, 80.0))
    hypothesis_filter._receiver_variance[:] = 16.0
    hypothesis_filter._receiver_variances[-1][:] = hypothesis_filter._receiver_variance
    return hypothesis_filter


def test_receiver_correction_reduces_state_error_when_particles_agree(
    conditioned_filter: HypothesisFilter,
) -> None:
    """
    Uncertainty within each particle permits correction despite zero ensemble spread.
    """
    reading = Reading(0.0, 0.0, 10.0, None, receiver_noise=2.0)
    before = conditioned_filter.committed.copy()
    variance = conditioned_filter._receiver_variance.copy()
    noise_variance = (
        reading.receiver_noise**2 + conditioned_filter.settings.reading_floor**2
    )
    expected = before + variance / (variance + noise_variance) * (
        reading.receiver_grains - before
    )
    conditioned_filter.condition_receiver(reading)
    numpy.testing.assert_allclose(conditioned_filter.committed, expected)


def test_receiver_correction_conserves_source_receiver_and_spill(
    conditioned_filter: HypothesisFilter,
) -> None:
    """
    Correction transfers inferred material between source and receiver only.
    """
    total = (
        conditioned_filter.source
        + conditioned_filter.committed
        + conditioned_filter.spilled
    )
    conditioned_filter.condition_receiver(
        Reading(0.0, 0.0, 10.0, None, receiver_noise=2.0)
    )
    numpy.testing.assert_allclose(
        conditioned_filter.source
        + conditioned_filter.committed
        + conditioned_filter.spilled,
        total,
    )


def test_delayed_correction_preserves_arrivals_after_the_observation(
    conditioned_filter: HypothesisFilter,
) -> None:
    """
    A past receiver observation corrects its state without erasing newer arrivals.
    """
    conditioned_filter.sensors = replace(conditioned_filter.sensors, receiver_delay=0.5)
    conditioned_filter.source -= 5.0
    conditioned_filter.committed += 5.0
    conditioned_filter._receiver_variance += 4.0
    conditioned_filter._remember(0.5)
    past = conditioned_filter._committed[0].copy()
    late = conditioned_filter.committed - past
    conditioned_filter.condition_receiver(
        Reading(0.5, 0.0, 10.0, None, receiver_noise=2.0)
    )
    numpy.testing.assert_allclose(
        conditioned_filter.committed - conditioned_filter._committed[0], late
    )


def test_prediction_carries_flow_uncertainty_into_receiver_state(
    conditioned_filter: HypothesisFilter,
) -> None:
    """
    Unresolved outflow noise grows state variance before the next measurement.
    """
    conditioned_filter.settings = replace(conditioned_filter.settings, rate_noise=0.5)
    before = conditioned_filter._receiver_variance.copy()
    conditioned_filter.predict(Reading(0.2, 2.0, None, None))
    assert numpy.all(conditioned_filter._receiver_variance > before)


def test_final_fill_spread_includes_unresolved_received_flow(
    conditioned_filter: HypothesisFilter,
) -> None:
    """
    An identical ensemble must not hide uncertainty inside its receiver states.
    """
    assert conditioned_filter.final_fill_spread(0.0) == pytest.approx(
        numpy.sqrt(
            numpy.average(
                conditioned_filter._receiver_variance,
                weights=conditioned_filter.weights,
            )
        )
    )


def test_resampling_keeps_initial_load_with_its_selected_particle(
    conditioned_filter: HypothesisFilter,
) -> None:
    """
    Count predictions retain the initial load belonging to the resampled source.
    """
    chosen = len(conditioned_filter.source) // 2
    expected = numpy.full_like(
        conditioned_filter._loaded, conditioned_filter._loaded[chosen]
    )
    weights = numpy.zeros_like(conditioned_filter.source)
    weights[chosen] = 1.0
    conditioned_filter._resample(weights)
    numpy.testing.assert_array_equal(conditioned_filter._loaded, expected)


def test_receiver_correction_cannot_create_negative_source(
    conditioned_filter: HypothesisFilter,
) -> None:
    """
    An impossible measurement is bounded by the material remaining in each particle.
    """
    conditioned_filter.condition_receiver(
        Reading(0.0, 0.0, 1000.0, None, receiver_noise=0.0)
    )
    assert numpy.all(conditioned_filter.source >= 0.0)
    assert numpy.all(conditioned_filter.committed >= 0.0)
