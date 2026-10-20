"""
Receiver observations identify rate when onset and geometry are held fixed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    DrainRateSpread,
    Hypothesis,
    HypothesisBank,
)
from .test_hypothesis_bank import EXACT_SENSORS, GIVEN, _bank, _poured_readings

# %% rate evidence


@pytest.fixture
def rate_bank() -> HypothesisBank:
    """
    Return a rate-only estimator with an exact receiver observation model.
    """
    bank = _bank(
        drain_rate=DrainRateSpread(GIVEN.outflow_rate_constant, 0.7),
        hypotheses=(Hypothesis.NOMINAL,),
        sensors=replace(
            EXACT_SENSORS,
            source_noise=None,
            conditions_receiver=True,
        ),
    )
    settings = BankSettings(
        rate_noise=0.0, stall_probability=0.0, flow_transition_rate=0.0
    )
    bank.settings = settings
    for model in bank.filters.values():
        model.settings = settings
    return bank


def test_receiver_evidence_learns_rate_without_source_observations(
    rate_bank: HypothesisBank,
) -> None:
    """
    Past receiver increments identify a slower rate with known onset.
    """
    truth = replace(GIVEN, outflow_rate_constant=GIVEN.outflow_rate_constant / 2)
    for index, reading in enumerate(_poured_readings(truth)):
        rate_bank.observe(
            reading if index == 0 else replace(reading, source_grains=None)
        )
    assert rate_bank.filters[Hypothesis.NOMINAL].mean().outflow_rate_constant == (
        pytest.approx(truth.outflow_rate_constant, rel=0.1)
    )


def test_without_receiver_evidence_rate_prior_stays_unchanged(
    rate_bank: HypothesisBank,
) -> None:
    """
    Recorded motion alone cannot identify a drain parameter.
    """
    readings = _poured_readings(GIVEN)
    rate_bank.observe(readings[0])
    initial = rate_bank.filters[Hypothesis.NOMINAL].mean().outflow_rate_constant
    for reading in readings[1:]:
        rate_bank.observe(replace(reading, receiver_grains=None, source_grains=None))
    assert rate_bank.filters[Hypothesis.NOMINAL].mean().outflow_rate_constant == initial
