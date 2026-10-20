"""
The live controller receives the bank's expected intermittent flow capacity.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import PropertyMock, patch

import pytest

from experiments.pouring_model_verification.hypothesis_bank import BankSettings
from experiments.pouring_model_verification.recovery import BankCoupling
from experiments.real_pouring.live_bank import (
    InvalidStallProbabilityError,
    LIVE_MODEL_TIME_CONSTANT,
)

from .test_live_bank import live_bank_tilted

# %% expected capacity at the live model boundary


@pytest.mark.parametrize(
    "stall_probability", (-0.1, 1.0, 1.1, float("nan"), float("inf"))
)
def test_live_bank_rejects_unrepresentable_stall_probabilities(
    stall_probability: float,
) -> None:
    """
    Logarithmic controller capacity needs a finite positive flowing share.
    """
    with pytest.raises(InvalidStallProbabilityError):
        live_bank_tilted(1.0, BankSettings(stall_probability=stall_probability))


@pytest.mark.parametrize("stall_probability", (0.0, 0.2, 0.6))
def test_live_capacity_matches_expected_fraction_of_physical_flow(
    stall_probability: float,
) -> None:
    """
    A settled export represents mean flow for the configured stall probability.
    """
    bank = live_bank_tilted(1.0, BankSettings(stall_probability=stall_probability))
    physical = bank.coupling.bank.base
    bank._blended(0.0)
    with patch.object(BankCoupling, "point", new_callable=PropertyMock) as point:
        point.return_value = physical
        exported = bank._blended(32 * LIVE_MODEL_TIME_CONSTANT)
    assert exported.outflow_rate_constant == pytest.approx(
        physical.outflow_rate_constant * (1 - stall_probability)
    )


def test_expected_capacity_is_applied_once_to_updated_physical_parameters() -> None:
    """
    Repeated exports preserve physical parameter learning without repeated discounts.
    """
    settings = BankSettings()
    bank = live_bank_tilted(1.0, settings)
    bank._blended(0.0)
    physical = replace(
        bank.coupling.bank.base,
        outflow_rate_constant=2 * bank.coupling.bank.base.outflow_rate_constant,
    )
    with patch.object(BankCoupling, "point", new_callable=PropertyMock) as point:
        point.return_value = physical
        for index in range(1, 4):
            exported = bank._blended(index * 32 * LIVE_MODEL_TIME_CONSTANT)
            assert exported.outflow_rate_constant == pytest.approx(
                physical.outflow_rate_constant * (1 - settings.stall_probability)
            )
    assert (
        bank.coupling.bank.base.outflow_rate_constant
        == physical.outflow_rate_constant / 2
    )
