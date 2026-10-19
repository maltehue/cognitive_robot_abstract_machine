"""
Deployment selects the validated wall-evidence scale and requested tolerance.
"""

import math

import pytest

from experiments.real_pouring.pour_perception import PourPerception
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import WALL_STILL_SECONDS
from experiments.real_pouring.wall_fill_validation import (
    WallFillDataset,
    WallFillObservation,
)

from .test_wall_fill_validation import (
    recorded_dataset,
    recorded_observations,
    recording_directory,
)

# %% deployed settings


def test_receiver_noise_is_derived_only_from_development_errors(
    recorded_dataset: WallFillDataset, recorded_observations: list[WallFillObservation]
) -> None:
    """
    The packaged error scales are reproducible without the reserved pours.
    """
    perception = PourPerception.for_cup(
        ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN
    )
    results = recorded_dataset.cross_validate(recorded_observations)
    calibrated = [result.error for result in results if result.calibrated]
    fallback = [result.baseline - result.weighed for result in results]
    assert perception.receiver_evidence.calibrated_noise == pytest.approx(
        math.sqrt(sum(error**2 for error in calibrated) / len(calibrated))
    )
    assert perception.receiver_evidence.geometric_noise == pytest.approx(
        math.sqrt(sum(error**2 for error in fallback) / len(fallback))
    )


def test_receiver_evidence_uses_the_wall_window() -> None:
    """
    The likelihood delay and minimum spacing match the causal observation window.
    """
    perception = PourPerception.for_cup(
        ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN
    )
    assert perception.receiver_evidence.delay == WALL_STILL_SECONDS
    assert perception.receiver_evidence.interval == WALL_STILL_SECONDS


def test_requested_stopping_tolerance_is_one_percentage_point() -> None:
    """
    The shared configuration supplies the user's requested fill tolerance.
    """
    perception = PourPerception.for_cup(
        ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN
    )
    assert perception.fill_tolerance == 0.01
