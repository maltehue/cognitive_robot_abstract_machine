"""
Recorded camera measurements reach the controller without a count override.
"""

import pytest
from pathlib import Path

from experiments.real_pouring.pour_perception import PourPerception
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import StreamReading
from experiments.real_pouring.wall_fill_validation import (
    WallFillDataset,
    WallFillObservation,
)

from .test_live_bank import live_bank_tilted
from .test_wall_fill_validation import (
    recording_directory,
    recorded_dataset,
    recorded_observations,
)

# %% recorded settled windows


def test_packaged_calibration_matches_the_declared_development_pours(
    recorded_dataset: WallFillDataset, recorded_observations: list[WallFillObservation]
) -> None:
    """
    The deployed calibration is reproducible from its declared training partition.
    """
    perception = PourPerception.for_cup(
        ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN
    )
    assert perception.wall_calibration == recorded_dataset.calibrate(
        recorded_observations
    )


def test_eighteen_recorded_wall_readings_reach_the_controller(
    recorded_dataset: WallFillDataset, recording_directory: Path
) -> None:
    """
    Production calibration and bank reporting preserve each recorded final fill.
    """
    perception = PourPerception.for_cup(
        ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN
    )
    for recording in recorded_dataset.read(recording_directory):
        measured = recording.replay(perception.wall_calibration)[-1]
        bank = live_bank_tilted(1.0)
        bank.fill_report = perception.fill_report
        bank.observe(
            StreamReading(0.0, False, bank.loaded_grains * 2, None, heap_fill=measured)
        )
        assert bank.receiver.fill_level == pytest.approx(measured), recording.recording
