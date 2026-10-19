"""
Live and offline pours select matching perception and receiver reporting.
"""

from experiments.real_pouring.pour_perception import PourPerception
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup, ReceiverFillReport
from experiments.real_pouring.wall_fill import WallFillCalibration

# %% cup-specific selection


def test_translucent_corn_pairs_calibration_with_measured_reporting() -> None:
    """
    The supported cup and material enable both halves of measured reporting.
    """
    perception = PourPerception.for_cup(
        ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN
    )
    assert perception.fill_report is ReceiverFillReport.MEASURED
    assert isinstance(perception.wall_calibration, WallFillCalibration)


def test_opaque_cup_preserves_its_counted_reporting() -> None:
    """
    An opaque cup does not receive a through-wall calibration.
    """
    perception = PourPerception.for_cup(ReceivingCup.JEROEN_CUP, PouredMaterial.CORN)
    assert perception.fill_report is ReceivingCup.JEROEN_CUP.fill_report
    assert perception.wall_calibration is None
