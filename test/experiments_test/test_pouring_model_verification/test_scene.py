import math

import pytest

from experiments.pouring_model_verification.parameters import DEMO_PARAMETERS
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.tilt_rig import TiltRig

# %% the rig tips each cup towards the side it pours from


def _outlet_height(rig: TiltRig) -> float:
    return float(
        rig.world.compute_forward_kinematics_np(rig.world.root, rig.cup.spout)[2, 3]
    )


def test_tilting_the_spouted_cup_lowers_its_outlet():
    rig = TiltRig.build(DEMO_PARAMETERS, SourceCup.SPOUTED_CUP)
    upright = _outlet_height(rig)
    rig.set_tilt(math.radians(30))
    rig.world.notify_state_change()
    assert _outlet_height(rig) < upright


def test_the_spouted_cup_pours_at_its_outlet():
    rig = TiltRig.build(DEMO_PARAMETERS, SourceCup.SPOUTED_CUP)
    geometry = SourceCup.SPOUTED_CUP.geometry
    assert rig.source.fill_equation.lip_offset == pytest.approx(geometry.outlet_radius)


def test_a_plain_cup_has_no_spout_to_count():
    rig = TiltRig.build(DEMO_PARAMETERS, SourceCup.CUP)
    assert rig.cup.parts == [rig.cup.root]
