"""
The live estimator and controller must support the same physical flow terms.
"""

from __future__ import annotations

from dataclasses import replace

import numpy
import pytest

from experiments.pouring_model_verification.hypothesis_bank import BankSettings
from experiments.pouring_model_verification.recovery import (
    BankCoupling,
    DrainCorrectionValues,
)
from experiments.real_pouring.live_bank import UnsupportedTrickleFlowError
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    SymbolicFillContext,
)

from ..test_pouring_model_verification.test_hypothesis_bank import (
    FLOWING_CONTENTS,
    GEOMETRY,
    TRICKLING,
    _at_tilt,
    _filter_of,
)
from .test_live_bank import live_bank_tilted

# %% unsupported flow must not disappear during export


def test_live_bank_rejects_learning_flow_its_controller_cannot_represent() -> None:
    """
    Learning an unexported trickle must fail before accepting observations.
    """
    with pytest.raises(UnsupportedTrickleFlowError):
        live_bank_tilted(tilt=1.0, settings=BankSettings(tracks_trickle=True))


def test_live_bank_rejects_fixed_flow_its_controller_cannot_represent() -> None:
    """
    A fixed trickle is incompatible even when no trickle parameter is learned.
    """
    original = live_bank_tilted(tilt=1.0)
    coupling = BankCoupling(bank=replace(original.coupling.bank, base=TRICKLING))
    with pytest.raises(UnsupportedTrickleFlowError):
        replace(original, coupling=coupling)


def test_live_bank_rejects_unrepresented_flow_in_its_reference_model() -> None:
    """
    The reference model must not claim a trickle absent from the built drain.
    """
    original = live_bank_tilted(tilt=1.0)
    with pytest.raises(UnsupportedTrickleFlowError):
        replace(
            original, built=replace(original.built, trickle_rate=TRICKLING.trickle_rate)
        )


# %% identical physical states on both sides of the model boundary


@pytest.mark.parametrize("source_fill", (0.3, 0.8))
@pytest.mark.parametrize("flowing", (False, True))
@pytest.mark.parametrize("onset_offset", (-0.1, 0.1))
def test_exported_bulk_flow_matches_particles_at_the_same_state(
    source_fill: float, flowing: bool, onset_offset: float
) -> None:
    """
    Static and continuing-flow branches agree before temporal smoothing.
    """
    model = _filter_of(FLOWING_CONTENTS)
    source_grains = GEOMETRY.grains_per_fill * source_fill
    tilt = GEOMETRY.onset_tilt(source_grains, FLOWING_CONTENTS) + onset_offset
    model.start(_at_tilt(0.0, tilt))
    model.source[:] = source_grains
    model._loaded[:] = GEOMETRY.grains_per_fill
    model.flowing[:] = flowing
    head = model._flowing_head(tilt, model.source)
    expected = float(
        numpy.average(GEOMETRY.outflow(head, model.parameters), weights=model.weights)
    )
    point = model.controller_point(tilt, trickle=0.0)
    corrections = DrainCorrectionValues.between(point, FLOWING_CONTENTS, cautious=False)
    equation = ArticulatedPouringEquation(
        container_height=GEOMETRY.container_height,
        container_width=2 * GEOMETRY.lip_offset,
        repose_angle=FLOWING_CONTENTS.repose_angle,
        outflow_rate_constant=FLOWING_CONTENTS.outflow_rate_constant,
    )
    context = SymbolicFillContext(
        tilt_expression=Scalar(tilt),
        fill_position=Scalar(source_fill),
        repose_correction=Scalar(corrections.repose_correction),
        drain_scale=Scalar(numpy.exp(corrections.log_drain_scale)),
    )
    actual = -float(equation.symbolic_velocity(context).evaluate()[0])
    assert actual * GEOMETRY.grains_per_fill == pytest.approx(expected)
