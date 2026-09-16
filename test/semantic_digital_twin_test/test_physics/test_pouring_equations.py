"""
Geometry tests for the analytic pouring-domain fill equations.
"""

import math

import krrood.symbolic_math.symbolic_math as sm
import pytest
from krrood.adapters.json_serializer import from_json, to_json

from semantic_digital_twin.exceptions import NonPositiveContainerGeometryError
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    InflowEquation,
    SymbolicFillContext,
)

# %% container geometry guard


class TestContainerGeometryGuard:
    """
    Non-positive container dimensions must raise instead of producing inf/NaN drains.
    """

    @pytest.mark.parametrize(
        "container_height, container_width",
        [(0.0, 0.08), (0.1, 0.0), (-0.1, 0.08), (0.1, -0.08)],
    )
    def test_pouring_equation_rejects_non_positive_geometry(
        self, container_height: float, container_width: float
    ):
        with pytest.raises(NonPositiveContainerGeometryError):
            ArticulatedPouringEquation(
                container_height=container_height, container_width=container_width
            )

    @pytest.mark.parametrize(
        "container_height, container_width",
        [(0.0, 0.08), (0.1, 0.0)],
    )
    def test_inflow_equation_rejects_non_positive_geometry(
        self, container_height: float, container_width: float
    ):
        with pytest.raises(NonPositiveContainerGeometryError):
            InflowEquation(
                container_height=container_height, container_width=container_width
            )

    def test_positive_geometry_is_accepted(self):
        equation = ArticulatedPouringEquation(
            container_height=0.1, container_width=0.08
        )
        assert equation.container_height == 0.1
        assert equation.container_width == 0.08


# %% pouring lip offset

_CONTAINER_HEIGHT = 0.2
"""
Height of the container the lip tests pour from, in metres.
"""

_CONTAINER_WIDTH = 0.08
"""
Width of the container the lip tests pour from, in metres.
"""

_SPOUT_LIP_OFFSET = 0.15
"""
Horizontal distance of a spout outlet from the tilt axis, well outside the body.
"""

_TILT = 0.5
"""
Tilt at which the lip tests evaluate the head, in radians.
"""

_FILL = 0.8
"""
Fill level at which the lip tests evaluate the head.
"""


def _expected_head(lip_offset: float) -> float:
    """
    Closed-form head of the tilted liquid surface above a lip at the given horizontal
    distance from the tilt axis.
    """
    dry_height = _CONTAINER_HEIGHT - _FILL * _CONTAINER_HEIGHT
    lip_distance = math.hypot(dry_height, lip_offset)
    lip_angle = math.atan2(dry_height, lip_offset)
    return max(0.0, lip_distance * math.sin(_TILT - lip_angle))


class TestPouringLipOffset:
    """
    The head that drives the pour is measured against the lip, which may sit farther
    from the tilt axis than the body wall, as a spout outlet does.
    """

    def test_lip_offset_defaults_to_half_the_width(self) -> None:
        equation = ArticulatedPouringEquation(
            container_height=_CONTAINER_HEIGHT, container_width=_CONTAINER_WIDTH
        )

        assert equation.lip_offset == pytest.approx(_CONTAINER_WIDTH / 2)

    def test_head_is_measured_against_the_offset_lip(self) -> None:
        equation = ArticulatedPouringEquation(
            container_height=_CONTAINER_HEIGHT,
            container_width=_CONTAINER_WIDTH,
            lip_offset=_SPOUT_LIP_OFFSET,
        )
        context = SymbolicFillContext(sm.Scalar(_TILT), sm.Scalar(_FILL))

        head = equation.head_above_lip(context).evaluate()[0]

        assert head == pytest.approx(_expected_head(_SPOUT_LIP_OFFSET))
        assert head > _expected_head(_CONTAINER_WIDTH / 2)

    def test_lip_offset_survives_json_and_gating(self) -> None:
        equation = ArticulatedPouringEquation(
            container_height=_CONTAINER_HEIGHT,
            container_width=_CONTAINER_WIDTH,
            lip_offset=_SPOUT_LIP_OFFSET,
        )

        assert from_json(to_json(equation)).lip_offset == _SPOUT_LIP_OFFSET
        gated = equation.with_gate(sm.Scalar(1.0))
        assert gated.lip_offset == _SPOUT_LIP_OFFSET
        assert gated.ungated().lip_offset == _SPOUT_LIP_OFFSET
