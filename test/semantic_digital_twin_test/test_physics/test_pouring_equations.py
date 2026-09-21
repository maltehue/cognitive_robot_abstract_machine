"""
Geometry tests for the analytic pouring-domain fill equations.
"""

import math

import krrood.symbolic_math.symbolic_math as sm
import pytest
from krrood.adapters.json_serializer import from_json, to_json

from semantic_digital_twin.exceptions import NonPositiveContainerGeometryError
from krrood.symbolic_math.symbolic_math import FloatVariable
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


# %% a drain a controller can correct while it runs


class TestDrainScale:
    """
    A drain carries an optional factor beside its constant, so something watching the
    pour can correct what the model predicts without the equation being rebuilt.
    """

    @staticmethod
    def _equation(**kwargs) -> ArticulatedPouringEquation:
        return ArticulatedPouringEquation(
            container_height=0.1,
            container_width=0.08,
            outflow_rate_constant=1.0,
            **kwargs,
        )

    @staticmethod
    def _context() -> SymbolicFillContext:
        return SymbolicFillContext(sm.Scalar(1.2), sm.Scalar(0.8))

    def test_a_drain_without_a_scale_is_the_calibrated_one(self):
        uncorrected = self._equation().symbolic_velocity(self._context()).evaluate()

        scaled = (
            self._equation(outflow_scale=sm.Scalar(1.0))
            .symbolic_velocity(self._context())
            .evaluate()
        )

        assert scaled == pytest.approx(uncorrected)

    def test_a_scaled_drain_pours_in_proportion_to_its_scale(self):
        scale = 0.25
        uncorrected = self._equation().symbolic_velocity(self._context()).evaluate()

        scaled = (
            self._equation(outflow_scale=sm.Scalar(scale))
            .symbolic_velocity(self._context())
            .evaluate()
        )

        assert scaled == pytest.approx(scale * uncorrected)

    def test_a_free_variable_scale_stays_free_in_the_drain(self):
        """
        The scale is only correctable while a controller runs if the compiled drain
        still reads it, rather than having folded its value in.
        """
        correction = FloatVariable("drain_scale")

        velocity = self._equation(outflow_scale=correction).symbolic_velocity(
            self._context()
        )

        assert correction in velocity.free_variables()

    def test_a_gated_drain_keeps_the_scale_it_was_given(self):
        """
        Coupling a source to a receiver rebuilds its drain as a gated one, which must
        not drop a correction the controller is holding.
        """
        correction = FloatVariable("drain_scale")

        gated = self._equation(outflow_scale=correction).with_gate(sm.Scalar(1.0))

        assert gated.outflow_scale is correction

    def test_the_calibrated_model_is_what_survives_a_round_trip(self):
        """
        A symbol cannot cross a process boundary, so what is written out is the model as
        it was calibrated rather than the correction of the moment.
        """
        correction = FloatVariable("drain_scale")

        restored = from_json(to_json(self._equation(outflow_scale=correction)))

        assert restored.outflow_scale is None
        assert restored.outflow_rate_constant == 1.0


# %% the angle the contents hold before they move

_REPOSE_ANGLE = 0.5
"""
Angle a heap of the contents holds before it slumps, in radians.
"""


class TestReposeAngle:
    """
    Contents that hold a heap do not spill the moment the lip dips: the surface has to
    steepen past the angle they stand at first.
    """

    def _equation(self, repose_angle: float) -> ArticulatedPouringEquation:
        return ArticulatedPouringEquation(
            container_height=_CONTAINER_HEIGHT,
            container_width=_CONTAINER_WIDTH,
            repose_angle=repose_angle,
        )

    def _head(self, equation: ArticulatedPouringEquation, tilt: float, fill: float):
        return equation.head_above_lip(
            SymbolicFillContext(sm.Scalar(tilt), sm.Scalar(fill))
        ).evaluate()[0]

    def test_contents_that_hold_no_heap_are_the_liquid_the_model_had(self) -> None:
        """
        A repose angle of zero is the free surface the equations assumed, so nothing
        that was calibrated against them changes.
        """
        equation = self._equation(repose_angle=0.0)

        assert self._head(equation, _TILT, _FILL) == pytest.approx(
            _expected_head(_CONTAINER_WIDTH / 2)
        )

    def test_a_brimming_container_holds_its_contents_until_it_passes_the_angle(
        self,
    ) -> None:
        """
        The case no lip offset can produce: full to the brim the lip is already at the
        surface, so without a repose angle any tilt at all pours.
        """
        equation = self._equation(repose_angle=_REPOSE_ANGLE)

        assert self._head(equation, _REPOSE_ANGLE - 0.05, fill=1.0) == 0.0
        assert self._head(equation, _REPOSE_ANGLE + 0.05, fill=1.0) > 0.0

    def test_the_settled_fill_is_the_one_the_angle_offsets(self) -> None:
        """
        The fill a tilt drains to is where the surface has fallen back to the angle the
        contents stand at, which is the lip angle less that same angle.
        """
        equation = self._equation(repose_angle=_REPOSE_ANGLE)
        dry_height = equation.lip_offset * math.tan(_TILT - _REPOSE_ANGLE)
        settled_fill = (_CONTAINER_HEIGHT - dry_height) / _CONTAINER_HEIGHT

        assert self._head(equation, _TILT, settled_fill) == pytest.approx(
            0.0, abs=1e-12
        )
        assert self._head(equation, _TILT, settled_fill + 0.01) > 0.0

    def test_the_angle_survives_gating_and_a_round_trip(self) -> None:
        """
        A property of the contents must not be lost when the drain is coupled to a
        receiver or written out.
        """
        equation = self._equation(repose_angle=_REPOSE_ANGLE)

        assert equation.with_gate(sm.Scalar(1.0)).repose_angle == _REPOSE_ANGLE
        assert from_json(to_json(equation)).repose_angle == _REPOSE_ANGLE


# %% the tilt a pour starts at


class TestOnsetTilt:
    """
    The tilt at which a fill starts pouring is what the head is measured against, so
    anything that has to start a pour asks the equation rather than deriving it again.
    """

    def _equation(self, repose_angle: float = 0.0) -> ArticulatedPouringEquation:
        return ArticulatedPouringEquation(
            container_height=_CONTAINER_HEIGHT,
            container_width=_CONTAINER_WIDTH,
            repose_angle=repose_angle,
        )

    def test_the_head_is_zero_below_the_onset_and_positive_above_it(self) -> None:
        equation = self._equation(repose_angle=_REPOSE_ANGLE)
        onset = self._onset(equation, _FILL)

        def head(tilt: float) -> float:
            return equation.head_above_lip(
                SymbolicFillContext(sm.Scalar(tilt), sm.Scalar(_FILL))
            ).evaluate()[0]

        assert head(onset) == pytest.approx(0.0, abs=1e-12)
        assert head(onset - 0.05) == 0.0
        assert head(onset + 0.05) > 0.0

    def test_a_fuller_container_starts_pouring_sooner(self) -> None:
        equation = self._equation()

        assert self._onset(equation, 1.0) < self._onset(equation, _FILL)

    def test_the_onset_carries_the_angle_the_contents_hold(self) -> None:
        heaped = self._onset(self._equation(repose_angle=_REPOSE_ANGLE), _FILL)

        assert heaped == pytest.approx(
            self._onset(self._equation(), _FILL) + _REPOSE_ANGLE
        )

    @staticmethod
    def _onset(equation: ArticulatedPouringEquation, fill: float) -> float:
        return float(equation.onset_tilt(fill).evaluate()[0])
