"""Tests of the analytic fill equations of the pouring domain."""

from __future__ import annotations

import math
from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
import pytest
from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix

from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    FillContext,
    FillEquation,
    InflowEquation,
    SymbolicFillContext,
)
from semantic_digital_twin.physics.exceptions import (
    NonPositiveContainerCapacityError,
    NonPositiveContainerGeometryError,
)

# %% the cup the equations are evaluated for


@dataclass(kw_only=True)
class PouringCup:
    """A rectangular cup and the operating point the equation tests evaluate it
    at."""

    container_height: float = 0.2
    """Height of the cup, in metres."""

    container_width: float = 0.08
    """Width of the cup, in metres."""

    spout_lip_offset: float = 0.15
    """Horizontal distance of a spout outlet from the tilt axis, well outside
    the body."""

    repose_angle: float = 0.5
    """Angle a heap of the contents holds before it slumps, in radians."""

    tilt: float = 0.5
    """Tilt at which the head is evaluated, in radians."""

    fill: float = 0.8
    """Fill level at which the head is evaluated."""

    def equation(
        self, *, lip_offset: float | None = None, repose_angle: float = 0.0
    ) -> ArticulatedPouringEquation:
        """
        :return: The drain of this cup.
        """
        return ArticulatedPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            lip_offset=lip_offset,
            repose_angle=repose_angle,
        )

    def expected_head(
        self, *, lip_offset: float, tilt: float, fill: float, repose_angle: float = 0.0
    ) -> float:
        """
        :return: Closed-form head of the tilted surface above a lip at ``lip_offset``.
        """
        dry_height = self.container_height * (1 - fill)
        lip_distance = math.hypot(dry_height, lip_offset)
        onset = math.atan2(dry_height, lip_offset) + repose_angle
        return max(0.0, lip_distance * math.sin(tilt - onset))


@pytest.fixture
def cup() -> PouringCup:
    """The cup the equation tests pour from."""
    return PouringCup()


def _head(*, equation: ArticulatedPouringEquation, tilt: float, fill: float) -> float:
    """
    :return: The equation's numeric head at the given tilt and fill.
    """
    return equation.head_above_lip(
        context=SymbolicFillContext(tilt=sm.Scalar(tilt), fill=sm.Scalar(fill))
    ).evaluate()[0]


def _drain(*, equation: ArticulatedPouringEquation, tilt: float, fill: float) -> float:
    """
    :return: The equation's numeric fill rate at the given tilt and fill.
    """
    return equation.symbolic_velocity(
        context=SymbolicFillContext(tilt=sm.Scalar(tilt), fill=sm.Scalar(fill))
    ).evaluate()[0]


# %% container geometry guard


class TestContainerGeometryGuard:
    """Non-positive container dimensions must raise instead of producing
    inf/NaN drains."""

    @pytest.mark.parametrize(
        "container_height, container_width",
        [(0.0, 0.08), (0.1, 0.0), (-0.1, 0.08), (0.1, -0.08)],
    )
    def test_pouring_equation_rejects_non_positive_geometry(
        self, *, container_height: float, container_width: float
    ) -> None:
        """A drain of no or negative extent is refused."""
        with pytest.raises(NonPositiveContainerGeometryError):
            ArticulatedPouringEquation(
                container_height=container_height, container_width=container_width
            )

    @pytest.mark.parametrize(
        "container_height, container_width",
        [(0.0, 0.08), (0.1, 0.0)],
    )
    def test_inflow_equation_rejects_non_positive_geometry(
        self, *, container_height: float, container_width: float
    ) -> None:
        """An inflow of no or negative extent is refused."""
        with pytest.raises(NonPositiveContainerGeometryError):
            InflowEquation(
                container_height=container_height, container_width=container_width
            )

    @pytest.mark.parametrize("capacity", [0.0, -1e-3])
    def test_a_non_positive_capacity_is_rejected(self, *, capacity: float) -> None:
        """A container that holds no volume cannot convert a fill rate to a
        volume rate."""
        with pytest.raises(NonPositiveContainerCapacityError) as error_info:
            InflowEquation(
                container_height=0.1, container_width=0.08, capacity=capacity
            )
        assert error_info.value.capacity == capacity

    def test_positive_geometry_is_accepted(self, *, cup: PouringCup) -> None:
        """A drain of positive extent keeps the extents it was given."""
        equation = cup.equation()

        assert equation.container_height == cup.container_height
        assert equation.container_width == cup.container_width


# %% pouring lip offset


class TestPouringLipOffset:
    """The head that drives the pour is measured against the lip, which may sit
    farther from the tilt axis than the body wall, as a spout outlet does."""

    def test_lip_offset_defaults_to_half_the_width(self, *, cup: PouringCup) -> None:
        """Without a spout the lip is the wall, half the width from the
        axis."""
        assert cup.equation().lip_offset == pytest.approx(cup.container_width / 2)

    def test_head_is_measured_against_the_offset_lip(self, *, cup: PouringCup) -> None:
        """A spout far from the axis gains more head at the same tilt than the
        wall."""
        equation = cup.equation(lip_offset=cup.spout_lip_offset)

        head = _head(equation=equation, tilt=cup.tilt, fill=cup.fill)

        assert head == pytest.approx(
            cup.expected_head(
                lip_offset=cup.spout_lip_offset, tilt=cup.tilt, fill=cup.fill
            )
        )
        assert head > cup.expected_head(
            lip_offset=cup.container_width / 2, tilt=cup.tilt, fill=cup.fill
        )

    def test_lip_offset_survives_json_and_gating(self, *, cup: PouringCup) -> None:
        """The lip offset is kept by a round trip, by gating and by dropping
        the gate."""
        equation = cup.equation(lip_offset=cup.spout_lip_offset)

        assert from_json(to_json(equation)).lip_offset == cup.spout_lip_offset
        gated = equation.gate_drain(gate=sm.Scalar(1.0))
        assert gated.lip_offset == cup.spout_lip_offset
        assert gated.drop_gate().lip_offset == cup.spout_lip_offset


# %% contents that stand in a heap


class TestReposeAngle:
    """Contents that hold a heap do not spill the moment the lip dips: the
    surface has to steepen past the angle they stand at first."""

    def test_contents_that_hold_no_heap_pour_as_a_free_surface(
        self, *, cup: PouringCup
    ) -> None:
        """A repose angle of zero is the free liquid surface."""
        equation = cup.equation(repose_angle=0.0)

        assert _head(equation=equation, tilt=cup.tilt, fill=cup.fill) == pytest.approx(
            cup.expected_head(
                lip_offset=cup.container_width / 2, tilt=cup.tilt, fill=cup.fill
            )
        )

    def test_a_brimming_container_holds_its_contents_until_it_passes_the_angle(
        self, *, cup: PouringCup
    ) -> None:
        """Full to the brim the lip is already at the surface, so only the
        repose angle holds the contents back."""
        equation = cup.equation(repose_angle=cup.repose_angle)
        steeper_tilt = cup.repose_angle + 0.05

        assert _head(equation=equation, tilt=cup.repose_angle - 0.05, fill=1.0) == 0.0
        assert _head(equation=equation, tilt=steeper_tilt, fill=1.0) == pytest.approx(
            cup.expected_head(
                lip_offset=equation.lip_offset,
                tilt=steeper_tilt,
                fill=1.0,
                repose_angle=cup.repose_angle,
            )
        )

    def test_the_settled_fill_is_the_one_the_angle_offsets(
        self, *, cup: PouringCup
    ) -> None:
        """A tilt drains to where the surface has fallen back to the angle the
        contents stand at, the lip angle less that same angle."""
        equation = cup.equation(repose_angle=cup.repose_angle)
        dry_height = equation.lip_offset * math.tan(cup.tilt - cup.repose_angle)
        settled_fill = (cup.container_height - dry_height) / cup.container_height
        fuller = settled_fill + 0.01

        assert _head(
            equation=equation, tilt=cup.tilt, fill=settled_fill
        ) == pytest.approx(0.0, abs=1e-12)
        assert _head(equation=equation, tilt=cup.tilt, fill=fuller) == pytest.approx(
            cup.expected_head(
                lip_offset=equation.lip_offset,
                tilt=cup.tilt,
                fill=fuller,
                repose_angle=cup.repose_angle,
            )
        )

    def test_the_angle_survives_gating_and_a_round_trip(
        self, *, cup: PouringCup
    ) -> None:
        """A property of the contents is kept when the drain is gated or
        written out."""
        equation = cup.equation(repose_angle=cup.repose_angle)

        assert equation.gate_drain(gate=sm.Scalar(1.0)).repose_angle == cup.repose_angle
        assert from_json(to_json(equation)).repose_angle == cup.repose_angle


# %% the tilt a pour starts at


class TestOnsetTilt:
    """The tilt at which a fill starts pouring is what the head is measured
    against."""

    def test_the_head_is_zero_below_the_onset_and_grows_above_it(
        self, *, cup: PouringCup
    ) -> None:
        """No head below the onset, the closed-form head above it."""
        equation = cup.equation(repose_angle=cup.repose_angle)
        onset = self._onset(equation=equation, fill=cup.fill)

        assert _head(equation=equation, tilt=onset, fill=cup.fill) == pytest.approx(
            0.0, abs=1e-12
        )
        assert _head(equation=equation, tilt=onset - 0.05, fill=cup.fill) == 0.0
        assert _head(
            equation=equation, tilt=onset + 0.05, fill=cup.fill
        ) == pytest.approx(
            cup.expected_head(
                lip_offset=equation.lip_offset,
                tilt=onset + 0.05,
                fill=cup.fill,
                repose_angle=cup.repose_angle,
            )
        )

    def test_a_fuller_container_starts_pouring_sooner(self, *, cup: PouringCup) -> None:
        """The fuller the container, the lower the tilt it starts pouring
        at."""
        equation = cup.equation()

        assert self._onset(equation=equation, fill=1.0) < self._onset(
            equation=equation, fill=cup.fill
        )

    def test_the_onset_carries_the_angle_the_contents_hold(
        self, *, cup: PouringCup
    ) -> None:
        """Heaped contents start pouring later by their repose angle."""
        heaped = self._onset(
            equation=cup.equation(repose_angle=cup.repose_angle), fill=cup.fill
        )

        assert heaped == pytest.approx(
            self._onset(equation=cup.equation(), fill=cup.fill) + cup.repose_angle
        )

    @staticmethod
    def _onset(*, equation: ArticulatedPouringEquation, fill: float) -> float:
        """
        :return: The equation's numeric onset tilt at the given fill.
        """
        return float(equation.onset_tilt(fill=fill).evaluate()[0])


# %% drain rate


class TestDrainRate:
    """The normalized fill falls at the outflow rate constant times the head
    over the container height."""

    def test_the_drain_follows_the_closed_form(self, *, cup: PouringCup) -> None:
        """
        ``ḣ = −k · d / A`` at a pouring operating point.
        """
        equation = cup.equation()
        head = cup.expected_head(
            lip_offset=equation.lip_offset, tilt=cup.tilt + 0.5, fill=cup.fill
        )

        assert _drain(
            equation=equation, tilt=cup.tilt + 0.5, fill=cup.fill
        ) == pytest.approx(
            -equation.outflow_rate_constant * head / cup.container_height
        )


# %% exit speed


class TestExitSpeed:
    """The stream leaves the lip at the discharge-scaled Torricelli speed of
    the head."""

    def test_the_exit_speed_is_torricelli_of_the_head(self, *, cup: PouringCup) -> None:
        """``C_d · √(2 g h)`` for the head above the lip."""
        equation = cup.equation()
        tilt = cup.tilt + 0.5
        head = cup.expected_head(
            lip_offset=equation.lip_offset, tilt=tilt, fill=cup.fill
        )

        exit_speed = equation.exit_speed(
            context=SymbolicFillContext(tilt=sm.Scalar(tilt), fill=sm.Scalar(cup.fill))
        )

        assert exit_speed.evaluate()[0] == pytest.approx(
            equation.discharge_coefficient
            * math.sqrt(2 * FillEquation.standard_gravity * head)
        )

    def test_the_exit_speed_of_no_head_is_that_of_the_minimum_head(
        self, *, cup: PouringCup
    ) -> None:
        """Below the onset the speed is computed from the minimum head."""
        equation = cup.equation()

        exit_speed = equation.exit_speed(
            context=SymbolicFillContext(tilt=sm.Scalar(0.0), fill=sm.Scalar(cup.fill))
        )

        assert exit_speed.evaluate()[0] == pytest.approx(
            equation.discharge_coefficient
            * math.sqrt(2 * FillEquation.standard_gravity * equation.minimum_head)
        )


# %% partial derivatives of the fill velocity


class TestFillVelocityPartialDerivatives:
    """The partial derivatives of the fill velocity match its finite
    differences."""

    @pytest.mark.parametrize("step", [1e-6])
    def test_the_derivatives_match_central_differences(
        self, *, cup: PouringCup, step: float
    ) -> None:
        """Each derivative equals the central difference of the drain along its
        variable."""
        equation = cup.equation()
        tilt = cup.tilt + 0.5

        derivatives = equation.fill_velocity_partial_derivatives(
            tilt=sm.Scalar(tilt), fill=sm.Scalar(cup.fill)
        )

        tilt_difference = (
            _drain(equation=equation, tilt=tilt + step, fill=cup.fill)
            - _drain(equation=equation, tilt=tilt - step, fill=cup.fill)
        ) / (2 * step)
        fill_difference = (
            _drain(equation=equation, tilt=tilt, fill=cup.fill + step)
            - _drain(equation=equation, tilt=tilt, fill=cup.fill - step)
        ) / (2 * step)
        assert derivatives.tilt.evaluate()[0] == pytest.approx(tilt_difference)
        assert derivatives.fill.evaluate()[0] == pytest.approx(fill_difference)


# %% tilt from the vertical


class TestTiltFromVertical:
    """The tilt is the angle between a container's up axis and the vertical."""

    @pytest.mark.parametrize(
        "roll, pitch, tilt", [(0.0, 0.7, 0.7), (-0.4, 0.0, 0.4), (0.0, 0.0, 0.0)]
    )
    def test_the_tilt_is_the_angle_from_the_vertical(
        self, *, roll: float, pitch: float, tilt: float
    ) -> None:
        """Rolling or pitching a container by an angle tilts it by that
        angle."""
        root_T_container = HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=roll, pitch=pitch
        )

        assert FillContext.tilt_from_vertical(
            root_T_container=root_T_container
        ).evaluate()[0] == pytest.approx(tilt, abs=1e-6)
