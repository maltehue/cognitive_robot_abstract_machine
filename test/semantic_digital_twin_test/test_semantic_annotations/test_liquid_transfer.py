# %% imports

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json

from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.exceptions import (
    FaucetValveWithoutPositionLimitsError,
    FillLevelAlreadyInitializedError,
    MissingFillEquationError,
    MissingFillLevelLimitsError,
    ReceiverAlreadyCoupledError,
    ReceiverNotInitializedError,
    SourceAlreadyCoupledError,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    DEFAULT_DISCHARGE_COEFFICIENT,
    DEFAULT_GATE_SHARPNESS,
    DEFAULT_POUR_EXIT_SPEED,
    GatedArticulatedPouringEquation,
    GatedInflowEquation,
    InflowEquation,
    MINIMUM_POUR_HEAD,
    STANDARD_GRAVITY,
    SymbolicFillContext,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasFillLevel,
    HasSpout,
    LiquidSource,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Faucet
from semantic_digital_twin.api import (
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    LiquidConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Cylinder, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

_INFLOW_CONTEXT = SymbolicFillContext(sm.Scalar(0.0), sm.Scalar(0.0))
"""
Placeholder context for inflow equations, whose velocity does not depend on the context.
"""


def _evaluated_xyz(spatial: Point3 | Vector3) -> np.ndarray:
    """
    The numeric ``x, y, z`` of a symbolic point or vector at the world's current state.
    """
    return np.array(
        [spatial.x.evaluate()[0], spatial.y.evaluate()[0], spatial.z.evaluate()[0]]
    )


# %% test containers and sources


@dataclass(eq=False)
class _TranslatingContainer(HasFillLevel):
    """
    A pourable container attached to its parent by a single translating DOF.
    """

    @classmethod
    def parent_connection_specification(
        cls,
        axis: Vector3 | None = None,
        dof_limits: DegreeOfFreedomLimits | None = None,
    ) -> PrismaticConnectionSpecification:
        """
        Build the single-degree-of-freedom connection the container hangs from.

        :param axis: Axis of the connection. Defaults to the z axis.
        :param dof_limits: Limits of the generated degree of freedom.
        :return: The connection specification.
        """
        return PrismaticConnectionSpecification(
            axis=axis if axis is not None else Vector3.Z(), dof_limits=dof_limits
        )


@dataclass(eq=False)
class _TiltingContainer(HasFillLevel):
    """
    A pourable container attached to its parent by a single tilting DOF.
    """

    @classmethod
    def parent_connection_specification(
        cls,
        axis: Vector3 | None = None,
        dof_limits: DegreeOfFreedomLimits | None = None,
    ) -> RevoluteConnectionSpecification:
        """
        Build the single-degree-of-freedom connection the container hangs from.

        :param axis: Axis of the connection. Defaults to the z axis.
        :param dof_limits: Limits of the generated degree of freedom.
        :return: The connection specification.
        """
        return RevoluteConnectionSpecification(
            axis=axis if axis is not None else Vector3.Z(), dof_limits=dof_limits
        )


@dataclass
class _StaticLiquidSource(LiquidSource):
    """
    A non-cup liquid source (a faucet stand-in) with a fixed exit point and constant
    stream.
    """

    exit_point: Point3
    """
    World-frame point at which the stream leaves the source.
    """

    volume_rate: float
    """
    Constant volume rate of the stream, in cubic metres per second.
    """

    def outflow_volume_rate(self, world: World) -> sm.Scalar:
        return sm.Scalar(self.volume_rate)

    def liquid_exit_point(self, world: World) -> Point3:
        return self.exit_point

    def liquid_exit_direction(self, world: World) -> Vector3:
        return Vector3.Z()

    @property
    def pour_tilt_expression(self) -> sm.Scalar:
        return sm.Scalar(0.0)

    def couple_drain_to_gate(self, gate: sm.Scalar, world: World) -> None:
        """
        The reservoir is infinite, so being gated does not change the source.
        """

    def validate_can_pour(self) -> None:
        """
        A static source is always ready to pour.
        """


# %% world builders


def _build_world(
    source_class: type[HasFillLevel] = _TranslatingContainer,
    source_axis: Vector3 | None = None,
    source_height: float = 0.3,
    receiver_height: float = 0.2,
    couple: bool = True,
    initialize_receiver_fill: bool = True,
    exit_speed: float = DEFAULT_POUR_EXIT_SPEED,
    height_gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
    overlap_gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
) -> tuple[World, HasFillLevel, _TranslatingContainer]:
    """
    Builds a fixed receiver at the origin and a source held above it on a single DOF.

    :param source_class: The container annotation type used for the source.
    :param source_axis: Axis of the source's single DOF; defaults to translation along
        x.
    :param source_height: Height of the source's origin above the world root, in metres.
    :param receiver_height: Height of the receiver's collision geometry, in metres.
    :param couple: Whether the receiver's inflow is coupled to the source's outflow.
    :param initialize_receiver_fill: Whether the receiver's fill level is initialized.
    :param exit_speed: Nominal exit speed forwarded to the coupling.
    :param height_gate_sharpness: Height-gate steepness forwarded to the coupling.
    :param overlap_gate_sharpness: Overlap-gate steepness forwarded to the coupling.
    :return: The world, the source, and the receiver.
    """
    if source_axis is None:
        source_axis = Vector3(1, 0, 0)
    wide_limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(position=-2.0, velocity=-1.0),
        upper=DerivativeMap(position=2.0, velocity=1.0),
    )
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))

    with world.modify_world():
        receiver = _TranslatingContainer.create_with_new_body_in_world(
            name="receiver",
            world=world,
            parent_connection_specification=_TranslatingContainer.parent_connection_specification(
                axis=Vector3(1, 0, 0),
                dof_limits=wide_limits,
            ),
            scale=Scale(0.1, 0.1, receiver_height),
        )
        source = source_class.create_with_new_body_in_world(
            name="source",
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=source_height
            ),
            parent_connection_specification=source_class.parent_connection_specification(
                axis=source_axis,
                dof_limits=wide_limits,
            ),
            scale=Scale(0.1, 0.1, 0.2),
        )
    if initialize_receiver_fill:
        receiver.initialize_fill_level(world=world, initial_fill=0.0)
    source.initialize_fill_level(world=world, initial_fill=1.0)
    if couple:
        receiver.receive_outflow_from(
            source=source,
            world=world,
            exit_speed=exit_speed,
            height_gate_sharpness=height_gate_sharpness,
            overlap_gate_sharpness=overlap_gate_sharpness,
        )
    return world, source, receiver


def _set_source_offset(world: World, source: HasFillLevel, offset: float) -> None:
    """
    Sets the position of the source's single DOF.
    """
    JointState.from_mapping({source.root.parent_connection: offset}).apply_to(world)


# %% gated inflow equation


class TestGatedInflowEquation:
    """
    Validates the volume-conserving, gated inflow conversion.
    """

    def test_half_cross_section_area_matches_rectangular_area(self):
        """
        The 2-D cup volume is half-width times height.
        """
        equation = InflowEquation(container_height=0.2, container_width=0.08)
        assert equation.half_cross_section_area == pytest.approx(0.04 * 0.2)

    def test_gate_scales_the_inflow_velocity(self):
        """
        A half-open gate halves the resulting fill velocity.
        """
        inflow = sm.Scalar(0.006)
        open_equation = GatedInflowEquation(
            container_height=0.2,
            container_width=0.06,
            inflow=inflow,
            gate=sm.Scalar(1.0),
        )
        half_equation = GatedInflowEquation(
            container_height=0.2,
            container_width=0.06,
            inflow=inflow,
            gate=sm.Scalar(0.5),
        )
        assert half_equation.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[
            0
        ] == pytest.approx(
            0.5 * open_equation.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0]
        )

    @pytest.mark.parametrize(
        "source_size, receiver_size",
        [((0.2, 0.08), (0.2, 0.08)), ((0.2, 0.08), (0.1, 0.06))],
    )
    def test_transfer_is_volume_conserving(self, source_size, receiver_size):
        """
        The volume the receiver gains per second equals the volume the source loses, for
        both equal and unequal cups, while the gate is fully open.
        """
        source_height, source_width = source_size
        receiver_height, receiver_width = receiver_size
        source = ArticulatedPouringEquation(
            container_height=source_height,
            container_width=source_width,
            outflow_rate_constant=1.0,
        )
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        source_normalized_loss = source.symbolic_velocity(
            SymbolicFillContext(tilt, fill)
        )
        source_volume_rate = -source_normalized_loss * source.half_cross_section_area

        receiver = GatedInflowEquation(
            container_height=receiver_height,
            container_width=receiver_width,
            inflow=source_volume_rate,
            gate=sm.Scalar(1.0),
        )
        receiver_volume_gain = (
            receiver.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0]
            * receiver.half_cross_section_area
        )
        source_volume_loss = (
            -source_normalized_loss.evaluate()[0] * source.half_cross_section_area
        )
        assert receiver_volume_gain == pytest.approx(source_volume_loss)


# %% gated source outflow


class TestGatedSourceOutflow:
    """
    Validates the gated source outflow that makes a controlled pour spill-free.
    """

    def test_closed_gate_stops_the_source_draining(self):
        """
        A closed gate zeroes the source outflow, so the source never spills while
        mispositioned.
        """
        equation = GatedArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            outflow_rate_constant=1.0,
            gate=sm.Scalar(0.0),
        )
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        assert equation.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[
            0
        ] == pytest.approx(0.0)

    def test_open_gate_matches_ungated_outflow(self):
        """
        A fully open gate leaves the tilt-driven outflow unchanged.
        """
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        ungated = ArticulatedPouringEquation(
            container_height=0.2, container_width=0.08, outflow_rate_constant=1.0
        )
        gated = GatedArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            outflow_rate_constant=1.0,
            gate=sm.Scalar(1.0),
        )
        assert gated.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[
            0
        ] == pytest.approx(
            ungated.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[0]
        )

    def test_partial_gate_transfer_is_conserved(self):
        """
        At a partly open gate the source's gated loss equals the receiver's gated gain.
        """
        gate = sm.Scalar(0.5)
        tilt, fill = sm.Scalar(1.3), sm.Scalar(0.8)
        ungated = ArticulatedPouringEquation(
            container_height=0.2, container_width=0.08, outflow_rate_constant=1.0
        )
        gated_source = GatedArticulatedPouringEquation(
            container_height=0.2,
            container_width=0.08,
            outflow_rate_constant=1.0,
            gate=gate,
        )
        receiver = GatedInflowEquation(
            container_height=0.2,
            container_width=0.08,
            inflow=-ungated.symbolic_velocity(SymbolicFillContext(tilt, fill))
            * ungated.half_cross_section_area,
            gate=gate,
        )
        source_volume_loss = (
            -gated_source.symbolic_velocity(SymbolicFillContext(tilt, fill)).evaluate()[
                0
            ]
            * gated_source.half_cross_section_area
        )
        receiver_volume_gain = (
            receiver.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0]
            * receiver.half_cross_section_area
        )
        assert receiver_volume_gain == pytest.approx(source_volume_loss)


# %% geometric transfer gate


class TestTransferGate:
    """
    Validates the differentiable geometric gate built by ``receive_outflow_from``.
    """

    def test_gate_is_open_when_source_is_above_receiver(self):
        """
        The gate is essentially fully open when the source is held directly over the
        receiver.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.0)
        gate = receiver.fill_connection.inflow_equation.gate
        assert gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)

    def test_gate_closes_monotonically_with_horizontal_offset(self):
        """
        Moving the source sideways past the receiver opening closes the gate
        monotonically.
        """
        world, source, receiver = _build_world()
        gate = receiver.fill_connection.inflow_equation.gate
        offsets = [0.0, 0.05, 0.1, 0.3]
        values = []
        for offset in offsets:
            _set_source_offset(world, source, offset)
            values.append(gate.evaluate()[0])
        assert all(earlier >= later for earlier, later in zip(values, values[1:]))
        assert values[0] > values[-1]
        assert values[-1] < 0.1

    def test_gate_has_nonzero_gradient_in_transition(self):
        """
        The gate is differentiable: its slope w.r.t.

        the source position is non-zero at the rim.
        """
        world, source, receiver = _build_world()
        gate = receiver.fill_connection.inflow_equation.gate
        source_position = source.root.parent_connection.dof.variables.position
        _set_source_offset(world, source, 0.05)
        gradient = gate.jacobian([source_position])[0, 0].evaluate()[0]
        assert abs(gradient) > 1e-3

    def test_gate_closes_when_source_tilts(self):
        """
        Tilting a source held in place sends the liquid's projectile past the receiver,
        closing the gate — the property that forces the optimizer to reposition the
        gripper while pouring.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        gate = receiver.fill_connection.inflow_equation.gate
        _set_source_offset(world, source, 0.0)
        upright_gate = gate.evaluate()[0]
        _set_source_offset(world, source, 1.2)
        tilted_gate = gate.evaluate()[0]
        assert upright_gate > 0.9
        assert tilted_gate < upright_gate
        assert tilted_gate < 0.5

    def test_gate_closes_when_the_source_lip_is_below_the_receiver_opening(self):
        """
        Liquid enters over the rim, so a receiver whose opening reaches above the
        source's lip cannot be poured into -- even while the lip is still well above the
        receiver's base.
        """
        world, source, receiver = _build_world(receiver_height=1.0)
        _set_source_offset(world, source, 0.0)
        source_lip = source.liquid_exit_point(world).z.evaluate()[0]
        receiver_base = (
            world.compose_forward_kinematics_expression(world.root, receiver.root)
            .to_position()
            .z.evaluate()[0]
        )
        receiver_opening = receiver.opening_point(world).z.evaluate()[0]
        assert (
            receiver_base < source_lip < receiver_opening
        ), "the setup must hold the lip above the receiver's base but below its opening"

        gate = receiver.fill_connection.inflow_equation.gate

        assert gate.evaluate()[0] < 0.1


# %% live exit speed


class TestCurrentOutflowVelocity:
    """
    Validates the discharge-scaled Torricelli exit speed derived from the live pour
    head.
    """

    def _tilted_source(
        self, tilt: float, discharge_coefficient: float = DEFAULT_DISCHARGE_COEFFICIENT
    ) -> tuple[World, HasFillLevel]:
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        source.fill_equation.discharge_coefficient = discharge_coefficient
        _set_source_offset(world, source, tilt)
        return world, source

    def test_matches_discharge_scaled_torricelli(self):
        """
        The exit speed is ``C_d * sqrt(2 g h_head)`` for the current head above the lip.
        """
        world, source = self._tilted_source(1.2, discharge_coefficient=0.4)
        head = source.fill_equation.head_above_lip(source.fill_connection).evaluate()[0]
        velocity = source.current_outflow_velocity(world)
        assert velocity is not None
        assert velocity.evaluate()[0] == pytest.approx(
            0.4 * math.sqrt(2 * STANDARD_GRAVITY * head)
        )

    def test_discharge_coefficient_scales_speed_linearly(self):
        """
        The discharge coefficient scales the exit speed proportionally.
        """
        low_world, low = self._tilted_source(1.2, discharge_coefficient=0.2)
        high_world, high = self._tilted_source(1.2, discharge_coefficient=0.6)
        ratio = (
            high.current_outflow_velocity(high_world).evaluate()[0]
            / low.current_outflow_velocity(low_world).evaluate()[0]
        )
        assert ratio == pytest.approx(3.0)

    def test_more_tilt_pours_faster(self):
        """
        A steeper tilt lifts more liquid above the lip, so the exit speed grows.
        """
        gentle_world, gentle = self._tilted_source(0.6)
        steep_world, steep = self._tilted_source(1.2)
        assert (
            steep.current_outflow_velocity(steep_world).evaluate()[0]
            > gentle.current_outflow_velocity(gentle_world).evaluate()[0]
        )

    def test_head_is_floored_when_barely_pouring(self):
        """
        With no head above the lip the head is floored so the exit-speed gradient stays
        finite.
        """
        world, source = self._tilted_source(0.0, discharge_coefficient=0.4)
        assert source.current_outflow_velocity(world).evaluate()[0] == pytest.approx(
            0.4 * math.sqrt(2 * STANDARD_GRAVITY * MINIMUM_POUR_HEAD)
        )

    def test_source_without_pour_head_has_no_velocity(self):
        """
        A source whose dynamics expose no head reports no exit speed.
        """
        source = _StaticLiquidSource(exit_point=Point3(), volume_rate=0.01)
        assert source.current_outflow_velocity(world=None) is None

    def test_initialize_fill_level_threads_discharge_coefficient(self):
        """
        The discharge coefficient passed to ``initialize_fill_level`` reaches the pour
        model.
        """
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
        with world.modify_world():
            cup = _TiltingContainer.create_with_new_body_in_world(
                name="cup",
                world=world,
                parent_connection_specification=_TiltingContainer.parent_connection_specification(
                    axis=Vector3(0, 1, 0),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap(position=-2.0, velocity=-1.0),
                        upper=DerivativeMap(position=2.0, velocity=1.0),
                    ),
                ),
                scale=Scale(0.1, 0.1, 0.2),
            )
        cup.initialize_fill_level(
            world=world, initial_fill=1.0, discharge_coefficient=0.5
        )
        assert cup.fill_equation.discharge_coefficient == 0.5


# %% liquid exit point


class TestLiquidExitPoint:
    """
    Validates that liquid leaves from the rim edge on the pour side, not the rim centre.
    """

    def _horizontal_half_extent(self, source: HasFillLevel) -> float:
        collision = source.root.collision
        return (collision.max_point.x - collision.min_point.x) / 2

    def _rim_center_world(self, world: World, source: HasFillLevel) -> Point3:
        return (
            world.compose_forward_kinematics_expression(world.root, source.root)
            @ source.rim_point()
        )

    def test_exit_point_at_rim_centre_when_upright(self):
        """
        An upright cup has no pour direction, so the exit point stays at the rim centre.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.0)

        exit_point = source.liquid_exit_point(world)
        rim_center = self._rim_center_world(world, source)

        assert exit_point.x.evaluate()[0] == pytest.approx(
            rim_center.x.evaluate()[0], abs=1e-3
        )
        assert exit_point.y.evaluate()[0] == pytest.approx(
            rim_center.y.evaluate()[0], abs=1e-3
        )

    def test_exit_point_at_rim_edge_along_pour_direction_when_tilted(self):
        """
        A tilted cup pours over the rim edge on the pour side, a full rim radius from
        centre.
        """
        world, source, _ = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 1.2)

        exit_point = source.liquid_exit_point(world)
        rim_center = self._rim_center_world(world, source)
        offset = exit_point - rim_center
        offset_x = offset.x.evaluate()[0]
        offset_y = offset.y.evaluate()[0]
        offset_z = offset.z.evaluate()[0]

        distance = math.sqrt(offset_x**2 + offset_y**2 + offset_z**2)
        assert distance == pytest.approx(self._horizontal_half_extent(source), abs=1e-3)

        pour_direction = source.liquid_exit_direction(world)
        horizontal_alignment = (
            offset_x * pour_direction.x.evaluate()[0]
            + offset_y * pour_direction.y.evaluate()[0]
        )
        assert horizontal_alignment > 0


# %% projectile landing point


class TestProjectileLandingPoint:
    """
    Validates the projectile model that locates where poured liquid lands.
    """

    def test_rim_source_launches_horizontally(self):
        """
        Liquid spilling over a rim has no vertical launch: the exit velocity is the
        horizontal part of the exit direction scaled by the exit speed.
        """
        world, source, _receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.6)
        exit_speed = 0.2

        exit_velocity = _evaluated_xyz(source.liquid_exit_velocity(world, exit_speed))

        exit_direction = _evaluated_xyz(source.liquid_exit_direction(world))
        assert exit_velocity == pytest.approx(
            [exit_speed * exit_direction[0], exit_speed * exit_direction[1], 0.0]
        )

    def test_upright_source_lands_below_its_rim(self):
        """
        With no tilt the liquid has no horizontal velocity, so it lands directly below
        the rim.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.1)
        landing = receiver.projectile_landing_point(source, world, exit_speed=0.2)
        source_rim = (
            world.compose_forward_kinematics_expression(world.root, source.root)
            @ source.rim_point()
        )
        assert landing.x.evaluate()[0] == pytest.approx(
            source_rim.x.evaluate()[0], abs=1e-3
        )
        assert landing.y.evaluate()[0] == pytest.approx(
            source_rim.y.evaluate()[0], abs=1e-3
        )

    def test_tilting_moves_landing_forward(self):
        """
        Tilting the source gives the liquid horizontal velocity, moving the landing
        forward.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.0)
        upright_landing = receiver.projectile_landing_point(
            source, world, exit_speed=0.2
        ).x.evaluate()[0]
        _set_source_offset(world, source, 0.6)
        tilted_landing = receiver.projectile_landing_point(
            source, world, exit_speed=0.2
        ).x.evaluate()[0]
        assert tilted_landing > upright_landing

    def test_higher_source_lands_farther(self):
        """
        A higher source gives the liquid a longer flight time, so it lands farther
        forward.
        """
        low_world, low_source, low_receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            source_height=0.3,
        )
        high_world, high_source, high_receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            source_height=0.7,
        )
        JointState.from_mapping({low_source.root.parent_connection: 0.5}).apply_to(
            low_world
        )
        JointState.from_mapping({high_source.root.parent_connection: 0.5}).apply_to(
            high_world
        )
        low_landing = low_receiver.projectile_landing_point(
            low_source, low_world, exit_speed=0.2
        ).x.evaluate()[0]
        high_landing = high_receiver.projectile_landing_point(
            high_source, high_world, exit_speed=0.2
        ).x.evaluate()[0]
        assert high_landing > low_landing

    def test_arc_terminates_at_the_opening_not_the_base(self):
        """
        The arc ends where the liquid actually arrives -- the receiver's opening.

        Terminating it at the receiver's origin, which lies at the base of the
        container, over-states the fall by the container's height and so over-states the
        throw distance.
        """
        world, source, receiver = _build_world()
        _set_source_offset(world, source, 0.1)

        landing = receiver.projectile_landing_point(source, world, exit_speed=0.2)

        assert landing.z.evaluate()[0] == pytest.approx(
            receiver.opening_point(world).z.evaluate()[0], abs=1e-9
        )

    def test_throw_distance_follows_the_fall_to_the_opening(self):
        """
        The horizontal throw is the exit velocity carried over the time the liquid takes
        to fall from the source's lip to the receiver's opening.
        """
        exit_speed = 0.2
        world, source, receiver = _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )
        _set_source_offset(world, source, 0.6)

        landing = receiver.projectile_landing_point(
            source, world, exit_speed=exit_speed
        )

        exit_point = source.liquid_exit_point(world)
        exit_direction = source.liquid_exit_direction(world)
        fall_height = (
            exit_point.z.evaluate()[0] - receiver.opening_point(world).z.evaluate()[0]
        )
        flight_time = math.sqrt(2 * fall_height / STANDARD_GRAVITY)
        expected_x = (
            exit_point.x.evaluate()[0]
            + exit_speed * exit_direction.x.evaluate()[0] * flight_time
        )
        assert landing.x.evaluate()[0] == pytest.approx(expected_x, abs=1e-9)


# %% coupling guards


class TestReceiveOutflowGuard:
    """
    Validates the guards protecting ``receive_outflow_from`` against illegal couplings.
    """

    def test_raises_when_source_has_no_fill_equation(self):
        """
        Coupling from a source that was never initialized raises a meaningful error.
        """
        source = _TranslatingContainer(
            name=PrefixedName("dry_source"), root=Body(name=PrefixedName("dry_source"))
        )
        receiver = _TranslatingContainer(
            name=PrefixedName("receiver"), root=Body(name=PrefixedName("receiver"))
        )
        with pytest.raises(MissingFillEquationError):
            receiver.receive_outflow_from(source=source, world=World())

    def test_raises_when_source_already_coupled(self):
        """
        Coupling a source whose outflow is already gated onto another receiver raises.
        """
        world, source, receiver = _build_world()
        with world.modify_world():
            second_receiver = _TranslatingContainer.create_with_new_body_in_world(
                name="second_receiver",
                world=world,
                parent_connection_specification=_TranslatingContainer.parent_connection_specification(
                    axis=Vector3(1, 0, 0),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap(position=-2.0, velocity=-1.0),
                        upper=DerivativeMap(position=2.0, velocity=1.0),
                    ),
                ),
                scale=Scale(0.1, 0.1, 0.2),
            )
        second_receiver.initialize_fill_level(world=world, initial_fill=0.0)
        with pytest.raises(SourceAlreadyCoupledError):
            second_receiver.receive_outflow_from(source=source, world=world)

    def test_raises_when_receiver_has_no_fill_connection(self):
        """
        Coupling into a receiver that was never initialized blames the receiver, not the
        source.
        """
        world, source, receiver = _build_world(
            couple=False, initialize_receiver_fill=False
        )
        with pytest.raises(ReceiverNotInitializedError) as error_info:
            receiver.receive_outflow_from(source=source, world=world)
        assert error_info.value.receiver is receiver

    def test_raises_when_receiver_already_coupled(self):
        """
        Coupling a second source into an already-coupled receiver raises instead of
        silently overwriting the first source's transfer.
        """
        world, source, receiver = _build_world()
        with world.modify_world():
            second_source = _TranslatingContainer.create_with_new_body_in_world(
                name="second_source",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.3),
                parent_connection_specification=_TranslatingContainer.parent_connection_specification(
                    axis=Vector3(1, 0, 0),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap(position=-2.0, velocity=-1.0),
                        upper=DerivativeMap(position=2.0, velocity=1.0),
                    ),
                ),
                scale=Scale(0.1, 0.1, 0.2),
            )
        second_source.initialize_fill_level(world=world, initial_fill=1.0)
        first_inflow_equation = receiver.fill_connection.inflow_equation
        with pytest.raises(ReceiverAlreadyCoupledError) as error_info:
            receiver.receive_outflow_from(source=second_source, world=world)
        assert error_info.value.receiver is receiver
        assert receiver.fill_connection.inflow_equation is first_inflow_equation


# %% fill-level initialization guard


class TestFillLevelInitializationGuard:
    """
    Validates that a container's fill level can only be initialized once.
    """

    def test_second_initialization_raises(self):
        """
        A second ``initialize_fill_level`` call raises instead of stacking a second
        phantom body and fill connection onto the container.
        """
        world, source, receiver = _build_world(couple=False)
        first_fill_connection = source.fill_connection
        with pytest.raises(FillLevelAlreadyInitializedError) as error_info:
            source.initialize_fill_level(world=world, initial_fill=1.0)
        assert error_info.value.container is source
        assert source.fill_connection is first_fill_connection


# %% coupling reconstruction


class TestCouplingReconstruction:
    """
    Validates that a transfer coupling is transmitted as a serializable parametric
    descriptor and rebuilt against the receiving world.

    The gate and inflow of a coupling are symbolic expressions bound to the world they were built
    in, so they cannot be serialized to another process. ``receive_outflow_from`` therefore records
    a :class:`~semantic_digital_twin.world_description.connections.LiquidTransferCoupling` descriptor
    on the fill connection, which survives synchronization and lets the receiving world rebuild the
    symbolic coupling locally.
    """

    def _coupled_world(self):
        return _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )

    def test_receive_outflow_records_serializable_descriptor(self):
        """
        The coupling descriptor names the source and survives a JSON round trip.
        """
        world, source, receiver = self._coupled_world()
        coupling = receiver.inflow_coupling
        assert coupling is not None
        assert coupling.source_id == source.id

        restored = from_json(to_json(coupling))
        assert restored.source_id == source.id
        assert restored.exit_speed == coupling.exit_speed
        assert restored.height_gate_sharpness == coupling.height_gate_sharpness
        assert restored.overlap_gate_sharpness == coupling.overlap_gate_sharpness

    def test_rebuild_reconstructs_inflow_when_symbolic_state_absent(self):
        """
        Given the state a synchronized world holds - the descriptor present but the symbolic inflow
        side effect absent - the receiver rebuilds a working, world-bound inflow equation.
        """
        world, source, receiver = self._coupled_world()
        receiver.fill_connection.inflow_equation = None

        receiver.ensure_inflow_coupling(world)

        inflow_equation = receiver.fill_connection.inflow_equation
        assert inflow_equation is not None
        # The gate is a symbolic function of the source's DOF in this world; evaluating it proves
        # the rebuilt coupling is bound to this world's symbols, not the ones it was first built in.
        JointState.from_mapping({source.root.parent_connection: 0.0}).apply_to(world)
        assert inflow_equation.gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)
        # The inflow tracks the source's outflow: zero while upright, positive once the source tilts.
        assert inflow_equation.inflow.evaluate()[0] == pytest.approx(0.0)
        JointState.from_mapping({source.root.parent_connection: 1.0}).apply_to(world)
        assert inflow_equation.inflow.evaluate()[0] > 0.0

    def test_rebuild_regates_source_outflow(self):
        """
        Rebuilding re-establishes the source's gated outflow, keeping the transfer
        spill-free.
        """
        world, source, receiver = self._coupled_world()
        receiver.fill_connection.inflow_equation = None

        receiver.ensure_inflow_coupling(world)

        source_outflow = source.fill_connection.outflow_equation
        assert isinstance(source_outflow, GatedArticulatedPouringEquation)
        JointState.from_mapping({source.root.parent_connection: 0.0}).apply_to(world)
        assert source_outflow.gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)

    def test_rebuild_is_noop_when_inflow_already_present(self):
        """
        A receiver already carrying a symbolic inflow equation is left untouched.
        """
        world, source, receiver = self._coupled_world()
        original_inflow = receiver.fill_connection.inflow_equation
        assert original_inflow is not None

        receiver.ensure_inflow_coupling(world)

        assert receiver.fill_connection.inflow_equation is original_inflow

    def test_rebuild_rebinds_fill_connection_to_the_target_world(self):
        """
        An annotation holding a fill connection detached from its world is re-pointed at
        the connection resident in that world, so the coupling evaluates against the
        world's own symbols.

        Synchronizing an annotation deserializes its fill connection by value, so the
        synchronized annotation starts with a copy no world holds.
        """
        world, source, receiver = self._coupled_world()
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        detached_connection = LiquidConnection.from_json(
            receiver.fill_connection.to_json(), **tracker.create_kwargs()
        )
        assert detached_connection._world is not world
        synchronized_receiver = HasFillLevel(
            name=receiver.name,
            root=receiver.root,
            fill_connection=detached_connection,
            inflow_coupling=receiver.inflow_coupling,
        )

        synchronized_receiver.ensure_inflow_coupling(world)

        assert synchronized_receiver.fill_connection is receiver.fill_connection
        assert synchronized_receiver.fill_connection._world is world


# %% coupling model switch


class TestCouplingModelSwitch:
    """
    Validates that switching the source's drain model against a live coupling takes
    effect.

    A client switches models by publishing a new (ungated) source fill equation; the
    symbolic coupling another process built earlier is then stale and must be rebuilt
    from the new equation instead of being kept because it merely exists.
    """

    def _coupled_world(self):
        return _build_world(
            source_class=_TiltingContainer, source_axis=Vector3(0, 1, 0)
        )

    def _drain_swapped_like_a_synchronized_diff(
        self, source: HasFillLevel
    ) -> ArticulatedPouringEquation:
        """
        Replace the source's fill equation the way a synchronized attribute diff does:

        a plain attribute assignment of a fresh ungated equation, doubling the outflow
        rate so the rebuilt coupling is distinguishable from the stale one.
        """
        swapped = ArticulatedPouringEquation(
            container_height=source.fill_equation.container_height,
            container_width=source.fill_equation.container_width,
            outflow_rate_constant=source.fill_equation.outflow_rate_constant * 2,
        )
        source.fill_equation = swapped
        return swapped

    def test_ensure_rebuilds_when_synchronized_source_equation_changed(self):
        """
        A synchronized source-equation swap marks the coupling stale: the next ensure
        call rebuilds the inflow and re-gates the drain from the new equation.
        """
        world, source, receiver = self._coupled_world()
        stale_inflow = receiver.fill_connection.inflow_equation
        swapped = self._drain_swapped_like_a_synchronized_diff(source)

        receiver.ensure_inflow_coupling(world)

        assert receiver.fill_connection.inflow_equation is not stale_inflow
        regated_drain = source.fill_equation
        assert isinstance(regated_drain, GatedArticulatedPouringEquation)
        assert regated_drain.outflow_rate_constant == swapped.outflow_rate_constant
        assert source.fill_connection.outflow_equation is regated_drain

    def test_rebuilt_inflow_reflects_the_new_drain(self):
        """
        The rebuilt inflow is derived from the swapped drain: doubling the source's
        outflow rate constant doubles the inflow volume rate at the same pose.
        """
        world, source, receiver = self._coupled_world()
        stale_inflow = receiver.fill_connection.inflow_equation
        self._drain_swapped_like_a_synchronized_diff(source)

        receiver.ensure_inflow_coupling(world)

        JointState.from_mapping({source.root.parent_connection: 1.0}).apply_to(world)
        rebuilt_inflow = receiver.fill_connection.inflow_equation
        assert rebuilt_inflow.inflow.evaluate()[0] == pytest.approx(
            2 * stale_inflow.inflow.evaluate()[0]
        )

    def test_ensure_is_stable_after_a_rebuild(self):
        """
        A rebuild settles: ensuring again without another synchronized swap is a no-op.
        """
        world, source, receiver = self._coupled_world()
        self._drain_swapped_like_a_synchronized_diff(source)
        receiver.ensure_inflow_coupling(world)
        rebuilt_inflow = receiver.fill_connection.inflow_equation

        receiver.ensure_inflow_coupling(world)

        assert receiver.fill_connection.inflow_equation is rebuilt_inflow

    def test_recouple_replaces_drain_and_coupling_of_a_coupled_source(self):
        """
        ``recouple_outflow_from`` accepts an already-coupled source, replacing its gated
        drain with the given equation and re-establishing the coupling from it — the
        client-side half of a live model switch.
        """
        world, source, receiver = self._coupled_world()
        previous_inflow = receiver.fill_connection.inflow_equation
        replacement = ArticulatedPouringEquation(
            container_height=source.fill_equation.container_height,
            container_width=source.fill_equation.container_width,
            outflow_rate_constant=source.fill_equation.outflow_rate_constant * 2,
        )

        receiver.recouple_outflow_from(
            source=source, world=world, fill_equation=replacement
        )

        regated_drain = source.fill_equation
        assert isinstance(regated_drain, GatedArticulatedPouringEquation)
        assert regated_drain.outflow_rate_constant == replacement.outflow_rate_constant
        assert receiver.fill_connection.inflow_equation is not previous_inflow
        assert receiver.inflow_coupling is not None

    def test_recouple_preserves_coupling_parameters(self):
        """
        Recoupling forwards the existing coupling's exit speed and gate sharpnesses
        instead of resetting them to the defaults.
        """
        world, source, receiver = _build_world(
            source_class=_TiltingContainer,
            source_axis=Vector3(0, 1, 0),
            exit_speed=0.7,
            height_gate_sharpness=42.0,
            overlap_gate_sharpness=17.0,
        )
        replacement = ArticulatedPouringEquation(
            container_height=source.fill_equation.container_height,
            container_width=source.fill_equation.container_width,
            outflow_rate_constant=source.fill_equation.outflow_rate_constant,
        )

        receiver.recouple_outflow_from(
            source=source, world=world, fill_equation=replacement
        )

        coupling = receiver.inflow_coupling
        assert coupling.exit_speed == 0.7
        assert coupling.height_gate_sharpness == 42.0
        assert coupling.overlap_gate_sharpness == 17.0
        assert receiver.fill_connection.inflow_equation.exit_speed == 0.7


# %% non-cup liquid source


class TestNonCupLiquidSource:
    """
    A receiver fills from a :class:`LiquidSource` that is not a :class:`HasFillLevel`
    cup.
    """

    def test_receiver_fills_from_static_source(self):
        """
        Coupling a static faucet-like source (no fill level, no tilt) opens the gate and
        gives the receiver a positive inflow — a transfer the cup-only API could not
        express.
        """
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
        with world.modify_world():
            receiver = _TranslatingContainer.create_with_new_body_in_world(
                name="receiver",
                world=world,
                parent_connection_specification=_TranslatingContainer.parent_connection_specification(
                    axis=Vector3(1, 0, 0),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap(position=-2.0, velocity=-1.0),
                        upper=DerivativeMap(position=2.0, velocity=1.0),
                    ),
                ),
                scale=Scale(0.1, 0.1, 0.2),
            )
        receiver.initialize_fill_level(world=world, initial_fill=0.0)

        source = _StaticLiquidSource(
            exit_point=Point3(x=0.0, y=0.0, z=0.5, reference_frame=world.root),
            volume_rate=0.001,
        )
        receiver.receive_outflow_from(source=source, world=world)

        inflow_equation = receiver.fill_connection.inflow_equation
        assert inflow_equation is not None
        assert inflow_equation.gate.evaluate()[0] == pytest.approx(1.0, abs=1e-2)
        assert inflow_equation.symbolic_velocity(_INFLOW_CONTEXT).evaluate()[0] > 0.0


# %% spouted source

_CAN_BODY_WIDTH = 0.12
"""
Width of the spouted container's body, in metres.
"""

_CAN_BODY_HEIGHT = 0.16
"""
Height of the spouted container's body, in metres.
"""

_SPOUT_OUTLET_HEIGHT = 0.14
"""
Height of the spout outlet above the container's base, in metres.
"""

_SPOUT_OUTLET_OFFSET = -0.15
"""
Horizontal offset of the spout outlet from the container's axis along ``x``, in metres.
"""

_SPOUT_PITCH = -math.pi / 4
"""
Pitch of the spout frame; its ``z`` axis, the outflow direction, points up and along
negative ``x``.
"""


def _build_spouted_source_world() -> tuple[World, HasSpout, _TranslatingContainer]:
    """
    Builds a spouted container held above a receiver on the world root, tilted a little,
    with the spout body attached at a known offset.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        receiver = _TranslatingContainer.create_with_new_body_in_world(
            name="receiver",
            world=world,
            parent_connection_specification=_TranslatingContainer.parent_connection_specification(
                axis=Vector3(1, 0, 0)
            ),
            scale=Scale(0.1, 0.1, 0.2),
        )
    can_body = Body.from_shape_collection(
        shape_collection=ShapeCollection(
            [
                Cylinder(
                    width=_CAN_BODY_WIDTH,
                    height=_CAN_BODY_HEIGHT,
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=_CAN_BODY_HEIGHT / 2
                    ),
                )
            ]
        ),
        name=PrefixedName("can"),
    )
    spout_body = Body(name=PrefixedName("spout"))
    with world.modify_world():
        world.add_body(can_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=can_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.3, z=0.5, pitch=0.2
                ),
            )
        )
        world.add_body(spout_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=can_body,
                child=spout_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=_SPOUT_OUTLET_OFFSET, z=_SPOUT_OUTLET_HEIGHT, pitch=_SPOUT_PITCH
                ),
            )
        )
    can = HasSpout(name=PrefixedName("watering_can"), root=can_body, spout=spout_body)
    with world.modify_world():
        world.add_semantic_annotation(can)
    return world, can, receiver


class TestSpoutedSource:
    """
    A container with a spout pours from the spout outlet, along the spout, and starts
    pouring once the liquid reaches the outlet rather than the body's rim.
    """

    def test_liquid_exit_point_is_the_spout_outlet(self) -> None:
        world, can, _receiver = _build_spouted_source_world()

        exit_point = can.liquid_exit_point(world).to_np()[:3]

        world_T_spout = world.compute_forward_kinematics_np(world.root, can.spout)
        assert exit_point == pytest.approx(world_T_spout[:3, 3])

    def test_liquid_exit_direction_is_the_spout_axis(self) -> None:
        world, can, _receiver = _build_spouted_source_world()

        exit_direction = can.liquid_exit_direction(world).to_np()[:3]

        world_T_spout = world.compute_forward_kinematics_np(world.root, can.spout)
        assert exit_direction == pytest.approx(world_T_spout[:3, 2])

    def test_fill_equation_uses_the_outlet_height_and_the_body_width(self) -> None:
        world, can, _receiver = _build_spouted_source_world()

        can.initialize_fill_level(world=world, initial_fill=1.0)

        assert can.fill_equation.container_height == pytest.approx(_SPOUT_OUTLET_HEIGHT)
        assert can.fill_equation.container_width == pytest.approx(_CAN_BODY_WIDTH)

    def test_fill_equation_lip_sits_at_the_outlet(self) -> None:
        """
        The head that drives the pour is measured against the spout outlet, which sits
        farther from the tilt axis than the body wall.
        """
        world, can, _receiver = _build_spouted_source_world()

        can.initialize_fill_level(world=world, initial_fill=1.0)

        assert can.fill_equation.lip_offset == pytest.approx(abs(_SPOUT_OUTLET_OFFSET))

    def test_liquid_exit_velocity_follows_the_spout(self) -> None:
        """
        A spout channels the whole stream along its axis, vertical component included.
        """
        world, can, _receiver = _build_spouted_source_world()
        exit_speed = 0.5

        exit_velocity = can.liquid_exit_velocity(world, exit_speed).to_np()[:3]

        world_T_spout = world.compute_forward_kinematics_np(world.root, can.spout)
        assert exit_velocity == pytest.approx(exit_speed * world_T_spout[:3, 2])

    def test_landing_point_includes_the_upward_launch(self) -> None:
        """
        A stream launched upwards flies longer before it reaches the receiver's opening,
        so it lands farther out than a horizontal launch would.
        """
        world, can, receiver = _build_spouted_source_world()
        exit_speed = 0.5

        landing = _evaluated_xyz(
            receiver.projectile_landing_point(can, world, exit_speed)
        )

        outlet = _evaluated_xyz(can.liquid_exit_point(world))
        velocity = _evaluated_xyz(can.liquid_exit_velocity(world, exit_speed))
        drop = outlet[2] - _evaluated_xyz(receiver.opening_point(world))[2]
        flight_time = (
            velocity[2] + math.sqrt(velocity[2] ** 2 + 2 * STANDARD_GRAVITY * drop)
        ) / STANDARD_GRAVITY
        assert landing[:2] == pytest.approx(outlet[:2] + velocity[:2] * flight_time)
        horizontal_flight_time = math.sqrt(2 * drop / STANDARD_GRAVITY)
        assert flight_time > horizontal_flight_time

    def test_spout_survives_a_json_round_trip(self) -> None:
        """
        The spout body must come back bound to the world's body, since the annotation
        crosses to the process that controls the pour.
        """
        world, can, _receiver = _build_spouted_source_world()
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)

        restored = HasSpout.from_json(can.to_json(), **tracker.create_kwargs())

        assert restored.spout is can.spout
        assert restored.root is can.root


# %% faucet

_OUTLET_HEIGHT = 0.5
"""
Height of the faucet outlet above the world root, in metres.
"""

_VALVE_TRAVEL = math.pi / 2
"""
Position of the valve at which the faucet is fully open, in radians; it is shut at zero.
"""

_FAUCET_VOLUME_RATE = 0.002
"""
Volume rate of the fully open faucet, in cubic metres per second.
"""


def _build_faucet_world(
    valve_limits: DegreeOfFreedomLimits | None = None,
) -> tuple[World, Faucet, _TranslatingContainer]:
    """
    Builds a faucet whose outlet hangs above a receiver at the origin, with a lever
    valve on the faucet's post.

    :param valve_limits: Limits of the valve joint; the shut-to-open travel by default.
    """
    if valve_limits is None:
        valve_limits = DegreeOfFreedomLimits(
            lower=DerivativeMap(position=0.0, velocity=-1.0),
            upper=DerivativeMap(position=_VALVE_TRAVEL, velocity=1.0),
        )
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        receiver = _TranslatingContainer.create_with_new_body_in_world(
            name="receiver",
            world=world,
            parent_connection_specification=_TranslatingContainer.parent_connection_specification(
                axis=Vector3(1, 0, 0)
            ),
            scale=Scale(0.1, 0.1, 0.2),
        )
    post = Body(name=PrefixedName("faucet_post"))
    outlet = Body(name=PrefixedName("faucet_outlet"))
    valve = Body(name=PrefixedName("faucet_valve"))
    with world.modify_world():
        world.add_body(post)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=post,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.15, z=_OUTLET_HEIGHT
                ),
            )
        )
        world.add_body(outlet)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=post,
                child=outlet,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.15
                ),
            )
        )
        world.add_body(valve)
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=post,
                child=valve,
                axis=Vector3.Z(),
                dof_limits=valve_limits,
            )
        )
    faucet = Faucet(
        name=PrefixedName("faucet"),
        root=post,
        outlet=outlet,
        valve=valve,
        maximum_volume_rate=_FAUCET_VOLUME_RATE,
    )
    with world.modify_world():
        world.add_semantic_annotation(faucet)
    return world, faucet, receiver


def _turn_valve(world: World, faucet: Faucet, position: float) -> None:
    """
    Sets the valve joint of the faucet.
    """
    JointState.from_mapping({faucet.valve_connection: position}).apply_to(world)


class TestFaucet:
    """
    A faucet's flow follows its valve continuously and its water falls straight down
    from the outlet.
    """

    def test_outflow_follows_the_valve(self) -> None:
        world, faucet, _receiver = _build_faucet_world()
        rate = faucet.outflow_volume_rate(world)

        _turn_valve(world, faucet, 0.0)
        assert rate.evaluate()[0] == pytest.approx(0.0)
        _turn_valve(world, faucet, _VALVE_TRAVEL / 2)
        assert rate.evaluate()[0] == pytest.approx(_FAUCET_VOLUME_RATE / 2)
        _turn_valve(world, faucet, _VALVE_TRAVEL)
        assert rate.evaluate()[0] == pytest.approx(_FAUCET_VOLUME_RATE)

    def test_water_leaves_the_outlet_straight_down(self) -> None:
        world, faucet, _receiver = _build_faucet_world()

        exit_point = _evaluated_xyz(faucet.liquid_exit_point(world))
        exit_velocity = _evaluated_xyz(faucet.liquid_exit_velocity(world, 0.5))

        world_T_outlet = world.compute_forward_kinematics_np(world.root, faucet.outlet)
        assert exit_point == pytest.approx(world_T_outlet[:3, 3])
        assert exit_velocity == pytest.approx([0.0, 0.0, 0.0])

    def test_receiver_fills_only_while_the_valve_is_open(self) -> None:
        world, faucet, receiver = _build_faucet_world()
        receiver.initialize_fill_level(world=world, initial_fill=0.0)
        receiver.receive_outflow_from(source=faucet, world=world)
        inflow_velocity = receiver.fill_connection.inflow_equation.symbolic_velocity(
            _INFLOW_CONTEXT
        )

        _turn_valve(world, faucet, 0.0)
        assert inflow_velocity.evaluate()[0] == pytest.approx(0.0)
        _turn_valve(world, faucet, _VALVE_TRAVEL)
        assert inflow_velocity.evaluate()[0] > 0.0

    def test_valve_without_position_limits_is_rejected(self) -> None:
        world, faucet, _receiver = _build_faucet_world(
            valve_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(velocity=-1.0), upper=DerivativeMap(velocity=1.0)
            )
        )

        with pytest.raises(FaucetValveWithoutPositionLimitsError) as error_info:
            faucet.opening()

        assert error_info.value.faucet_name == faucet.name

    def test_faucet_survives_a_json_round_trip(self) -> None:
        world, faucet, _receiver = _build_faucet_world()
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)

        restored = Faucet.from_json(faucet.to_json(), **tracker.create_kwargs())

        assert restored.outlet is faucet.outlet
        assert restored.valve is faucet.valve
        assert restored.maximum_volume_rate == faucet.maximum_volume_rate


# %% container that receives and pours


class TestReceivingAndPouringContainer:
    """
    A container filled from one source can pour into another, so a synchronized change
    to it must not stumble over the process-local inflow equation it carries.
    """

    def test_coupled_receiver_serializes_without_its_symbolic_inflow(self) -> None:
        world, _source, receiver = _build_world()
        assert receiver.fill_connection.inflow_equation is not None

        payload = receiver.to_json()

        assert "inflow_equation" not in payload
        assert (
            HasFillLevel.from_json(
                payload,
                **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
            ).inflow_coupling
            == receiver.inflow_coupling
        )

    def test_receiver_can_pour_into_a_third_container(self) -> None:
        world, faucet, can = _build_faucet_world()
        can.initialize_fill_level(world=world, initial_fill=0.0)
        can.receive_outflow_from(source=faucet, world=world)
        with world.modify_world():
            pot = _TranslatingContainer.create_with_new_body_in_world(
                name="pot",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
                parent_connection_specification=_TranslatingContainer.parent_connection_specification(
                    axis=Vector3(1, 0, 0)
                ),
                scale=Scale(0.2, 0.2, 0.15),
            )
        pot.initialize_fill_level(world=world, initial_fill=0.2)

        pot.receive_outflow_from(source=can, world=world)

        assert isinstance(can.fill_equation, GatedArticulatedPouringEquation)
        assert pot.fill_connection.inflow_equation is not None
        assert can.fill_connection.inflow_equation is not None


# %% fill-level integration limits


class TestFillLevelIntegrationLimits:
    """
    Validates that physics integration keeps the fill level inside the DOF limits.
    """

    def _lone_tilting_cup(self) -> tuple[World, _TiltingContainer]:
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
        with world.modify_world():
            cup = _TiltingContainer.create_with_new_body_in_world(
                name="cup",
                world=world,
                parent_connection_specification=_TiltingContainer.parent_connection_specification(
                    axis=Vector3(0, 1, 0),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap(position=-2.0, velocity=-1.0),
                        upper=DerivativeMap(position=2.0, velocity=1.0),
                    ),
                ),
                scale=Scale(0.1, 0.1, 0.2),
            )
        return world, cup

    def test_tilted_empty_source_does_not_drain_below_empty(self):
        """
        A fully tilted cup with no liquid left must stay at fill 0, not integrate
        negative.
        """
        world, cup = self._lone_tilting_cup()
        cup.initialize_fill_level(world=world, initial_fill=0.0)
        JointState.from_mapping({cup.root.parent_connection: 1.5}).apply_to(world)

        for _ in range(50):
            world.step_physics(0.05)

        assert cup.fill_level == pytest.approx(0.0)

    def test_receiver_saturates_at_full(self):
        """
        A receiver with a steady inflow must stop at fill 1, not fill past the brim.
        """
        world, cup = self._lone_tilting_cup()
        cup.initialize_fill_level(world=world, initial_fill=0.9)
        with world.modify_world():
            cup.add_inflow_equation(
                InflowEquation(
                    container_height=0.2,
                    container_width=0.1,
                    inflow=sm.Scalar(0.005),
                )
            )

        for _ in range(50):
            world.step_physics(0.05)

        assert cup.fill_level == pytest.approx(1.0)


# %% coupling serialization


class TestLiquidCouplingSerialization:
    """
    Validates that a serialized coupling never resurrects as a silently open transfer.
    """

    def test_serialized_drain_is_ungated(self):
        """
        The source drain crosses the process boundary ungated, so a deserialized world
        cannot carry an always-open gate; the receiving world re-gates it when
        rebuilding the coupling.
        """
        world, source, receiver = _build_world()
        assert isinstance(source.fill_equation, GatedArticulatedPouringEquation)

        payload = source.fill_connection.to_json()
        restored_drain = from_json(payload["outflow_equation"])

        assert type(restored_drain) is ArticulatedPouringEquation
        assert restored_drain.container_height == pytest.approx(
            source.fill_equation.container_height
        )

    def test_symbolic_inflow_is_not_serialized(self):
        """
        The symbolic inflow cannot cross a process boundary; only the coupling
        descriptor does, so the receiving world rebuilds the inflow via
        ensure_inflow_coupling.
        """
        world, source, receiver = _build_world()
        assert receiver.fill_connection.inflow_equation is not None

        payload = receiver.fill_connection.to_json()

        assert "inflow_equation" not in payload


# %% fill-level connection builders


def build_world_with_liquid_connection(
    dof_limits: DegreeOfFreedomLimits | None = None,
) -> tuple[World, LiquidConnection]:
    """
    Build a world containing a container body with a fill-level connection.
    """
    world = World()
    container = Body(name=PrefixedName("container"))
    phantom = Body(name=PrefixedName("container_fill_level_phantom"))
    with world.modify_world():
        world.add_body(container)
    with world.modify_world():
        connection = LiquidConnection.create_with_dofs(
            world=world,
            parent=container,
            child=phantom,
            axis=Vector3.Z(),
            dof_limits=dof_limits,
        )
        world.add_connection(connection)
    return world, connection


def build_fill_level_limits() -> DegreeOfFreedomLimits:
    """
    The [0, 1] fill-level limits a fill DOF is normally created with.
    """
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(position=0.0, velocity=-1.0),
        upper=DerivativeMap(position=1.0, velocity=1.0),
    )


# %% passive fill degree of freedom


def test_fill_dof_is_passive():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    assert connection.active_dofs == []
    assert connection.passive_dofs == [connection.raw_dof]
    assert connection.raw_dof in world.passive_degrees_of_freedom
    assert connection.raw_dof not in world.active_degrees_of_freedom
    assert connection.is_controlled is False


# %% fill-level connection clamping


def test_update_state_without_position_limits_raises():
    _, connection = build_world_with_liquid_connection(dof_limits=None)
    with pytest.raises(MissingFillLevelLimitsError) as error_info:
        connection.update_state(0.05)
    assert error_info.value.connection_name == connection.name


# %% fill-level connection serialization


def test_json_round_trip_restores_ungated_outflow():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    connection.outflow_equation = GatedArticulatedPouringEquation(
        container_height=0.2,
        container_width=0.1,
        outflow_rate_constant=2.0,
        gate=sm.Scalar(0.5),
    )

    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    restored = LiquidConnection.from_json(
        connection.to_json(), **tracker.create_kwargs()
    )

    assert type(restored) is LiquidConnection
    assert type(restored.outflow_equation) is ArticulatedPouringEquation
    assert restored.outflow_equation.container_height == pytest.approx(0.2)
    assert restored.outflow_equation.container_width == pytest.approx(0.1)
    assert restored.outflow_equation.outflow_rate_constant == pytest.approx(2.0)
    assert restored.inflow_equation is None
    assert restored.raw_dof.id == connection.raw_dof.id


def test_json_round_trip_without_outflow_equation():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    restored = LiquidConnection.from_json(
        connection.to_json(), **tracker.create_kwargs()
    )
    assert restored.outflow_equation is None
    assert restored.inflow_equation is None


# %% fill-level connection copy


def test_copy_for_world_preserves_equations():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    connection.outflow_equation = ArticulatedPouringEquation(
        container_height=0.2, container_width=0.1
    )
    connection.inflow_equation = InflowEquation(
        container_height=0.2, container_width=0.1, inflow=sm.Scalar(0.005)
    )
    world_copy = deepcopy(world)

    copied_connection = connection.copy_for_world(world_copy)

    assert type(copied_connection) is LiquidConnection
    assert copied_connection.outflow_equation is connection.outflow_equation
    assert copied_connection.inflow_equation is connection.inflow_equation
    assert copied_connection.parent is world_copy.get_kinematic_structure_entity_by_id(
        connection.parent.id
    )
    assert copied_connection.child is world_copy.get_kinematic_structure_entity_by_id(
        connection.child.id
    )
    assert copied_connection.raw_dof is world_copy.get_degree_of_freedom_by_id(
        connection.raw_dof.id
    )
