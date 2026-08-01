import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.wrist_cable_guide import (
    InvalidCableRouteError,
    WristCableGuide,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% test fixtures


ANCHOR_TRANSFORM = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1, z=0.28)
GUIDE_TRANSFORM = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1, z=0.02)
WRIST_OFFSET_Z = 0.3
SLACK_LENGTH = 0.05


def _build_wrist_world() -> tuple[World, RevoluteConnection]:
    """
    Build a world with a fixed forearm and a revolute wrist carrying a hand.

    :return: The world and its wrist connection.
    """
    world = World()
    with world.modify_world():
        forearm = Body(name=PrefixedName("forearm"))
        world.add_body(forearm)
        hand = Body(name=PrefixedName("hand"))
        world.add_body(hand)
        wrist = RevoluteConnection.create_with_dofs(
            world=world,
            parent=forearm,
            child=hand,
            name=PrefixedName("wrist"),
            axis=Vector3.Z(),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-3.0, velocity=-1.0),
                upper=DerivativeMap(position=3.0, velocity=1.0),
            ),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=WRIST_OFFSET_Z
            ),
        )
        world.add_connection(wrist)
    return world, wrist


def _expected_span(wrist_angle: float) -> float:
    """
    Analytic anchor-to-guide span at a given wrist angle for the test geometry.

    :param wrist_angle: Wrist rotation about the local z-axis in radians.
    :return: The span in meters.
    """
    anchor = np.array([0.1, 0.0, 0.28])
    guide = np.array(
        [
            0.1 * np.cos(wrist_angle),
            0.1 * np.sin(wrist_angle),
            WRIST_OFFSET_Z + 0.02,
        ]
    )
    return float(np.linalg.norm(anchor - guide))


@pytest.fixture()
def cable_world() -> tuple[World, WristCableGuide]:
    """
    A wrist world with a single-segment soft cable routed around the wrist.
    """
    world, _ = _build_wrist_world()
    forearm = world.get_body_by_name("forearm")
    hand = world.get_body_by_name("hand")
    cable = WristCableGuide.route_around_wrist(
        world=world,
        proximal_body=forearm,
        distal_body=hand,
        proximal_body_T_anchor=ANCHOR_TRANSFORM,
        distal_body_T_guide=GUIDE_TRANSFORM,
        slack_length=SLACK_LENGTH,
    )
    return world, cable


def _set_wrist_angle(world: World, wrist_angle: float) -> None:
    """
    Set the wrist joint to a given angle and refresh the world state.
    """
    wrist = world.get_connection_by_name("wrist")
    world.state[wrist.dof.id].position = wrist_angle
    world.notify_state_change()


# %% construction


class TestCableConstruction:
    def test_rest_length_is_taut_span_plus_slack(
        self, cable_world: tuple[World, WristCableGuide]
    ):
        _, cable = cable_world
        assert cable.rest_length == pytest.approx(_expected_span(0.0) + SLACK_LENGTH)

    def test_single_segment_has_one_extension_dof(
        self, cable_world: tuple[World, WristCableGuide]
    ):
        _, cable = cable_world
        assert len(cable.cable_segments) == 1
        assert len(cable.extension_dofs) == 1

    def test_extension_starts_at_rest(self, cable_world: tuple[World, WristCableGuide]):
        world, cable = cable_world
        assert world.state[cable.extension_dofs[0].id].position == 1.0

    def test_multiple_segments_split_rest_length(self):
        world, _ = _build_wrist_world()
        cable = WristCableGuide.route_around_wrist(
            world=world,
            proximal_body=world.get_body_by_name("forearm"),
            distal_body=world.get_body_by_name("hand"),
            proximal_body_T_anchor=ANCHOR_TRANSFORM,
            distal_body_T_guide=GUIDE_TRANSFORM,
            slack_length=SLACK_LENGTH,
            segment_count=3,
        )
        assert len(cable.cable_segments) == 3
        assert len(cable.extension_dofs) == 3

    def test_negative_slack_is_rejected(self):
        world, _ = _build_wrist_world()
        with pytest.raises(InvalidCableRouteError):
            WristCableGuide.route_around_wrist(
                world=world,
                proximal_body=world.get_body_by_name("forearm"),
                distal_body=world.get_body_by_name("hand"),
                proximal_body_T_anchor=ANCHOR_TRANSFORM,
                distal_body_T_guide=GUIDE_TRANSFORM,
                slack_length=-0.01,
            )

    def test_zero_segments_is_rejected(self):
        world, _ = _build_wrist_world()
        with pytest.raises(InvalidCableRouteError):
            WristCableGuide.route_around_wrist(
                world=world,
                proximal_body=world.get_body_by_name("forearm"),
                distal_body=world.get_body_by_name("hand"),
                proximal_body_T_anchor=ANCHOR_TRANSFORM,
                distal_body_T_guide=GUIDE_TRANSFORM,
                segment_count=0,
            )


# %% stretch state


class TestCableStretchState:
    def test_slack_cable_reports_zero_stretch(
        self, cable_world: tuple[World, WristCableGuide]
    ):
        world, cable = cable_world
        _set_wrist_angle(world, 0.0)
        assert cable.current_span() == pytest.approx(_expected_span(0.0))
        assert cable.current_stretch() == 0.0

    def test_rotating_wrist_stretches_cable(
        self, cable_world: tuple[World, WristCableGuide]
    ):
        world, cable = cable_world
        _set_wrist_angle(world, np.pi / 2)
        span = _expected_span(np.pi / 2)
        assert cable.current_span() == pytest.approx(span)
        assert cable.current_stretch() == pytest.approx(span - cable.rest_length)

    def test_synchronize_sets_extension_to_span_ratio(
        self, cable_world: tuple[World, WristCableGuide]
    ):
        world, cable = cable_world
        _set_wrist_angle(world, np.pi / 2)
        cable.synchronize_soft_model_with_geometry()
        expected_ratio = _expected_span(np.pi / 2) / cable.rest_length
        assert world.state[cable.extension_dofs[0].id].position == pytest.approx(
            expected_ratio
        )

    def test_synchronized_soft_model_length_matches_span(
        self, cable_world: tuple[World, WristCableGuide]
    ):
        world, cable = cable_world
        _set_wrist_angle(world, np.pi / 2)
        cable.synchronize_soft_model_with_geometry()
        anchor_T_tip = world.compute_forward_kinematics_np(
            cable.cable_anchor, cable.cable_segments[-1]
        )
        rod_length = float(np.linalg.norm(anchor_T_tip[:3, 3]))
        assert rod_length == pytest.approx(_expected_span(np.pi / 2), abs=1e-6)
