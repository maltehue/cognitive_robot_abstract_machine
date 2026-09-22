"""
Demonstration transports resolve their semantic destination after acquiring the object.
"""

import pytest

from coraplex.demonstrations import RobotDemonstration
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from cramera.live.placement_surface import PlacementSurface
from cramera.mobile_transport_demo import MobileTransportDemo
from cramera.semantic_transport_demo import SemanticTransportDemo
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World


# %% deferred semantic destination
@pytest.mark.parametrize(
    "demonstration_type", (MobileTransportDemo, SemanticTransportDemo)
)
def test_demonstration_defers_destination_until_transport(
    demonstration_type: type[RobotDemonstration], pr2_world_copy: World
) -> None:
    """
    A transport retains the surface provider instead of grounding a pose before pickup.

    :param demonstration_type: Demo composing a semantic transport.
    :param pr2_world_copy: Isolated robot world for inspecting the constructed action.
    """
    demonstration = demonstration_type(used_robot=PR2)
    demonstration.populate_scene(pr2_world_copy)
    plan = demonstration.build_plan(demonstration.build_context(pr2_world_copy))

    [transport] = plan.get_nodes_by_designator_type(TransportAction)
    destination = transport.designator.target_location

    assert isinstance(destination, PlacementSurface)
    assert destination.world is pr2_world_copy
    assert destination.body is transport.designator.object_designator.root
