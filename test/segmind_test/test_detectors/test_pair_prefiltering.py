"""
Tests for ruling pairs of bodies out before they are measured exactly.
"""

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.detectors.atomic_event_detectors_nodes import ContactDetector
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% two bodies whose distance the test sets

BOX_EDGE = 0.1
"""
Edge length of the boxes the bodies below are made of.
"""


def world_with_two_bodies(distance: float) -> World:
    """
    A world holding two equal boxes that distance apart along x.

    :param distance: How far the second body's origin sits from the first's.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
        for index in range(2):
            body = Body(
                name=PrefixedName("body%d" % index),
                collision=ShapeCollection(
                    shapes=[Box(scale=Scale(BOX_EDGE, BOX_EDGE, BOX_EDGE))]
                ),
            )
            world.add_connection(
                Connection6DoF.create_with_dofs(world=world, parent=root, child=body)
            )
    world.get_body_by_name("body1").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            distance, 0, 0, reference_frame=world.root
        )
    )
    return world


class TestDistantPairsAreNotMeasured:
    """
    Measuring a pair exactly costs far more than deciding it is too far apart to matter,
    and most pairs in a world are too far apart.

    What the prefilter must never do is rule out a pair the predicate would have called
    related.
    """

    def asked_pairs(self, distance: float, reach: float) -> list:
        """
        The pairs the predicate was actually asked about.

        :param distance: How far apart the two bodies stand.
        :param reach: The furthest separation at which the predicate can still hold.
        """
        world = world_with_two_bodies(distance)
        context = MotionStatechartContext(world=world)
        asked = []

        def predicate(one: Body, other: Body) -> bool:
            asked.append((one, other))
            return False

        ContactDetector().get_relation(
            context, [world.get_body_by_name("body0")], predicate, reach
        )
        return asked

    def test_a_pair_too_far_apart_to_touch_is_never_asked(self):
        assert self.asked_pairs(distance=10.0, reach=0.0) == []

    def test_an_overlapping_pair_is_asked(self):
        assert len(self.asked_pairs(distance=0.0, reach=0.0)) == 1

    def test_a_pair_within_the_predicate_s_reach_is_asked(self):
        just_apart = BOX_EDGE * 2

        assert self.asked_pairs(distance=just_apart, reach=0.0) == []
        assert len(self.asked_pairs(distance=just_apart, reach=just_apart)) == 1
