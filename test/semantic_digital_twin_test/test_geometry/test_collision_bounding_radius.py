"""
Tests for the sphere around a body's origin that encloses its collision geometry.
"""

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% bodies whose radius is known by hand

BOX_EDGE = 0.2
"""
Edge length of the box the bodies below are made of.
"""


def body_with_box_at(origin: HomogeneousTransformationMatrix) -> Body:
    """
    A body whose collision geometry is one box, placed as given within the body.

    :param origin: Where the box sits relative to the body's own frame.
    """
    return Body(
        name=PrefixedName("body"),
        collision=ShapeCollection(
            shapes=[Box(origin=origin, scale=Scale(BOX_EDGE, BOX_EDGE, BOX_EDGE))]
        ),
    )


class TestTheRadiusEnclosesTheCollisionGeometry:
    """
    The radius has to be an upper bound on how far the geometry reaches from the body's
    origin: it is used to rule pairs of bodies out before they are measured exactly, and
    one that is too small rules out a pair that does touch.
    """

    def test_a_box_reaches_to_its_own_corner(self):
        body = body_with_box_at(HomogeneousTransformationMatrix())

        assert body.collision_bounding_radius == np.linalg.norm([BOX_EDGE / 2] * 3)

    def test_geometry_placed_away_from_the_origin_reaches_further(self):
        offset = 1.0
        body = body_with_box_at(
            HomogeneousTransformationMatrix.from_xyz_rpy(offset, 0, 0)
        )

        assert body.collision_bounding_radius == np.linalg.norm(
            [offset + BOX_EDGE / 2, BOX_EDGE / 2, BOX_EDGE / 2]
        )

    def test_a_body_without_collision_geometry_reaches_nowhere(self):
        assert Body(name=PrefixedName("frame")).collision_bounding_radius == 0.0
