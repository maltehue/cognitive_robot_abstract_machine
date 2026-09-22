"""
Navigation maps preserve empty space and respect obstacle height.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

# %% bounded navigation space


@pytest.fixture
def navigation_space() -> BoundingBoxCollection:
    """
    Provide a finite navigation band in an empty world.
    """
    world = World.create_with_root_body()
    return BoundingBoxCollection(
        [
            BoundingBox(
                min_x=-2.0,
                min_y=-2.0,
                min_z=0.5,
                max_x=2.0,
                max_y=2.0,
                max_z=1.5,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        reference_frame=world.root,
    )


def test_empty_world_has_a_direct_navigation_path(
    navigation_space: BoundingBoxCollection,
) -> None:
    """
    An obstacle-free world allows a direct path between interior points.

    :param navigation_space: Finite search space in an empty world.
    """
    root = navigation_space.reference_frame
    start = Point3(-1.0, -1.0, 1.0, reference_frame=root)
    goal = Point3(1.0, 1.0, 1.0, reference_frame=root)
    graph = GraphOfBoundingBoxes.navigation_map_from_world(
        root._world, search_space=navigation_space
    )
    assert graph.path_from_to(start, goal) == [start, goal]


def test_navigation_rejects_mismatched_frames(
    navigation_space: BoundingBoxCollection,
) -> None:
    """
    Coordinate events from unrelated frames must not be combined silently.

    :param navigation_space: Search space belonging to the navigation world.
    """
    other_world = World.create_with_root_body()
    obstacles = BoundingBoxCollection([], reference_frame=other_world.root)
    with pytest.raises(ValueError, match="same reference frame"):
        GraphOfBoundingBoxes.navigation_map_from_bounding_boxes(
            navigation_space, obstacles
        )


# %% vertical obstacle filtering


@pytest.mark.parametrize(
    "minimum_z,maximum_z,overlapping",
    [
        pytest.param(0.0, 0.4, False, id="below"),
        pytest.param(1.6, 2.0, False, id="above"),
        pytest.param(0.9, 1.1, True, id="overlapping"),
    ],
)
def test_planar_obstacles_respect_the_search_height(
    navigation_space: BoundingBoxCollection,
    minimum_z: float,
    maximum_z: float,
    overlapping: bool,
) -> None:
    """
    Only obstacles intersecting the navigation height block their footprint.

    :param navigation_space: Search space defining the occupied height band.
    :param minimum_z: Lower obstacle boundary.
    :param maximum_z: Upper obstacle boundary.
    :param overlapping: Whether the obstacle intersects the search height.
    """
    root = navigation_space.reference_frame
    obstacles = BoundingBoxCollection(
        [
            BoundingBox(
                -0.5,
                -0.5,
                minimum_z,
                0.5,
                0.5,
                maximum_z,
                HomogeneousTransformationMatrix(reference_frame=root),
            )
        ],
        reference_frame=root,
    )
    search = navigation_space.event
    search_xy = search.marginal(SpatialVariables.xy)
    footprint = obstacles.event.marginal(SpatialVariables.xy)
    free_space = GraphOfBoundingBoxes.free_space_from_bounding_boxes(
        obstacles, search, keep_z=False
    )
    occupied_space = GraphOfBoundingBoxes.obstacles_from_bounding_boxes(
        obstacles, search, keep_z=False
    )
    if overlapping:
        assert free_space == search_xy - footprint
        assert occupied_space == footprint
    else:
        assert free_space == search_xy
        assert occupied_space is None
