"""
Physical support of semantic poses on imported apartment meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cramera.live.placement_surface import PlacementSurface
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.semantic_annotations import CounterTop
from semantic_digital_twin.world import World

# %% imported supporting geometry


@dataclass(init=False, eq=False)
class TestImportedSurfaceSupport:
    """
    Generated object poses rest on the actual supporting mesh.
    """

    def test_apartment_object_bottom_rests_on_countertop(
        self, apartment_world_copy: World
    ) -> None:
        """
        An imported countertop's annotation must not leave the object floating.

        :param apartment_world_copy: Existing apartment fixture with milk geometry.
        """
        world = apartment_world_copy
        with world.modify_world():
            WorldReasoner(world).reason()
        body = world.get_body_by_name("milk.stl")
        countertop = CounterTop(root=world.get_body_by_name("island_countertop"))
        with world.modify_world():
            world.add_semantic_annotation(countertop)
        pose = next(
            iter(PlacementSurface(world, body, CounterTop, str(countertop.root.name)))
        )
        countertop_T_object = world.transform(pose, countertop.root)
        object_bottom = (
            float(countertop_T_object.to_position().z) + body.combined_mesh.bounds[0, 2]
        )
        assert object_bottom == pytest.approx(
            countertop.root.combined_mesh.bounds[1, 2], abs=1e-4
        )
