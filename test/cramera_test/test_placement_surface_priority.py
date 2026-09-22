from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest

from cramera.live.placement_surface import PlacementSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale

from .test_placement_surface import (
    PlacementScene,
    SurfaceSamples,
    placement_scene,
    surface_scene,
)

# %% alternative semantic surfaces


@pytest.fixture(params=[0.0, np.pi])
def placement_surfaces(
    surface_scene: PlacementScene,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> list[PlacementScene]:
    with surface_scene.world.modify_world():
        surface = Table.create_with_new_body_in_world(
            name="nearby_table",
            world=surface_scene.world,
            scale=Scale(2.0, 2.0, 0.1),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=5.0, yaw=request.param
            ),
        )
        surface.calculate_supporting_surface()
    scenes = [
        surface_scene,
        PlacementScene(surface_scene.world, surface, surface_scene.object),
    ]
    for scene in scenes:
        samples = SurfaceSamples(scene)
        monkeypatch.setattr(
            scene.surface,
            "sample_points_from_surface",
            Mock(return_value=[samples.point(x=-0.7), samples.point(x=0.7)]),
        )
    return scenes


# %% search priority across surfaces


@dataclass(init=False, eq=False)
class TestSurfacePriority:
    def test_orders_all_surfaces_by_world_distance(
        self, placement_surfaces: list[PlacementScene]
    ) -> None:
        earlier, nearer = placement_surfaces
        provider = PlacementSurface(earlier.world, earlier.object.root, Table)

        poses = list(provider)

        assert [pose.reference_frame for pose in poses] == [
            nearer.surface.supporting_surface,
            earlier.surface.supporting_surface,
            nearer.surface.supporting_surface,
            earlier.surface.supporting_surface,
        ]

    def test_checks_support_only_for_requested_candidates(
        self,
        placement_surfaces: list[PlacementScene],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        earlier, nearer = placement_surfaces
        provider = PlacementSurface(earlier.world, earlier.object.root, Table)
        support = Mock(wraps=provider.supported_pose)
        monkeypatch.setattr(provider, "supported_pose", support)

        first = next(iter(provider))

        assert first.reference_frame is nearer.surface.supporting_surface
        assert support.call_count == 1
        assert support.call_args.args[0] is nearer.surface

    def test_exact_surface_name_keeps_the_requested_table(
        self, placement_surfaces: list[PlacementScene]
    ) -> None:
        earlier, _ = placement_surfaces
        provider = PlacementSurface(
            earlier.world,
            earlier.object.root,
            Table,
            surface_name=str(earlier.surface.root.name),
        )

        poses = list(provider)

        assert [pose.reference_frame for pose in poses] == [
            earlier.surface.supporting_surface,
            earlier.surface.supporting_surface,
        ]
