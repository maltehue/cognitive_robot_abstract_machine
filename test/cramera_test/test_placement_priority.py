"""
Semantic placement preserves free samples while prioritizing nearby poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest

from cramera.live.placement_surface import PlacementSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from .test_placement_surface import (
    PlacementScene,
    SurfaceSamples,
    placement_scene,
    surface_scene,
)


# %% distance ordering in physical coordinates
@dataclass(init=False, eq=False)
class TestPlacementPriority:
    """
    Candidate priority follows current world positions and retains every sample.
    """

    @staticmethod
    def move_body(
        scene: PlacementScene, body: Body, pose: HomogeneousTransformationMatrix
    ) -> None:
        """
        Replace a fixed fixture connection with a new world-relative pose.

        :param scene: Fixture containing the moved body.
        :param body: Table or object whose origin changes.
        :param pose: New world-relative transform.
        """
        with scene.world.modify_world():
            scene.world.remove_connection(body.parent_connection)
            scene.world.add_connection(
                FixedConnection(
                    parent=scene.world.root,
                    child=body,
                    parent_T_connection_expression=pose,
                )
            )

    @pytest.mark.parametrize(
        "table_pose, near_x",
        [
            (HomogeneousTransformationMatrix(), 0.7),
            (HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.0, yaw=np.pi), -0.7),
        ],
    )
    def test_prioritizes_world_distance_without_dropping_samples(
        self,
        surface_scene: PlacementScene,
        monkeypatch: pytest.MonkeyPatch,
        table_pose: HomogeneousTransformationMatrix,
        near_x: float,
    ) -> None:
        """
        A far-first native sample is reordered correctly through a rotated frame.

        :param surface_scene: Real supported table and movable object.
        :param monkeypatch: Replace only the native sample order.
        :param table_pose: Table transform testing world-frame distance.
        :param near_x: Local coordinate of the physically nearer sample.
        """
        self.move_body(surface_scene, surface_scene.surface.root, table_pose)
        samples = SurfaceSamples(surface_scene)
        near, far = samples.point(x=near_x), samples.point(x=-near_x)
        sampler = Mock(return_value=[far, near])
        monkeypatch.setattr(Table, "sample_points_from_surface", sampler)
        provider = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table
        )

        poses = list(provider)

        np.testing.assert_allclose(
            [pose.to_position().to_np()[:2] for pose in poses],
            [near.to_np()[:2], far.to_np()[:2]],
            atol=1e-12,
        )
        assert sampler.call_args.kwargs["amount"] == provider.sample_count
        assert all(pose.reference_frame is near.reference_frame for pose in poses)

    def test_validates_support_only_as_candidates_are_requested(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Returning the nearest pose leaves farther mesh support checks deferred.

        :param surface_scene: Real supported table and movable object.
        :param monkeypatch: Observe support checks for controlled sample poses.
        """
        samples = SurfaceSamples(surface_scene)
        far, near = samples.point(x=-0.7), samples.point(x=0.7)
        monkeypatch.setattr(
            Table, "sample_points_from_surface", Mock(return_value=[far, near])
        )
        provider = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table
        )
        support = Mock(wraps=provider.supported_pose)
        monkeypatch.setattr(provider, "supported_pose", support)

        first = next(iter(provider))

        np.testing.assert_allclose(
            first.to_position().to_np()[:2], near.to_np()[:2], atol=1e-12
        )
        assert support.call_count == 1

    def test_recomputes_priority_when_the_object_moves(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Reusing a lazy provider observes the object's position at each iteration.

        :param surface_scene: Real supported table and movable object.
        :param monkeypatch: Supply the same sampled points for both iterations.
        """
        samples = SurfaceSamples(surface_scene)
        left, right = samples.point(x=-0.7), samples.point(x=0.7)
        monkeypatch.setattr(
            Table, "sample_points_from_surface", Mock(return_value=[left, right])
        )
        provider = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table
        )
        first = list(provider)
        source = surface_scene.object.root.parent_connection.origin
        self.move_body(
            surface_scene,
            surface_scene.object.root,
            HomogeneousTransformationMatrix.from_xyz_rpy(x=-float(source.x)),
        )

        second = list(provider)

        np.testing.assert_allclose(
            first[0].to_position().to_np()[:2], right.to_np()[:2], atol=1e-12
        )
        np.testing.assert_allclose(
            second[0].to_position().to_np()[:2], left.to_np()[:2], atol=1e-12
        )
        np.testing.assert_allclose(
            [pose.to_np() for pose in first],
            [pose.to_np() for pose in reversed(second)],
            atol=1e-12,
        )
