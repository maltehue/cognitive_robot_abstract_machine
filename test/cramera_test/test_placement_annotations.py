"""
Supporting annotations supplied for CRAM's bundled apartment model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cramera.live.placement_annotations import ApartmentBody, PlacementAnnotations
from cramera.model_catalog import ModelCatalog
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Table,
)
from semantic_digital_twin.world import World

# %% known model annotations


@dataclass(init=False, eq=False)
class TestApartmentPlacementAnnotations:
    """
    Known apartment bodies receive their existing SDT semantic types exactly once.
    """

    def test_adds_both_countertops(self, apartment_world_copy: World) -> None:
        """
        Default semantic CounterTop targets resolve against both kitchen surfaces.

        :param apartment_world_copy: Existing full apartment URDF fixture.
        """
        annotation = PlacementAnnotations(
            apartment_world_copy, PlacementAnnotations.apartment_path()
        )
        annotation.apply()
        assert {
            surface.root.name.name
            for surface in apartment_world_copy.get_semantic_annotations_by_type(
                CounterTop
            )
        } == {ApartmentBody.COUNTERTOP, ApartmentBody.ISLAND_COUNTERTOP}

    def test_adds_the_existing_table_bodies(self, apartment_world_copy: World) -> None:
        """
        Table targets retain the identity of the original apartment geometry.

        :param apartment_world_copy: Existing full apartment URDF fixture.
        """
        PlacementAnnotations(
            apartment_world_copy, PlacementAnnotations.apartment_path()
        ).apply()
        assert {
            surface.root.name.name
            for surface in apartment_world_copy.get_semantic_annotations_by_type(Table)
        } == {
            ApartmentBody.COFFEE_TABLE,
            ApartmentBody.BEDSIDE_TABLE,
            ApartmentBody.TABLE_AREA,
        }

    def test_preserves_existing_annotations_on_repeated_setup(
        self, apartment_world_copy: World
    ) -> None:
        """
        Repeated context creation must not replace or duplicate annotations.

        :param apartment_world_copy: Existing full apartment URDF fixture.
        """
        annotation = PlacementAnnotations(
            apartment_world_copy, PlacementAnnotations.apartment_path()
        )
        annotation.apply()
        before = {surface.id for surface in apartment_world_copy.semantic_annotations}
        annotation.apply()
        assert {
            surface.id for surface in apartment_world_copy.semantic_annotations
        } == before

    def test_does_not_annotate_an_unrelated_same_named_file(
        self, apartment_world_copy: World, tmp_path: Path
    ) -> None:
        """
        A filename alone must not identify a foreign world as CRAM's apartment.

        :param apartment_world_copy: Existing full apartment URDF fixture.
        :param tmp_path: Unrelated temporary model directory.
        """
        before = {surface.id for surface in apartment_world_copy.semantic_annotations}
        PlacementAnnotations(
            apartment_world_copy, tmp_path / PlacementAnnotations.apartment_path().name
        ).apply()
        assert {
            surface.id for surface in apartment_world_copy.semantic_annotations
        } == before

    def test_uses_the_catalogs_installed_environment_directory(self) -> None:
        """
        Model scoping follows the same installed inventory as the builder.
        """
        assert (
            PlacementAnnotations.apartment_path().parent
            == ModelCatalog.installed().worlds_directory
        )
