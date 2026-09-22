"""Placement annotations for the named surfaces in CRAM's bundled apartment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import ClassVar, TYPE_CHECKING

from cramera.model_catalog import ModelCatalog
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Table,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% bundled model identity


class ApartmentAsset(StrEnum):
    """Exact identity of the apartment description bundled with coraplex."""

    MODEL = "apartment.urdf"
    """Installed apartment URDF filename."""

    PREFIX = "apartment"
    """Body prefix written by the apartment's URDF parser."""


class ApartmentBody(StrEnum):
    """Supporting bodies explicitly named in the bundled apartment URDF."""

    COUNTERTOP = "countertop"
    """Kitchen wall countertop."""

    ISLAND_COUNTERTOP = "island_countertop"
    """Kitchen island countertop."""

    COFFEE_TABLE = "coffee_table"
    """Living-room coffee table."""

    BEDSIDE_TABLE = "bedside_table"
    """Bedroom bedside table."""

    TABLE_AREA = "table_area_main"
    """Dining table body."""


# %% existing geometry annotations


@dataclass
class PlacementAnnotations:
    """Add the bundled environment's missing semantic surface types idempotently."""

    world: World
    """Parsed environment and robot world receiving surface annotations."""

    environment_path: Path | str
    """Exact selected model path, checked against the installed apartment asset."""

    SURFACES: ClassVar[dict[ApartmentBody, type[HasSupportingSurface]]] = {
        ApartmentBody.COUNTERTOP: CounterTop,
        ApartmentBody.ISLAND_COUNTERTOP: CounterTop,
        ApartmentBody.COFFEE_TABLE: Table,
        ApartmentBody.BEDSIDE_TABLE: Table,
        ApartmentBody.TABLE_AREA: Table,
    }
    """Existing SDT annotation types for the apartment's supporting bodies."""

    @classmethod
    def apartment_path(cls) -> Path:
        """Return the apartment asset from the builder's installed model inventory."""
        return ModelCatalog.installed().worlds_directory / ApartmentAsset.MODEL

    def apply(self) -> None:
        """Annotate known apartment bodies without replacing existing annotations."""
        if Path(self.environment_path).resolve() != self.apartment_path().resolve():
            return
        with self.world.modify_world():
            for body in self.world.bodies:
                if body.name.prefix != ApartmentAsset.PREFIX:
                    continue
                surface_type = self.SURFACES.get(body.name.name)
                if surface_type is None:
                    continue
                if any(
                    surface.root is body
                    for surface in self.world.get_semantic_annotations_by_type(
                        surface_type
                    )
                ):
                    continue
                self.world.add_semantic_annotation(surface_type(root=body))
