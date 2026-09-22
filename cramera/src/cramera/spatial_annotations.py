"""Frozen semantic geometry shared by query answers and recorded scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Sequence

from typing_extensions import Any, TYPE_CHECKING

from krrood.adapters.json_serializer import from_json, to_json
from krrood.exceptions import DataclassException
from cramera.body_geometry import (
    POSE_PRECISION,
    rounded_pose,
    rounded_scale,
    shape_collections_of,
)
from cramera.knowledge.entity import NamedEntity
from cramera.live.markers import MarkerEntry
from cramera.live.free_placement_regions import FreePlacementRegions
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Table,
    CounterTop,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% published semantic geometry
class SpatialField(StrEnum):
    """Fields carrying frozen semantic geometry outside a live world."""

    ANNOTATIONS = "semanticAnnotations"
    """Annotations retained in a recorded scene."""
    ANSWER = "spatial"
    """World-space markers accompanying an answer."""


class SemanticMarker(StrEnum):
    """Rendering vocabulary for semantic query highlights."""

    NAMESPACE = "semantic_query"
    """Namespace separating query highlights from ROS debug markers."""
    BOX = "cube"
    """Bounding geometry rendered over the annotated body."""
    COLOR = "#35ebba"
    """Semantic highlight colour."""


class ObservationSource(StrEnum):
    """Observation time displayed with semantic answers."""

    CURRENT = "Current world"
    """Geometry frozen when the live question is evaluated."""
    RECORDED = "Recording snapshot at finalization"
    """Geometry retained when the saved bundle was written."""


@dataclass
class SpatialEntity(NamedEntity):
    """An annotation's identity and geometry at one observation time."""

    semantic_type: str
    """Original annotation class name."""
    body_name: str
    """Root body identified by the annotation."""
    position: list[float]
    """Root-body position in world coordinates, in metres."""
    markers: list[MarkerEntry] = field(default_factory=list, repr=False)
    """Frozen world-space geometry to highlight for this answer."""
    semantic_types: list[str] = field(default_factory=list, repr=False)
    """Original class ancestry preserving inherited query categories."""
    observation: str = ObservationSource.CURRENT
    """When the answer's frozen geometry was observed."""

    @classmethod
    def of_entity(cls, entity: Body | HasRootBody) -> SpatialEntity:
        """Freeze the identity and bounding geometry of a body or annotation.

        :param entity: Live body or annotation whose root body is highlighted.
        :return: Independent numeric geometry retaining no reference to the world.
        """
        body = entity if isinstance(entity, Body) else entity.root
        markers = []
        with body._world.state.world_lock:
            position = rounded_pose(body)[:3]
            for collection in shape_collections_of(body):
                if not collection.shapes:
                    continue
                bounds = collection.as_bounding_box_collection_in_frame(
                    body._world.root
                ).bounding_box()
                markers.append(
                    MarkerEntry(
                        ns=SemanticMarker.NAMESPACE,
                        id=0,
                        kind=SemanticMarker.BOX,
                        frame=str(body._world.root.name),
                        position=[float(value) for value in bounds.center.to_np()[:3]],
                        quaternion=[0.0, 0.0, 0.0, 1.0],
                        scale=rounded_scale(bounds.scale, POSE_PRECISION),
                        color=SemanticMarker.COLOR,
                        opacity=0.45,
                    )
                )
                break
            return cls(
                name=str(entity.name),
                semantic_type=type(entity).__name__,
                body_name=str(body.name),
                position=position,
                markers=markers,
                semantic_types=[base.__name__ for base in type(entity).__mro__],
            )


# %% object-specific areas
@dataclass
class PlacementAnswer(SpatialEntity):
    """A frozen geometric-center region for one object on one tabletop."""

    object_name: str = ""
    """Body whose footprint determines the allowed placement area."""
    surface_name: str = ""
    """Semantic supporting surface where the area was computed."""
    status: str = ""
    """Whether this supported geometric query found available space."""

    @staticmethod
    def code_for(object_name: str, surface_name: str) -> str:
        """Build the same named query for live and recorded sources.

        :param object_name: Exact world body name of the candidate.
        :param surface_name: Exact semantic annotation name of the tabletop.
        """
        return f"placement_region({object_name!r}, {surface_name!r})"

    @staticmethod
    def question_for(object_name: str, surface_name: str) -> str:
        """Word an object-specific question for presets and voice matching.

        :param object_name: Candidate body named in the question.
        :param surface_name: Supporting annotation named in the question.
        """
        return f"where can I place {object_name} on {surface_name}?"

    @classmethod
    def of_world(
        cls, world: World, object_name: str, surface_name: str
    ) -> PlacementAnswer:
        """Resolve names against the current world and freeze the free geometry.

        :param world: Live world whose current geometry answers the question.
        :param object_name: Candidate body name.
        :param surface_name: Table or countertop annotation name.
        :raises UnknownPlacementQuery: When the requested body or surface is absent.
        """
        body = next(
            (body for body in world.bodies if str(body.name) == object_name), None
        )
        surface = next(
            (
                surface
                for surface in world.semantic_annotations
                if isinstance(surface, (Table, CounterTop))
                and str(surface.name) == surface_name
            ),
            None,
        )
        if body is None or surface is None:
            raise UnknownPlacementQuery(object_name, surface_name)
        markers = FreePlacementRegions(world).for_body(body, surface)
        return cls(
            name=cls.question_for(object_name, surface_name),
            semantic_type="PlacementRegion",
            body_name=str(surface.root.name),
            position=rounded_pose(surface.root)[:3],
            markers=markers,
            object_name=object_name,
            surface_name=surface_name,
            status=(
                "Available center positions" if markers else "No supported free region"
            ),
        )


@dataclass
class UnknownPlacementQuery(DataclassException):
    """The requested object/surface pair is unavailable in this query source."""

    object_name: str
    """Requested candidate body."""
    surface_name: str
    """Requested supporting annotation."""

    def error_message(self) -> str:
        """Identify the unavailable pair."""
        return f"No placement answer for {self.object_name} on {self.surface_name}"

    def suggest_correction(self) -> str:
        """Point to questions this live world or recording offers."""
        return "Choose an object and surface from this scene's placement presets."


# %% recorded annotations
@dataclass
class SpatialAnnotations:
    """Semantic observations preserved independently of their original world."""

    entities: list[SpatialEntity] = field(default_factory=list)
    """Annotated bodies and their frozen geometry."""
    placements: list[PlacementAnswer] = field(default_factory=list)
    """Object-specific free regions at recording finalization."""

    @classmethod
    def of_world(cls, world: World, objects: Sequence[Body] = ()) -> SpatialAnnotations:
        """Capture root-body annotations from a live world.

        :param world: World providing the current annotation identities and poses.
        :param objects: Published loose objects whose placement areas are recorded.
        :return: A snapshot suitable for recording and later queries.
        """
        with world.state.world_lock:
            entities = [
                SpatialEntity.of_entity(annotation)
                for annotation in world.semantic_annotations
                if isinstance(annotation, HasRootBody)
            ]
            object_names = [str(body.name) for body in objects]
            surfaces = {
                str(surface.name): str(surface.root.name)
                for surface in world.semantic_annotations
                if isinstance(surface, (Table, CounterTop))
            }
        placements = [
            PlacementAnswer.of_world(world, object_name, surface_name)
            for object_name in object_names
            for surface_name, root_name in surfaces.items()
            if object_name != root_name
        ]
        return cls(entities=entities, placements=placements)

    @classmethod
    def of_scene(cls, scene: dict[str, Any]) -> SpatialAnnotations:
        """Read recorded observations through KRROOD's dataclass serializer.

        :param scene: Recorded scene metadata, possibly predating semantic snapshots.
        :return: Recorded annotations, or an empty snapshot for older recordings.
        """
        recorded = scene.get(SpatialField.ANNOTATIONS)
        snapshot = from_json(recorded) if recorded else cls()
        for entity in snapshot.entities + snapshot.placements:
            entity.observation = ObservationSource.RECORDED
        return snapshot

    def scene_fields(self) -> dict[str, Any]:
        """Provide the scene metadata field holding this typed snapshot."""
        return {SpatialField.ANNOTATIONS: to_json(self)}

    def of_type(self, semantic_type: str) -> list[SpatialEntity]:
        """Select annotations by their original semantic class name.

        :param semantic_type: Class name recorded for the semantic annotation.
        :return: Matching frozen annotations.
        """
        return [
            entity
            for entity in self.entities
            if semantic_type in entity.semantic_types
            or entity.semantic_type == semantic_type
        ]

    def placement_region(self, object_name: str, surface_name: str) -> PlacementAnswer:
        """Answer using the area's state at recording finalization.

        :param object_name: Candidate body name stored with the recording.
        :param surface_name: Tabletop annotation name stored with the recording.
        :raises UnknownPlacementQuery: If this recording contains no such pair.
        """
        for answer in self.placements:
            if (
                answer.object_name == object_name
                and answer.surface_name == surface_name
            ):
                return answer
        raise UnknownPlacementQuery(object_name, surface_name)
