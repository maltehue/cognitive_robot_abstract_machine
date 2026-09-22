"""Standard queries answered directly by an attached semantic digital twin."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Callable

from cramera.knowledge.presets import Preset, DirectAnswerPreset
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope
from cramera.live.query import LiveQuerySource
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Table,
    CounterTop,
)
from cramera.spatial_annotations import PlacementAnswer
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% standard world vocabulary
class WorldQueryName(StrEnum):
    """Ready-made variables for the entities in an attached world."""

    BODY = "body"
    """Bodies composing the world."""
    SCENE_OBJECT = "scene_object"
    """Scene bodies available to object queries."""
    ANNOTATION = "annotation"
    """Every semantic annotation in the world."""
    HANDLE = "handle"
    """Annotated handles."""
    SURFACE = "surface"
    """Annotated supporting surfaces."""
    ROBOT = "robot"
    """Annotated robots."""
    ARM = "arm"
    """Annotated robot arms."""


class WorldQuestion(StrEnum):
    """Standard questions independent of a demonstration's extensions."""

    BODIES = "show all scene bodies"
    """List the world's bodies."""
    ANNOTATIONS = "show all semantic annotations"
    """List semantic types and their bodies."""
    HANDLES = "show all handles"
    """Find annotated handles in the scene."""
    SURFACES = "show all supporting surfaces"
    """Find annotated placement surfaces."""


# %% default query source
@dataclass
class WorldQuerySource(LiveQuerySource):
    """Answer from the current bodies and annotations of an attached world."""

    world: World
    """World read afresh whenever a question is evaluated."""
    objects: Callable[[], list[Body]] | None = None
    """Current published loose objects used to offer placement questions."""

    def title(self) -> str:
        """Identify the world supplying live answers."""
        return "Live world"

    def knowledge(self) -> list[QueryableKnowledge]:
        """Expose SDT entities with their original classes and relationships."""
        return [
            QueryableKnowledge(
                scope=QueryScope.CURRENT_STATE,
                domains=[
                    QueryDomain(WorldQueryName.BODY, Body, list(self.world.bodies)),
                    QueryDomain(
                        WorldQueryName.SCENE_OBJECT, Body, list(self.world.bodies)
                    ),
                    QueryDomain(
                        WorldQueryName.ANNOTATION,
                        SemanticAnnotation,
                        list(self.world.semantic_annotations),
                    ),
                    QueryDomain(
                        WorldQueryName.HANDLE,
                        Handle,
                        self.world.get_semantic_annotations_by_type(Handle),
                    ),
                    QueryDomain(
                        WorldQueryName.SURFACE,
                        HasSupportingSurface,
                        self.world.get_semantic_annotations_by_type(
                            HasSupportingSurface
                        ),
                    ),
                    QueryDomain(
                        WorldQueryName.ROBOT,
                        AbstractRobot,
                        self.world.get_semantic_annotations_by_type(AbstractRobot),
                    ),
                    QueryDomain(
                        WorldQueryName.ARM,
                        Arm,
                        self.world.get_semantic_annotations_by_type(Arm),
                    ),
                ],
                extra_names={"placement_region": self.placement_region},
            )
        ]

    def placement_region(self, object_name: str, surface_name: str) -> PlacementAnswer:
        """Compute a region only when this named question is asked.

        :param object_name: Candidate body whose current geometry must fit.
        :param surface_name: Table or countertop annotation supplying the support.
        """
        return PlacementAnswer.of_world(self.world, object_name, surface_name)

    def presets(self) -> list[Preset]:
        """Offer robot questions and questions about annotated surroundings."""
        robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
        presets = Preset.for_robots(len(robots)) + [
            Preset(question, f"an(entity({variable}))")
            for question, variable in (
                (WorldQuestion.BODIES, WorldQueryName.BODY),
                (WorldQuestion.ANNOTATIONS, WorldQueryName.ANNOTATION),
                (WorldQuestion.HANDLES, WorldQueryName.HANDLE),
                (WorldQuestion.SURFACES, WorldQueryName.SURFACE),
            )
        ]
        if self.objects is not None:
            presets.extend(
                DirectAnswerPreset(
                    PlacementAnswer.question_for(str(body.name), str(surface.name)),
                    PlacementAnswer.code_for(str(body.name), str(surface.name)),
                )
                for body in self.objects()
                for surface in self.world.semantic_annotations
                if isinstance(surface, (Table, CounterTop)) and body is not surface.root
            )
        return presets
