from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor, Table
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdSemanticLabels
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% support label interpretation


class UsdSupportLabel(StrEnum):
    """
    Support labels with a core semantic interpretation.
    """

    FLOOR = "floor"
    TABLE = "table"
    WORKBENCH = "workbench"


@dataclass
class UsdSceneSemantics:
    """
    Preserve open-vocabulary labels and identify configured support surfaces.
    """

    support_labels: dict[str, type[HasSupportingSurface]] = field(
        default_factory=lambda: {
            UsdSupportLabel.FLOOR: Floor,
            UsdSupportLabel.TABLE: Table,
            UsdSupportLabel.WORKBENCH: Table,
        }
    )
    """
    Case-insensitive labels mapped to support annotation types.
    """

    def __post_init__(self) -> None:
        """
        Normalize the configured label vocabulary for case-insensitive matching.
        """
        self.support_labels = {
            label.casefold(): annotation_type
            for label, annotation_type in self.support_labels.items()
        }

    def find_support_type(
        self, labels_by_taxonomy: dict[str, list[str]]
    ) -> type[HasSupportingSurface] | None:
        """
        Find the first configured support label among directly authored labels.
        """
        return next(
            (
                self.support_labels[label.casefold()]
                for labels in labels_by_taxonomy.values()
                for label in labels
                if label.casefold() in self.support_labels
            ),
            None,
        )

    def apply(
        self,
        world: World,
        body: Body,
        labels_by_taxonomy: dict[str, list[str]],
        support_type: type[HasSupportingSurface] | None,
    ) -> None:
        """
        Attach preserved raw labels and the resolved support annotation.
        """
        for taxonomy, labels in labels_by_taxonomy.items():
            world.add_semantic_annotation(
                UsdSemanticLabels(root=body, taxonomy=taxonomy, labels=labels)
            )
        if support_type is not None:
            world.add_semantic_annotation(support_type(root=body))
