from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types.spatial_types import Point3


@dataclass(eq=False)
class UsdSemanticLabels(HasRootBody):
    """
    The semantic labels a USD prim carries under one ``UsdSemantics.LabelsAPI``
    taxonomy, attached to the Body a USD-reading parser built for that prim. A prim with
    labels in several taxonomies (e.g. ``"class"`` and ``"category"``) gets one of these
    per taxonomy.

    Unlike the domain-specific annotations in ``semantic_annotations.py`` (``Furniture``,
    ``Room``, ...), a taxonomy's labels are open-vocabulary strings authored by whoever
    modelled the USD asset, not a fixed set this codebase defines - this type only
    carries them, it does not interpret them.
    """

    taxonomy: str = field(kw_only=True)
    """
    The taxonomy (label namespace, e.g. ``"class"``) these labels were authored under.
    """

    labels: List[str] = field(kw_only=True)
    """
    The labels authored under this taxonomy (e.g. ``["chair", "furniture"]``).
    """


@dataclass(eq=False)
class UsdStageOrigin(HasRootBody):
    """
    Where the origin of the USD stage a world was parsed from lies.

    A world whose root is not placed at that origin - a scan rooted on its own geometry
    rather than on the survey origin it was measured against - would otherwise lose the
    coordinates it was authored in.
    """

    position: Point3 = field(kw_only=True)
    """
    The stage's origin, in the root body's frame.
    """
