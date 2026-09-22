"""Frozen placement-center regions on horizontal rectangular tabletops."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

import numpy as np
from typing_extensions import TYPE_CHECKING

from cramera.body_geometry import rounded_pose
from cramera.live.markers import MarkerEntry
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

if TYPE_CHECKING:
    from random_events.product_algebra import Event
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body


# %% overlay vocabulary
class PlacementOverlayStyle(StrEnum):
    """Protocol values identifying the placement overlay."""

    NAMESPACE = "free_placement"
    """Namespace identifying available placement-center areas."""
    SHAPE = "cube"
    """Existing renderer shape for a thin rectangle."""
    COLOR = "#39d5c8"
    """Color distinguishing free placement areas."""


# %% free placement areas
@dataclass
class FreePlacementRegions:
    """Show conservative center positions on rectangular Table and CounterTop tops.

    Object axes are aligned with the tabletop. Internal storage spaces, tilted
    surfaces and top patches with holes or nonrectangular outlines return no region.
    """

    world: World
    """Current digital twin supplying geometry and collision obstacles."""
    tolerance: float = 0.00001
    """Tolerance for horizontal normals, coplanarity and rectangular top area."""
    thickness: float = 0.006
    """Visible overlay thickness in metres above the physical supporting mesh."""
    opacity: float = 0.45
    """Opacity of the translucent placement-center overlay."""

    def for_body(self, body: Body, surface: HasSupportingSurface) -> list[MarkerEntry]:
        """Freeze supported, obstacle-free geometric-center positions in world coordinates.

        :param body: Object whose local bounding box defines placement clearance.
        :param surface: Annotated horizontal rectangular outer tabletop.
        :return: Frozen rectangles; empty when this surface is unsupported or full.
        """
        if not isinstance(surface, (Table, CounterTop)):
            return []
        if surface.supporting_surface is None:
            with self.world.modify_world():
                if surface.supporting_surface is None:
                    surface.calculate_supporting_surface()
        with self.world.state.world_lock:
            return self._markers(body, surface)

    def _markers(self, body: Body, surface: HasSupportingSurface) -> list[MarkerEntry]:
        """Compute a coherent snapshot while the digital twin state is locked.

        :param body: Object whose geometry must fit on the surface.
        :param surface: Annotated supporting surface being queried.
        """
        mesh = body.combined_mesh
        if surface.supporting_surface is None or mesh is None or mesh.is_empty:
            return []
        world_T_surface = surface.root.global_transform.to_np()
        if not np.allclose(
            world_T_surface[:3, 2], [0, 0, 1], atol=self.tolerance, rtol=0
        ):
            return []
        top = self._physical_top(surface)
        if top is None:
            return []
        object_bounds = BoundingBox.from_mesh(
            mesh, HomogeneousTransformationMatrix(reference_frame=body)
        )
        support = self._supported_centers(surface, top, object_bounds)
        if not support.shapes:
            return []
        obstacles = self._obstacles(body, surface, top.max_z, object_bounds)
        # Obstacle heights were filtered against the resting object's full height;
        # the support itself is only the plane on which centers may be placed.
        available = GraphOfBoundingBoxes.free_space_from_bounding_boxes(
            obstacles, support.event.marginal(SpatialVariables.xy), keep_z=False
        )
        return self._draw(available, surface, top.max_z, world_T_surface)

    def _physical_top(self, surface: HasSupportingSurface) -> BoundingBox | None:
        """Locate a complete rectangular top patch on the physical root mesh.

        :param surface: Tabletop whose mesh determines the true supporting height.
        """
        mesh = surface.root.combined_mesh
        if mesh is None or mesh.is_empty:
            return None
        top = float(mesh.bounds[1, 2])
        faces = np.all(np.abs(mesh.triangles[:, :, 2] - top) <= self.tolerance, axis=1)
        faces &= np.abs(mesh.face_normals[:, 2] - 1) <= self.tolerance
        if not faces.any():
            return None
        vertices = mesh.triangles[faces].reshape(-1, 3)
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        area = float((maximum[0] - minimum[0]) * (maximum[1] - minimum[1]))
        if area <= 0 or not np.isclose(
            float(mesh.area_faces[faces].sum()),
            area,
            atol=self.tolerance,
            rtol=self.tolerance,
        ):
            return None
        return BoundingBox(
            float(minimum[0]),
            float(minimum[1]),
            top,
            float(maximum[0]),
            float(maximum[1]),
            top,
            HomogeneousTransformationMatrix(reference_frame=surface.root),
        )

    def _supported_centers(
        self,
        surface: HasSupportingSurface,
        top: BoundingBox,
        object_bounds: BoundingBox,
    ) -> BoundingBoxCollection:
        """Intersect semantic support with the mesh top and remove footprint overhang.

        :param surface: Annotation supplying the semantic support footprint.
        :param top: Physical rectangular top in the surface-root frame.
        :param object_bounds: Candidate dimensions in its own local frame.
        """
        half_x, half_y = object_bounds.depth / 2, object_bounds.width / 2
        support = surface.supporting_surface.area.as_bounding_box_collection_in_frame(
            surface.root
        )
        boxes = []
        for box in support:
            lower_x = max(top.min_x, box.x_interval.lower) + half_x
            lower_y = max(top.min_y, box.y_interval.lower) + half_y
            upper_x = min(top.max_x, box.x_interval.upper) - half_x
            upper_y = min(top.max_y, box.y_interval.upper) - half_y
            if lower_x >= upper_x or lower_y >= upper_y:
                continue
            boxes.append(
                BoundingBox(
                    lower_x,
                    lower_y,
                    top.max_z,
                    upper_x,
                    upper_y,
                    top.max_z,
                    HomogeneousTransformationMatrix(reference_frame=surface.root),
                )
            )
        return BoundingBoxCollection(shapes=boxes, reference_frame=surface.root)

    def _obstacles(
        self,
        body: Body,
        surface: HasSupportingSurface,
        top: float,
        object_bounds: BoundingBox,
    ) -> BoundingBoxCollection:
        """Collect clearance-expanded obstacles intersecting the object's resting height.

        :param body: Candidate excluded from its own collision checks.
        :param surface: Supporting body excluded from obstacle subtraction.
        :param top: Physical supporting height in the surface-root frame.
        :param object_bounds: Candidate dimensions defining horizontal and vertical clearance.
        """
        boxes = []
        for obstacle in self.world.bodies_with_collision:
            if obstacle is body or obstacle is surface.root:
                continue
            for box in obstacle.collision.as_bounding_box_collection_in_frame(
                surface.root
            ):
                if (
                    box.z_interval.upper <= top
                    or box.z_interval.lower >= top + object_bounds.height
                ):
                    continue
                boxes.append(
                    box.bloat(object_bounds.depth / 2, object_bounds.width / 2, 0)
                )
        return BoundingBoxCollection(shapes=boxes, reference_frame=surface.root)

    def _draw(
        self,
        available: Event,
        surface: HasSupportingSurface,
        top: float,
        world_T_surface: np.ndarray,
    ) -> list[MarkerEntry]:
        """Render free event rectangles at the real tabletop height in world coordinates.

        :param available: Allowed geometric-center positions in the surface-root frame.
        :param surface: Root frame supplying the rectangle orientation.
        :param top: Physical supporting height in that frame.
        :param world_T_surface: Frozen transform from the surface-root frame to the world.
        """
        markers = []
        quaternion = rounded_pose(surface.root)[3:]
        for simple_event in available.simple_sets:
            for x_interval, y_interval in product(
                simple_event[SpatialVariables.x.value].simple_sets,
                simple_event[SpatialVariables.y.value].simple_sets,
            ):
                width = float(x_interval.upper - x_interval.lower)
                depth = float(y_interval.upper - y_interval.lower)
                if width <= 0 or depth <= 0:
                    continue
                position = world_T_surface @ np.array(
                    [
                        (x_interval.lower + x_interval.upper) / 2,
                        (y_interval.lower + y_interval.upper) / 2,
                        top + self.thickness / 2,
                        1,
                    ],
                    dtype=float,
                )
                markers.append(
                    MarkerEntry(
                        ns=PlacementOverlayStyle.NAMESPACE,
                        id=len(markers),
                        kind=PlacementOverlayStyle.SHAPE,
                        frame=str(self.world.root.name),
                        position=position[:3].tolist(),
                        quaternion=list(quaternion),
                        scale=[width, depth, self.thickness],
                        color=PlacementOverlayStyle.COLOR,
                        opacity=self.opacity,
                    )
                )
        return markers
