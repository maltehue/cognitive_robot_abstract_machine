from __future__ import annotations

from dataclasses import dataclass

import trimesh
from typing_extensions import List

from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale

# %% enclosing a mesh in one box


@dataclass
class BoundingBoxDecomposer(MeshDecomposer):
    """
    Replaces a mesh with the single axis-aligned box enclosing it.

    The cheapest decomposition there is, and the only one that does not read a mesh's
    geometry: a glTF binary states the extent of its own vertex data in its header. A
    scanned surface carries far too many triangles to collide against, while the slabs
    a building is made of - walls, a floor - are already close to boxes.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Box]:
        """
        :param mesh: The mesh to enclose.
        :return: The enclosing box, placed where the mesh sits.
        """
        low, high = mesh.bounds
        center = (low + high) / 2.0
        mesh_T_box = HomogeneousTransformationMatrix.from_xyz_rpy(*center)
        return [Box(origin=mesh.origin @ mesh_T_box, scale=Scale(*(high - low)))]

    def apply_to_mesh_and_save(self, mesh: Mesh, output_path: str) -> str:
        """
        :param mesh: The mesh to enclose.
        :param output_path: The ``.obj`` file to write the enclosing box to.
        :return: The path written to.
        """
        [enclosing_box] = self.apply_to_mesh(mesh)
        box_mesh = trimesh.creation.box(extents=enclosing_box.scale.to_np())
        box_mesh.apply_transform(enclosing_box.origin.to_np())
        box_mesh.export(output_path, file_type="obj")
        return output_path
