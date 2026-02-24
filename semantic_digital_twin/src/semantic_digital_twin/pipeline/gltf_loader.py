from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import re

import numpy as np
import trimesh

from .pipeline import Step
from ..world import World
from ..world_description.world_entity import Body
from ..world_description.connections import FixedConnection
from ..world_description.shape_collection import ShapeCollection
from ..world_description.geometry import TriangleMesh, Scale
from ..spatial_types import HomogeneousTransformationMatrix
from ..datastructures.prefixed_name import PrefixedName


@dataclass
class NodeProcessingResult:
    """Result of processing a single node in the scene graph."""

    body: Body
    visited_nodes: Set[str]
    children_to_visit: Set[str]


@dataclass
class GLTFLoader(Step):
    """Load GLTF/GLB files into a World.

    This loader parses GLTF/GLB files (including FreeCAD exports) and creates
    Body objects with FixedConnection relationships matching the scene hierarchy.

    Features:
    - Handles FreeCAD naming conventions (e.g., Bolt_001, Bolt_002 are fused)
    - Applies node transformations correctly
    - Skips non-geometry nodes while preserving hierarchy
    - Creates proper parent-child connections

    Example:
        >>> world = World()
        >>> loader = GLTFLoader(file_path="model.gltf")
        >>> world = loader.apply(world)

    Limitations:
    - Only creates FixedConnection (no joints/articulations)
    - Does not handle GLTF extensions for physics/joints

    Attributes:
        file_path: Path to the GLTF/GLB file
        scene: The loaded trimesh Scene (set after _apply is called)

    Raises:
        ValueError: If the file cannot be loaded or parsed.
    """

    file_path: str
    """Path to the GLTF/GLB file."""

    scene: Optional[trimesh.Scene] = field(default=None, init=False)
    """The loaded trimesh Scene (set after _apply is called)."""

    def _get_root_node(self) -> str:
        base_frame = self.scene.graph.base_frame
        root_children = self.scene.graph.transforms.children.get(base_frame, [])
        if len(root_children) > 1 or len(root_children) == 0:
            raise ValueError(
                "More than one root node found in the scene, or no root node found."
            )
        return root_children[0]

    def _get_relative_transform(
        self, parent_node: str, child_node: str
    ) -> HomogeneousTransformationMatrix:
        """Get the relative transform from parent to child node.

        Computes the transform that converts from parent frame to child frame.

        Args:
            parent_node: Name of the parent node
            child_node: Name of the child node

        Returns:
            The relative transformation matrix from parent to child
        """
        parent_transform, _ = self.scene.graph.get(parent_node)
        child_transform, _ = self.scene.graph.get(child_node)

        # Compute relative transform: parent_inv @ child
        parent_inv = np.linalg.inv(parent_transform)
        relative = parent_inv @ child_transform

        return HomogeneousTransformationMatrix(relative)

    def _trimesh_to_body(self, mesh: trimesh.Trimesh, name: str) -> Body:
        """Convert a trimesh.Trimesh to a Body object."""
        # Create TriangleMesh geometry from trimesh
        triangle_mesh = TriangleMesh(
            mesh=mesh,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),  # Identity transform
            scale=Scale(1.0, 1.0, 1.0),  # No scaling
        )

        # Create ShapeCollection for collision and visual
        shape_collection = ShapeCollection([triangle_mesh])

        # Create Body
        body = Body(
            name=PrefixedName(name),
            collision=shape_collection,
            visual=shape_collection,  # Use same for both collision and visual
        )

        return body

    def _extract_base_name(self, node_name: str) -> str:
        """Extract base name by removing FreeCAD suffixes like _001, _002."""
        match = re.match(r"^(.+?)(?:_\d+|_[A-Za-z]\d*)?$", str(node_name))
        return match.group(1) if match else str(node_name)

    def _collect_matching_children(
            self, node: str, base_name: str, object_nodes: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Collect children that match base_name and those that don't."""
        matching = set()
        non_matching = set()
        for child in self.scene.graph.transforms.children.get(node, []):
            if child in object_nodes:
                continue
            if self._extract_base_name(child) == base_name:
                matching.add(child)
            else:
                non_matching.add(child)
        return matching, non_matching

    def _grouping_similar_meshes(self, base_node: str) -> Tuple[Set[str], Set[str]]:
        """Group meshes with similar names (e.g., Bolt_001, Bolt_002 -> Bolt)."""
        base_name = self._extract_base_name(base_node)
        object_nodes = {base_node}
        new_object_nodes = set()
        to_search = [base_node]
        max_iterations = 10000

        for _ in range(max_iterations):
            if not to_search:
                break
            node = to_search.pop()
            matching, non_matching = self._collect_matching_children(
                node, base_name, object_nodes
            )
            object_nodes.update(matching)
            to_search.extend(matching)
            new_object_nodes.update(non_matching)
        else:
            print(f"Warning: Hit max iterations in _grouping_similar_meshes for {base_node}")

        return object_nodes, new_object_nodes

    def _fusion_meshes(self, object_nodes: Set[str]) -> trimesh.Trimesh:
        """Fuse multiple mesh nodes into a single mesh.

        Applies the world transform to each mesh before concatenating them.

        Args:
            object_nodes: Set of node names to fuse

        Returns:
            A single concatenated mesh, or empty Trimesh if no geometry found
        """
        meshes: List[trimesh.Trimesh] = []
        for node in object_nodes:
            transform, geometry_name = self.scene.graph.get(node)
            if geometry_name is None:
                continue
            geometry = self.scene.geometry.get(geometry_name)
            if geometry is None:
                continue
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
        if meshes:
            return trimesh.util.concatenate(meshes)  # type: ignore[return-value]
        return trimesh.Trimesh()  # Empty mesh if no geometry found

    def _add_child_connection(
            self, world: World, parent_node: str, child_node: str,
            world_elements: Dict[str, Body]
    ) -> None:
        """Add a child body and its connection to the world."""
        if child_node not in world_elements or parent_node not in world_elements:
            return
        parent_body = world_elements[parent_node]
        child_body = world_elements[child_node]
        world.add_kinematic_structure_entity(child_body)
        relative_transform = self._get_relative_transform(parent_node, child_node)
        conn = FixedConnection(
            parent=parent_body,
            child=child_body,
            parent_T_connection_expression=relative_transform,
            name=PrefixedName(f"{parent_node}_{child_node}"),
        )
        world.add_connection(conn)

    def _build_world_from_elements(
        self,
        world_elements: Dict[str, Body],
        connection: Dict[str, List[str]],
        world: World,
    ) -> World:
        """Build the world from parsed elements and their connections.

        Args:
            world_elements: Dictionary mapping node names to Body objects
            connection: Dictionary mapping parent node names to list of child node names
            world: The world to add entities to

        Returns:
            The modified world
        """
        object_root = self._get_root_node()
        if object_root not in world_elements:
            raise ValueError(f"Root node '{object_root}' not found in world_elements")

        object_root_body = world_elements[object_root]
        world.add_kinematic_structure_entity(object_root_body)

        # Connect root to world root if exists
        if world.root is not None and world.root != object_root_body:
            root_transform, _ = self.scene.graph.get(object_root)
            conn = FixedConnection(
                parent=world.root,
                child=object_root_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix(root_transform),
                name=PrefixedName(f"object_root_{object_root}"),
            )
            world.add_connection(conn)

        # Add all child connections via BFS
        to_add_nodes = [object_root]
        while to_add_nodes:
            node = to_add_nodes.pop()
            for child in connection.get(node, []):
                to_add_nodes.append(child)
                self._add_child_connection(world, node, child, world_elements)

        return world

    def _create_empty_body(self, name: str) -> Body:
        """Create an empty body with no geometry."""
        return Body(
            name=PrefixedName(name),
            collision=ShapeCollection([]),
            visual=ShapeCollection([]),
        )

    def _process_geometry_node(
        self, node: str, visited_nodes: Set[str]
    ) -> Optional[NodeProcessingResult]:
        """Process a geometry node and return the result.

        Returns None if the node has no valid geometry (empty mesh).
        """
        object_nodes, children = self._grouping_similar_meshes(node)
        mesh = self._fusion_meshes(object_nodes)

        if len(mesh.vertices) == 0:
            return None

        body = self._trimesh_to_body(mesh, node)
        return NodeProcessingResult(
            body=body,
            visited_nodes=object_nodes,
            children_to_visit=children.difference(visited_nodes),
        )

    def _process_root_node(
            self, root: str
    ) -> Tuple[Body, Set[str], Set[Tuple[str, str]]]:
        """Process the root node and return body, visited nodes, and children to visit."""
        _, root_geometry = self.scene.graph.get(root)
        children_raw = self.scene.graph.transforms.children.get(root, [])

        if root_geometry is None:
            body = self._create_empty_body(root)
            return body, {root}, {(child, root) for child in children_raw}

        result = self._process_geometry_node(root, set())
        if result and len(result.body.visual.shapes) > 0:
            return result.body, result.visited_nodes, {(c, root) for c in result.children_to_visit}

        # Root has empty geometry
        body = self._create_empty_body(root)
        object_nodes, children = self._grouping_similar_meshes(root)
        return body, object_nodes, {(child, root) for child in children}

    def _handle_non_geometry_node(
            self, node: str, body_parent: str, visited_nodes: Set[str]
    ) -> Set[Tuple[str, str]]:
        """Handle a non-geometry node, returning children to visit."""
        children_to_visit = set()
        for child in self.scene.graph.transforms.children.get(node, []):
            if child not in visited_nodes:
                children_to_visit.add((child, body_parent))
        return children_to_visit

    def _handle_empty_geometry_node(
            self, node: str, body_parent: str, visited_nodes: Set[str]
    ) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        """Handle a node with empty geometry, returning updated visited and children."""
        object_nodes, children = self._grouping_similar_meshes(node)
        new_visited = object_nodes | {node}
        children_to_visit = {(c, body_parent) for c in children.difference(visited_nodes)}
        return new_visited, children_to_visit

    def _create_world_objects(self, world: World) -> World:
        """Parse the scene graph and create world objects with connections.

        This method traverses the scene graph, groups similar meshes (e.g., Bolt_001, Bolt_002),
        fuses them and creates Body objects with parent-child connections.

        Non-geometry nodes (like transforms/sketches) are skipped but their children
        are still processed with the correct parent body.
        """
        root = self._get_root_node()
        world_elements: Dict[str, Body] = {}
        connection: Dict[str, List[str]] = {}
        visited_nodes: Set[str] = set()

        # Process root
        root_body, root_visited, to_visit_new_node = self._process_root_node(root)
        world_elements[root] = root_body
        visited_nodes = visited_nodes.union(root_visited)


        while to_visit_new_node:
            node, body_parent = to_visit_new_node.pop()
            if node in visited_nodes:
                continue

            _, geometry_name = self.scene.graph.get(node)

            # Non-geometry node: pass through to children
            if geometry_name is None:
                to_visit_new_node.update(self._handle_non_geometry_node(node, body_parent, visited_nodes))
                visited_nodes.add(node)
                continue

            result = self._process_geometry_node(node, visited_nodes)
            if result is None:
                new_visited, children = self._handle_empty_geometry_node(node, body_parent, visited_nodes)
                visited_nodes.update(new_visited)
                to_visit_new_node.update(children)
                continue

            # Register body and connection
            world_elements[node] = result.body
            visited_nodes.update(result.visited_nodes | {node})
            connection.setdefault(body_parent, []).append(node)
            connection[node] = []
            to_visit_new_node.update((c, node) for c in result.children_to_visit)

        return self._build_world_from_elements(world_elements, connection, world)

    def _apply(self, world: World) -> World:
        """Load GLTF/GLB file and create world objects."""
        try:
            self.scene = trimesh.load(self.file_path)  # type: ignore[assignment]
        except Exception as e:
            raise ValueError(f"Failed to load file '{self.file_path}': {e}") from e

        # Handle case where trimesh loads a single mesh instead of a Scene
        if isinstance(self.scene, trimesh.Trimesh):
            mesh = self.scene
            self.scene = trimesh.Scene()
            self.scene.add_geometry(mesh, node_name="root", geom_name="root_geom")

        if len(self.scene.geometry) == 0:
            root = self._get_root_node()
            world.add_kinematic_structure_entity(self._create_empty_body(root))
            return world

        return self._create_world_objects(world)
