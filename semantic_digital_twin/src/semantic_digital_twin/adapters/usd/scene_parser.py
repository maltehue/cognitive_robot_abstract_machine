from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rustworkx

from semantic_digital_twin.adapters.usd import parser as usd
from semantic_digital_twin.adapters.usd.exceptions import (
    UsdSceneTopologyError,
    UsdScaledJointFrameError,
)
from semantic_digital_twin.adapters.usd.parser import USDParser
from semantic_digital_twin.adapters.world_model_parser import JointDescription
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.world_entity import Body

# %% scene objects and articulations


@dataclass
class USDSceneParser(USDParser):
    """
    Import separate scene objects and tree-shaped articulations in stage coordinates.

    Geometry-owning prims and joint targets become bodies named by their stage paths.
    Grouping transforms position those bodies; unconstrained bodies are fixed to their
    nearest enclosing body or the stage frame. Physics joints use their authored local
    anchors and start at zero position, as in :class:`USDParser`. Scaled joint frames
    and closed-loop constraints are rejected.
    """

    _body_paths: set[str] = field(default_factory=set, init=False)
    """
    Object boundaries excluded from an enclosing body's geometry traversal.
    """

    def parse(self) -> World:
        """
        Build the scene without flattening objects or discarding unrelated geometry.
        """
        joint_prims = [
            prim for prim in self.stage.Traverse() if prim.IsA(usd.UsdPhysics.Joint)
        ]
        object_prims = self._collect_object_prims(joint_prims)
        self._body_paths = {str(path) for path in object_prims}
        bodies: dict[str, Body] = {}
        for path, prim in object_prims.items():
            body = self._create_link_body(
                prim, self._resolve_rigid_world_transform(prim)
            )
            body.name = PrefixedName(str(path).lstrip("/"), self.prefix)
            bodies[str(path)] = body

        # A stage frame is distinct from authored prims, including the default prim.
        world = World.create_with_root_body(
            self.stage.GetPseudoRoot().GetName(), self.prefix
        )
        descriptions = [
            self._describe_joint(prim, world.root, bodies) for prim in joint_prims
        ]
        self._complete_connections(world.root, object_prims, bodies, descriptions)
        self._validate_tree(world.root, descriptions)
        with world.modify_world():
            for path, body in bodies.items():
                world.add_body(body)
                self._attach_semantic_labels(
                    world, self.stage.GetPrimAtPath(path), body
                )
            for description in descriptions:
                world.add_connection(self._create_connection(world, description))
        return world

    def _resolve_object_prim(self, prim_path: usd.Sdf.Path) -> usd.Usd.Prim:
        """
        Resolve a joint target or geometry prim to its enclosing object.
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or prim.IsPseudoRoot() or prim.IsA(usd.UsdPhysics.Joint):
            raise UsdSceneTopologyError(
                file_path=self.source_description, prim_path=str(prim_path)
            )
        if prim.IsA(usd.UsdGeom.Gprim) and not prim.GetParent().IsPseudoRoot():
            return prim.GetParent()
        return prim

    def _collect_object_prims(
        self, joint_prims: list[usd.Usd.Prim]
    ) -> dict[usd.Sdf.Path, usd.Usd.Prim]:
        """
        Find geometry owners and validated joint endpoints throughout the stage.
        """
        objects: dict[usd.Sdf.Path, usd.Usd.Prim] = {}
        for prim in self.stage.Traverse():
            if (
                prim.IsA(usd.UsdGeom.Gprim)
                and usd.UsdGeom.Imageable(prim).ComputePurpose()
                != usd.UsdGeom.Tokens.guide
            ):
                owner = self._resolve_object_prim(prim.GetPath())
                objects[owner.GetPath()] = owner
        for prim in joint_prims:
            joint = usd.UsdPhysics.Joint(prim)
            for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                targets = relationship.GetTargets()
                if len(targets) > 1:
                    raise UsdSceneTopologyError(
                        file_path=self.source_description, prim_path=str(prim.GetPath())
                    )
                for target in targets:
                    owner = self._resolve_object_prim(target)
                    # Joint anchors are local rigid poses; scaled frames need a separate conversion.
                    authored = usd.UsdGeom.Xformable(
                        owner
                    ).ComputeLocalToWorldTransform(usd.Usd.TimeCode.Default())
                    if not np.allclose(
                        np.array(authored),
                        np.array(self._resolve_rigid_world_transform(owner)),
                    ):
                        raise UsdScaledJointFrameError(
                            file_path=self.source_description, prim_path=str(target)
                        )
                    objects[owner.GetPath()] = owner
        return objects

    def _resolve_link_body(
        self, link_bodies: dict[str, Body], prim_path: usd.Sdf.Path
    ) -> Body:
        """
        Return the already-created body for a validated joint endpoint.
        """
        return link_bodies[str(self._resolve_object_prim(prim_path).GetPath())]

    def _shapes_in_subtree(
        self, link_prim: usd.Usd.Prim, link_to_world: usd.Gf.Matrix4d | None = None
    ) -> list[Shape]:
        """
        Build this object's geometry while excluding nested object boundaries.
        """
        if link_to_world is None:
            link_to_world = self._resolve_rigid_world_transform(link_prim)
        shapes = []
        iterator = iter(usd.Usd.PrimRange(link_prim))
        for prim in iterator:
            if prim != link_prim and str(prim.GetPath()) in self._body_paths:
                iterator.PruneChildren()
                continue
            if (
                prim.IsA(usd.UsdGeom.Gprim)
                and usd.UsdGeom.Imageable(prim).ComputePurpose()
                == usd.UsdGeom.Tokens.guide
            ):
                continue
            shape = self._create_shape(prim, link_to_world)
            if shape is not None:
                shapes.append(shape)
        return shapes

    def _complete_connections(
        self,
        root_body: Body,
        object_prims: dict[usd.Sdf.Path, usd.Usd.Prim],
        bodies: dict[str, Body],
        descriptions: list[JointDescription],
    ) -> None:
        """
        Attach objects not controlled by physics joints to their enclosing body.
        """
        connected_children = set()
        for description in descriptions:
            if description.child in connected_children:
                raise UsdSceneTopologyError(
                    file_path=self.source_description,
                    prim_path=str(description.child.name),
                )
            connected_children.add(description.child)
        for path, prim in object_prims.items():
            child = bodies[str(path)]
            if child in connected_children:
                continue
            parent_prim = prim.GetParent()
            while (
                not parent_prim.IsPseudoRoot()
                and parent_prim.GetPath() not in object_prims
            ):
                parent_prim = parent_prim.GetParent()
            parent = (
                root_body
                if parent_prim.IsPseudoRoot()
                else bodies[str(parent_prim.GetPath())]
            )
            world_T_parent = (
                usd.Gf.Matrix4d(1.0)
                if parent is root_body
                else self._resolve_rigid_world_transform(parent_prim)
            )
            parent_T_child = (
                self._resolve_rigid_world_transform(prim) * world_T_parent.GetInverse()
            )
            descriptions.append(
                JointDescription(
                    connection_type=FixedConnection,
                    parent=parent,
                    child=child,
                    parent_T_connection=HomogeneousTransformationMatrix(
                        data=np.array(parent_T_child).T,
                        reference_frame=parent,
                        child_frame=child,
                    ),
                    connection_T_child=HomogeneousTransformationMatrix(),
                )
            )

    def _validate_tree(
        self, root_body: Body, descriptions: list[JointDescription]
    ) -> None:
        """
        Reject cycles before the scene is added to the world.
        """
        graph = rustworkx.PyDiGraph()
        bodies = {root_body}
        bodies.update(description.child for description in descriptions)
        indices = {body: graph.add_node(body) for body in bodies}
        for description in descriptions:
            graph.add_edge(
                indices[description.parent], indices[description.child], description
            )
        cycle = rustworkx.digraph_find_cycle(graph)
        if not cycle:
            return
        _, child_index = cycle[0]
        raise UsdSceneTopologyError(
            file_path=self.source_description, prim_path=str(graph[child_index].name)
        )
