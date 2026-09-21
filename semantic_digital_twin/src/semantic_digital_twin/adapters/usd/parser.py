from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Dict, List, Type

from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedUsdPhysicsJointTypeError,
    UsdPhysicsJointMissingChildBodyError,
)
from semantic_digital_twin.adapters.usd.stage_parser import (
    Sdf,
    Usd,
    UsdAxis,
    UsdGeom,
    UsdGeomPrimType,
    UsdPhysics,
    USDStageParser,
    _usd_pose_to_transform,
)
from semantic_digital_twin.adapters.world_model_parser import JointDescription
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection,
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% joint types


class UsdPhysicsJointType(StrEnum):
    """
    The USD physics joint prim type names this parser builds a Connection for.
    """

    FIXED = "PhysicsFixedJoint"
    REVOLUTE = "PhysicsRevoluteJoint"
    PRISMATIC = "PhysicsPrismaticJoint"

    @property
    def connection_type(self) -> Type[Connection]:
        """
        :return: The Connection class this joint type becomes.
        """
        return {
            UsdPhysicsJointType.FIXED: FixedConnection,
            UsdPhysicsJointType.REVOLUTE: RevoluteConnection,
            UsdPhysicsJointType.PRISMATIC: PrismaticConnection,
        }[self]


# %% parser


@dataclass
class USDParser(USDStageParser):
    """
    Parses a USD stage describing one physically articulated asset into a world.

    A USD stage's arbitrary ``Xform`` nesting is scene organisation, not rigid-body
    structure: the rigid links of a physically articulated asset are the prims
    connected by its :class:`~pxr.UsdPhysics.Joint` prims (a joint's ``body0``/``body1``
    relationships), so those - not the raw prim hierarchy - are what is walked to build
    the kinematic tree. A stage with no physics joints at all is treated as one rigid
    body: every mesh and primitive shape under its root becomes a shape of a single
    root Body.

    .. note::
        Every rigid link is assumed to appear as the child (``body1``) of exactly one
        joint - a link that does not would be created (as another joint's parent) but
        never connected, and so would not appear in the parsed world.

    .. note::
        A stage holding separately placed static objects, whose structure is its
        ``Xform`` tree rather than physics joints, is read by
        :class:`~semantic_digital_twin.adapters.usd.scene_parser.USDSceneParser`
        instead.
    """

    # %% entry point

    def parse(self) -> World:
        """
        Parses the stage into a world.

        :return: The parsed world.
        :raises UnsupportedUsdPhysicsJointTypeError: If the stage contains a physics
            joint of a type this parser does not build a Connection for.
        :raises UsdPhysicsJointMissingChildBodyError: If a physics joint's ``body1``
            relationship has no target.
        """
        joint_prims = [
            prim for prim in self.stage.Traverse() if prim.IsA(UsdPhysics.Joint)
        ]
        if not joint_prims:
            return self._parse_jointless_stage()
        return self._parse_joint_graph(joint_prims)

    def _parse_joint_graph(self, joint_prims: List[Usd.Prim]) -> World:
        """
        Builds the world for a stage with physics joints.

        The joint graph's own root (a joint's unset ``body0``, the USD convention for
        "the stage's own frame") becomes a Body named after :meth:`_root_prim`, or a
        synthetic one if that is ambiguous - unlike a joint's ``body1``, which always
        names a specific link, an unset ``body0`` never depended on the stage having an
        identifiable root prim in the first place, so there is nothing to name it after.

        :param joint_prims: Every ``UsdPhysics.Joint`` prim in the stage.
        :return: The parsed world.
        """
        root_prim = self._root_prim()
        root_body_name = root_prim.GetName() if root_prim is not None else self.prefix
        world = World.create_with_root_body(root_body_name, self.prefix)
        root_body = world.root

        # Every joint is described (and so validated) before the world is touched
        # further: a World left partway through a failed modification is unusable, so
        # anything that can raise must run before entering another modify_world block.
        link_bodies: Dict[str, Body] = {}
        descriptions = [
            self._describe_joint(joint_prim, root_body, link_bodies)
            for joint_prim in joint_prims
        ]

        with world.modify_world():
            if root_prim is not None:
                self._attach_semantic_labels(world, root_prim, root_body)
            for path_string, link_body_instance in link_bodies.items():
                world.add_body(link_body_instance)
                self._attach_semantic_labels(
                    world, self.stage.GetPrimAtPath(path_string), link_body_instance
                )
            for description in descriptions:
                world.add_connection(self._create_connection(world, description))

        return world

    def _parse_jointless_stage(self) -> World:
        """
        Builds the world for a stage with no physics joints at all.

        Its root prim (see :meth:`_root_prim`) becomes a single-body World. With no
        identifiable root prim, there is nothing to unambiguously treat as "the" object
        either - so a synthetic root is created instead, and every top-level prim
        becomes its own body, attached to it with a
        :class:`~semantic_digital_twin.world_description.connections.Connection6DoF`
        (freely posable, since the stage itself asserts no relationship between them).

        :return: The parsed world.
        """
        root_prim = self._root_prim()

        if root_prim is not None:
            world = World.create_with_root_body(root_prim.GetName(), self.prefix)
            root_body = world.root
            shapes = self._shapes_in_subtree(root_prim)
            shape_collection = ShapeCollection(shapes, reference_frame=root_body)
            shape_collection.transform_all_shapes_to_own_frame()
            root_body.visual = shape_collection
            root_body.collision = shape_collection
            with world.modify_world():
                self._attach_semantic_labels(world, root_prim, root_body)
            return world

        world = World.create_with_root_body(self.prefix, self.prefix)
        root_body = world.root
        with world.modify_world():
            for prim in self._top_level_prims():
                body = self._create_link_body(prim)
                world.add_body(body)
                self._attach_semantic_labels(world, prim, body)
                world.add_connection(
                    Connection6DoF.create_with_dofs(
                        world=world, parent=root_body, child=body
                    )
                )
        return world

    # %% joints

    def _describe_joint(
        self,
        joint_prim: Usd.Prim,
        root_body: Body,
        link_bodies: Dict[str, Body],
    ) -> JointDescription:
        """
        Validates and describes the Connection one physics joint prim becomes, without
        touching a world - a World left partway through a failed modification is
        unusable, so every joint is described before any world modification begins.

        :param joint_prim: The USD physics joint prim (a Fixed/Revolute/PrismaticJoint).
        :param root_body: The world's root body, used as the connection's parent if the
            joint's ``body0`` relationship has no target.
        :param link_bodies: The link Bodies created so far, by stage path - extended in
            place as new links are resolved.
        :return: The description of the connection the joint becomes.
        :raises UnsupportedUsdPhysicsJointTypeError: If the joint's type has no
            Connection counterpart.
        :raises UsdPhysicsJointMissingChildBodyError: If the joint's ``body1``
            relationship has no target.
        """
        try:
            usd_joint_type = UsdPhysicsJointType(joint_prim.GetTypeName())
        except ValueError as error:
            raise UnsupportedUsdPhysicsJointTypeError(
                file_path=self.source_description,
                joint_path=str(joint_prim.GetPath()),
                joint_type=joint_prim.GetTypeName(),
                supported_types=list(UsdPhysicsJointType),
            ) from error
        connection_type = usd_joint_type.connection_type

        joint = UsdPhysics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if not body1_targets:
            raise UsdPhysicsJointMissingChildBodyError(
                file_path=self.source_description, joint_path=str(joint_prim.GetPath())
            )
        parent = (
            self._resolve_link_body(link_bodies, body0_targets[0])
            if body0_targets
            else root_body
        )
        child = self._resolve_link_body(link_bodies, body1_targets[0])

        parent_T_connection = _usd_pose_to_transform(
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
            reference_frame=parent,
        )
        # UsdPhysics.Joint documents localPos1/localRot1 as the joint frame's pose
        # relative to body1 (child_T_connection), the opposite of what is needed here -
        # unlike localPos0/localRot0, which is already parent_T_connection as authored.
        child_T_connection = _usd_pose_to_transform(
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
            reference_frame=child,
        )
        connection_T_child = child_T_connection.inverse()

        if connection_type is FixedConnection:
            return JointDescription(
                connection_type=connection_type,
                parent=parent,
                child=child,
                parent_T_connection=parent_T_connection,
                connection_T_child=connection_T_child,
            )

        axis_joint = (
            UsdPhysics.RevoluteJoint(joint_prim)
            if connection_type is RevoluteConnection
            else UsdPhysics.PrismaticJoint(joint_prim)
        )
        axis = UsdAxis(axis_joint.GetAxisAttr().Get()).vector(reference_frame=parent)
        lower = axis_joint.GetLowerLimitAttr().Get()
        upper = axis_joint.GetUpperLimitAttr().Get()
        if connection_type is RevoluteConnection:
            lower, upper = math.radians(lower), math.radians(upper)
        if lower > upper:
            # Seen authored this way on a mirrored part (e.g. one blade of a pair of
            # scissors): the pair's joints share one axis convention, so the mirrored
            # joint's authored "lower"/"upper" swap relative to it even though both
            # describe the same-sized range of motion. The DOF's own lower/upper are
            # just its two extremes, so swapping the values (not negating them) keeps
            # the authored range of motion intact.
            lower, upper = upper, lower
        return JointDescription(
            connection_type=connection_type,
            parent=parent,
            child=child,
            parent_T_connection=parent_T_connection,
            connection_T_child=connection_T_child,
            axis=axis,
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=lower), upper=DerivativeMap(position=upper)
            ),
        )

    def _resolve_link_body(
        self, link_bodies: Dict[str, Body], prim_path: Sdf.Path
    ) -> Body:
        """
        Resolves a USD prim path to its link Body, creating (and caching in
        ``link_bodies``) the Body on first use.

        :param link_bodies: The link Bodies created so far, by stage path - extended in
            place if ``prim_path`` has not been resolved yet.
        :param prim_path: The stage path a joint's ``body0``/``body1`` relationship
            targets.
        :return: The resolved link Body.
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim.GetTypeName() == UsdGeomPrimType.MESH:
            # Most joints target the link's enclosing Xform, whose subtree holds its
            # shape(s); some instead target a link's mesh prim directly. Both resolve
            # to the same parent Xform, so a link is never split into two disconnected
            # bodies.
            prim = prim.GetParent()
        path_string = str(prim.GetPath())
        if path_string not in link_bodies:
            link_bodies[path_string] = self._create_link_body(prim)
        return link_bodies[path_string]

    @staticmethod
    def _create_connection(world: World, description: JointDescription) -> Connection:
        """
        Creates the Connection a joint description denotes, adding its degree of freedom
        to the world.

        :param world: The world the degree of freedom is added to.
        :param description: The description of the joint.
        :return: The connection describing the joint.
        """
        if description.connection_type is FixedConnection:
            return FixedConnection.create_with_dofs(
                world=world,
                name=description.name,
                parent=description.parent,
                child=description.child,
                parent_T_connection_expression=description.parent_T_connection,
                connection_T_child_expression=description.connection_T_child,
            )
        connection = description.connection_type.create_with_dofs(
            world=world,
            name=description.name,
            parent=description.parent,
            child=description.child,
            parent_T_connection_expression=description.parent_T_connection,
            connection_T_child_expression=description.connection_T_child,
            axis=description.axis,
            dof_limits=description.limits,
        )
        connection.dynamics = description.dynamics
        return connection

    # %% links and shapes

    def _create_link_body(self, link_prim: Usd.Prim) -> Body:
        """
        Creates the Body for one rigid link, with a Shape for every mesh/primitive in
        its USD subtree and its :class:`~pxr.UsdPhysics.MassAPI` inertial properties, if
        applied.

        :param link_prim: The link's root USD prim.
        :return: The created body, not yet added to a world.
        """
        shapes = self._shapes_in_subtree(link_prim)
        shape_collection = ShapeCollection(shapes)
        body = Body(
            name=PrefixedName(link_prim.GetName(), self.prefix),
            visual=shape_collection,
            collision=shape_collection,
        )
        inertial = self._parse_inertial(link_prim, body)
        if inertial is not None:
            body.inertial = inertial
        return body

    def _shapes_in_subtree(self, link_prim: Usd.Prim) -> List[Shape]:
        """
        Creates the Shape for every mesh/primitive prim in a link's subtree.

        :param link_prim: The link the shapes are positioned relative to, and the root
            of the subtree to search for them.
        :return: The created shapes.
        """
        link_to_world = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        shapes = []
        for prim in Usd.PrimRange(link_prim):
            shape = self._create_shape(prim, link_to_world)
            if shape is not None:
                shapes.append(shape)
        return shapes
