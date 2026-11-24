"""
Mixins that give semantic annotations a shared trait, such as having a handle or moving
on a joint of their own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Callable, ClassVar, Generic, Self, TypeVar

import numpy as np
import trimesh
from trimesh.util import concatenate
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.entity_query_language.factories import variable_from, entity, variable, an
from krrood.patterns.classproperty import classproperty
from krrood.parametrization.model_registries import ModelRegistry
from krrood.patterns.mixin import Mixin
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_event,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from random_events.product_algebra import Event
from random_events.set import Set as EventSet
from random_events.variable import Symbolic

from semantic_digital_twin.specifications.connections import (
    ActiveConnection1DOFSpecification,
    ConnectionSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
    RegionSpecification,
    KinematicStructureEntitySpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    PartSpecificationBinding,
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    AmbiguousPart,
    CannotBeAPartOf,
    MissingMovableJointError,
    NoGraspGeometry,
    NoSupportingSurfaceError,
    UnknownPartWholeRelationshipField,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.grasping.grasp_strategies import (
    GraspFromAnySide,
    GraspStrategy,
    PinchTheTopOfAWall,
)
from semantic_digital_twin.grasping.object_relative_grasp import draw_grasps
from semantic_digital_twin.reasoning.predicates import SupportedBy
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from krrood_spatial_math.spatial_types import (
    LineSegment3,
    Point3,
    HomogeneousTransformationMatrix,
    Vector3,
)
from krrood_spatial_math.spatial_types import Pose, RotationMatrix
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ActiveConnection1DOF,
    LiquidConnection,
    LiquidTransferCoupling,
)
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
    Color,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
    KinematicStructureEntity,
    Connection,
    WorldEntityWithID,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)
import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import to_json
from krrood.patterns.field_metadata import JSONMetadata
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.exceptions import (
    FillLevelAlreadyInitializedError,
    MissingFillEquationError,
    ReceiverAlreadyCoupledError,
    ReceiverNotInitializedError,
    SourceAlreadyCoupledError,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    DEFAULT_DISCHARGE_COEFFICIENT,
    DEFAULT_GATE_SHARPNESS,
    DEFAULT_POUR_EXIT_SPEED,
    MINIMUM_DROP_HEIGHT,
    STANDARD_GRAVITY,
    ArticulatedPouringEquation,
    GatedArticulatedPouringEquation,
    GatedInflowEquation,
    InflowEquation,
    PouringEquation,
    TransferGate,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.derivatives import DerivativeMap

if TYPE_CHECKING:
    from semantic_digital_twin.grasping.object_relative_grasp import GraspResult
    from semantic_digital_twin.world import World
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Drawer,
        Door,
        Handle,
        Rim,
        Aperture,
        Leg,
        Sink,
        ShelfLayer,
        Wall,
        CounterTop,
        Cabinet,
    )
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
        PlanarGraphOfBoundingBoxes,
    )


@dataclass(eq=False)
class IsPerceivable:
    """
    A mixin class for semantic annotations that can be perceived.
    """

    class_label: str | None = field(default=None, kw_only=True)
    """
    The exact class label of the perceived object.
    """


TKinematicStructureEntity = TypeVar(
    "TKinematicStructureEntity", bound=KinematicStructureEntity
)


@dataclass(eq=False)
class HasRootKinematicStructureEntity(
    SemanticAnnotation, Generic[TKinematicStructureEntity], SubClassSafeGeneric, Mixin
):
    """
    Base class for shared method for HasRootBody and HasRootRegion.

    Building a specification takes two steps:
    :meth:`get_default_root_kinematic_structure_entity_specification` describes the root entity's geometry,
    and :meth:`get_annotation_specification` wraps that root into the spawnable
    annotation specification.

    .. code-block:: python

        Handle.get_annotation_specification(
            "handle",
            Handle.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(0.1, 0.05, 0.05), thickness=0.01
            ),
        )

    Every geometry parameter belongs to the first step, so a type's geometry is
    described in exactly one place.
    """

    root: TKinematicStructureEntity = field(kw_only=True)
    """
    The root kinematic structure entity of the semantic annotation.
    """

    @property
    def combined_mesh(self) -> trimesh.Trimesh:
        """
        :return: The collision geometry of every body of this annotation, merged into a single
        mesh expressed in the frame of :attr:`root`.

        ..note:: Rebuilt on every access, since the bodies move relative to each other
            with the world state.
        """
        return concatenate(
            [
                shape.mesh_in_frame(self.root)
                for body in self.bodies_with_collision
                for shape in body.collision
            ]
        )

    @property
    def scale(self) -> Scale:
        return Scale(
            *(self.root.combined_mesh.bounds[1] - self.root.combined_mesh.bounds[0])
        )

    @property
    def min_max_points(self) -> tuple[Point3, Point3]:
        min = Point3.from_iterable(self.root.combined_mesh.bounds[0])
        max = Point3.from_iterable(self.root.combined_mesh.bounds[1])
        return min, max

    def __hash__(self):
        return hash((self.__class__, self.root))

    @classmethod
    @abstractmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> KinematicStructureEntitySpecification:
        """
        Build this type's default root entity specification, the geometry only.

        Implemented once per root entity kind (:class:`HasRootBody` yields a body
        specification, :class:`HasRootRegion` a region one) and overridden by types
        whose geometry takes further parameters, such as a handle's ``thickness`` or a
        case's ``wall_thickness``. Those parameters live here and nowhere else, so a
        type's geometry is described in a single place.

        .. warning:: The result carries no annotation. Spawning it directly puts a bare
            body or region into the world; pass it to
            :meth:`get_annotation_specification` to obtain the annotation.

        :param name: The name of entities created from the specification. ``None``
            leaves naming to the spawning annotation, which overrides it with its own
            name.
        :param scale: The scale the geometry is generated from.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: The root entity specification.
        """

    @classmethod
    def get_annotation_specification(
        cls,
        name: str,
        root_specification: KinematicStructureEntitySpecification,
        *,
        parent_connection_specification: ConnectionSpecification | None = None,
        annotation_kwargs: dict | None = None,
        part_specifications: dict | None = None,
    ) -> SemanticAnnotationWithRootSpecification[Self]:
        """
        Wrap a root entity specification, typically from
        :meth:`get_default_root_kinematic_structure_entity_specification`, into the spawnable annotation
        specification.

        The root geometry is always supplied by the caller. That builder owns every
        geometry parameter (a scale, a handle's ``thickness``, a case's
        ``wall_thickness``), so geometry is described in exactly one place.

        :param name: The name of the annotation and its root entity.
        :param root_specification: The specification of the annotation's root entity.
        :param parent_connection_specification: Connection attaching the root to its
            parent. When omitted, the root is fixed to its parent.
        :param annotation_kwargs: Inert keyword arguments for the annotation
            constructor.
        :param part_specifications: Nested annotation parts keyed by part-whole
            relationship field name.
        :return: The annotation specification.
        """
        if parent_connection_specification is not None:
            root_specification.connection_specification = (
                parent_connection_specification
            )
        return SemanticAnnotationWithRootSpecification(
            name=name,
            semantic_annotation_type=cls,
            root_specification=root_specification,
            annotation_kwargs=(
                annotation_kwargs if annotation_kwargs is not None else {}
            ),
            part_bindings=cls._to_part_bindings(part_specifications),
        )

    @staticmethod
    def _to_part_bindings(
        part_specifications: dict | None,
    ) -> list[PartSpecificationBinding]:
        """
        Convert the keyword-friendly mapping of parts into the bindings the
        specification stores.

        :param part_specifications: Nested annotation parts keyed by part-whole
            relationship field name, each a single specification or a list of them.
        :return: One binding per named field.
        """
        if part_specifications is None:
            return []
        return [
            PartSpecificationBinding(
                field_name=field_name,
                specifications=(list(value) if isinstance(value, list) else [value]),
            )
            for field_name, value in part_specifications.items()
        ]

    def _mount_strategy(
        self,
        main_has_root_body_annotation: HasRootBody,
        relationship: IsPartWholeRelationship,
    ) -> None:
        """
        Realize the relationship between this annotation (as a part) and the
        ``main_has_root_body_annotation`` (the whole) in the kinematic structure. The
        default is to become a kinematic child of the whole, keeping the connection
        the part hangs from; parts with a different strategy (e.g. apertures that cut
        the whole) override this.

        :param main_has_root_body_annotation: The annotation (the whole) this one is
            being added to as a part.
        :param relationship: The metadata of the part-whole relationship field being
            mounted into, describing how the mount affects the whole.
        """
        main_has_root_body_annotation._world.move_branch(
            self.root, main_has_root_body_annotation.root, True
        )

    @property
    def global_transform(self) -> HomogeneousTransformationMatrix:
        return self.root.global_transform

    @property
    def connections(self) -> list[Connection]:
        return self._world.get_connections_of_branch(self.root)

    def _kinematic_structure_entities(
        self, visited: set[int]
    ) -> list[KinematicStructureEntity]:
        if id(self) in visited:
            return []
        visited.add(id(self))
        return self._world.get_kinematic_structure_entities_of_branch(self.root)


@dataclass(eq=False)
class HasRootBody(HasRootKinematicStructureEntity[Body], Mixin):
    """
    Abstract base class for all objects which have a unambiguous root reference frame.

    Each semantic annotation refers to a single Body. Each subclass automatically
    derives a MatchRule from its own class name and the names of its ancestors.
     This makes specialized subclasses naturally more specific than their bases.
    """

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: str,
        world: World,
        world_root_T_self: HomogeneousTransformationMatrix | None = None,
        parent_connection_specification: ConnectionSpecification | None = None,
        scale: Scale | None = None,
    ) -> Self:
        """
        Create a new semantic annotation with a new body in the given world.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and body to.
        :param world_root_T_self: The initial pose of the body in the world root frame.
        :param parent_connection_specification: Connection attaching the body to the
            world root. When omitted, the entity is fixed to the world root.
        :param scale: The scale used to generate the geometry of the body. When omitted,
            the type's default geometry scale applies.
        :return: The created semantic annotation instance.
        """
        return cls.get_annotation_specification(
            name,
            cls.get_default_root_kinematic_structure_entity_specification(scale=scale),
            parent_connection_specification=parent_connection_specification,
        ).spawn(world, parent_T_self=world_root_T_self)

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> BodySpecification:
        """
        Build the default body specification whose geometry matches what
        :meth:`create_with_new_body_in_world` generates from ``scale``.

        This is the geometry-extraction counterpart of the factory: instead of
        mutating a world, it returns a reusable, world-independent specification.

        .. warning:: The result carries no annotation. Spawning it directly puts a bare
            body into the world; pass it to :meth:`get_annotation_specification` to
            obtain the annotation.

        :param name: The name of bodies created from the specification. ``None`` leaves
            naming to the spawning annotation, which overrides it with its own name.
        :param scale: The scale used to generate the box geometry. ``None`` yields a
            geometry-less specification, matching a factory call without a scale.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with a single solid box derived from ``scale``.
        """
        if scale is None:
            return BodySpecification(
                name=name, connection_specification=connection_specification
            )
        return BodySpecification.from_event(
            name,
            scale.to_simple_event().as_composite_set(),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class IsGraspable(HasRootBody, Mixin):
    """
    A mixin class for semantic annotations that state where they may be grasped, by
    their :attr:`grasp_strategy`, and offer grasp candidates.
    """

    grasp_candidate_count: int = field(default=12, kw_only=True)
    """
    How many grasp candidates the annotation offers.
    """

    @classproperty
    def grasp_strategy(cls) -> GraspStrategy:
        """
        :return: Where the annotation may be grasped.
        """
        return GraspFromAnySide()

    def grasp_candidates(
        self,
        model_registry: ModelRegistry | None = None,
        result_pattern: Callable[[], Match[GraspResult]] | None = None,
    ) -> list[GraspCandidate]:
        """
        The grasps this annotation offers, in no particular order: those of its
        :meth:`grasped_part` when it is grasped at one of its parts; otherwise
        :meth:`drawn_grasp_candidates`, or the :meth:`default_grasp_candidates` while no
        model is given.

        :param model_registry: Answers the statement of where the object may be grasped.
        :param result_pattern: Builds the pattern of the result the grasps asked of
            ``model_registry`` are to have had, such as
            :meth:`~semantic_digital_twin.grasping.object_relative_grasp.GraspResult.lifting_pattern`;
            any result when ``None``.
        """
        part = self.grasped_part()
        if part is not self:
            return part.grasp_candidates(model_registry, result_pattern)
        if model_registry is None:
            return self.default_grasp_candidates()
        return self.drawn_grasp_candidates(model_registry, result_pattern)

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: The grasps offered while no model is given: the object at its own
            origin, approached from evenly spaced yaws about its
            z-axis, which needs no knowledge of its shape.
        """
        return [
            GraspCandidate(
                self,
                Pose(
                    orientation=RotationMatrix.from_rpy(yaw=yaw).quaternion,
                    reference_frame=self.root,
                ),
            )
            for yaw in np.linspace(
                0, 2 * np.pi, self.grasp_candidate_count, endpoint=False
            )
        ]

    def drawn_grasp_candidates(
        self,
        model_registry: ModelRegistry | None = None,
        result_pattern: Callable[[], Match[GraspResult]] | None = None,
    ) -> list[GraspCandidate]:
        """
        :param model_registry: Answers the statement of :attr:`grasp_strategy`;
            ``None`` draws uniformly from what it allows.
        :param result_pattern: Builds the pattern of the result the grasps are to have
            had; any result when ``None``.
        :return: Grasps drawn from the statement of :attr:`grasp_strategy` that reach the
            surface of the object, at most :attr:`grasp_candidate_count` of them.
        """
        grasps = draw_grasps(
            self.grasp_strategy.statement(self, result_pattern),
            self.grasp_candidate_count,
            model_registry,
        )
        return [
            grasp.grasp_candidate(graspable=self)
            for grasp in grasps
            if grasp.reaches_surface_of(graspable=self)
        ]

    def grasp_surface(self) -> trimesh.Trimesh:
        """
        :return: The surface that grasps are placed on, in the root body's frame:
            that of every body the annotation references, its parts included, each as
            what it looks like, or else what it collides as.
        :raises NoGraspGeometry: If none of the bodies has either.
        """
        surfaces = []
        for body in self._distinct_bodies():
            shapes = body.visual or body.collision
            if not shapes:
                continue
            surface = shapes.combined_mesh
            if body is not self.root:
                surface.apply_transform(
                    self._world.compute_forward_kinematics_np(self.root, body)
                )
            surfaces.append(surface)
        if not surfaces:
            raise NoGraspGeometry(self)
        return trimesh.util.concatenate(surfaces)

    def grasp_axis(self) -> LineSegment3:
        """
        :return: The axis an elongated object is grasped around, in the root body's
            frame: through the centre of the bounding box of :meth:`grasp_surface`,
            along its longest extent, from its lower to its upper end.
        :raises NoGraspGeometry: If the object has no grasp surface.
        """
        lowest, highest = self.grasp_surface().bounds
        axis_index = int(np.argmax(highest - lowest))
        start, end = (lowest + highest) / 2, (lowest + highest) / 2
        start[axis_index], end[axis_index] = lowest[axis_index], highest[axis_index]
        return LineSegment3(
            start=Point3.from_iterable(start, reference_frame=self.root),
            end=Point3.from_iterable(end, reference_frame=self.root),
        )

    def grasped_part(self) -> IsGraspable:
        """
        :return: The annotation whose surface the object is grasped at; the object
            itself unless one of its parts is grasped instead.
        """
        return self


@dataclass(eq=False)
class HasDrawnGraspsByDefault(IsGraspable, Mixin):
    """
    A mixin class for semantic annotations whose shape is known well enough that, while
    no model is given, their grasps are drawn uniformly from where their
    :attr:`grasp_strategy` allows instead of taken at their origin.
    """

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: Grasps drawn uniformly from where :attr:`grasp_strategy` allows.
        """
        return self.drawn_grasp_candidates()


@dataclass(eq=False)
class HasRootRegion(HasRootKinematicStructureEntity[Region], Mixin):
    """
    A mixin class for semantic annotations that have a region.
    """

    @classmethod
    def create_with_new_region_in_world(
        cls,
        name: str,
        world: World,
        world_root_T_self: HomogeneousTransformationMatrix | None = None,
        parent_connection_specification: ConnectionSpecification | None = None,
        scale: Scale | None = None,
    ) -> Self:
        """
        Create a new semantic annotation with a new region in the given world.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and region to.
        :param world_root_T_self: The initial pose of the region in the world root
            frame.
        :param parent_connection_specification: Connection attaching the region to the
            world root. When omitted, the entity is fixed to the world root.
        :param scale: The scale used to generate the region area geometry.
        :return: The created semantic annotation instance.
        """
        return cls.get_annotation_specification(
            name,
            cls.get_default_root_kinematic_structure_entity_specification(scale=scale),
            parent_connection_specification=parent_connection_specification,
        ).spawn(world, parent_T_self=world_root_T_self)

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> RegionSpecification:
        """
        Build the default region specification whose geometry matches what
        :meth:`create_with_new_region_in_world` generates.

        .. warning:: The result carries no annotation. Spawning it directly puts a bare
            region into the world; pass it to :meth:`get_annotation_specification` to
            obtain the annotation.

        :param name: The name of regions created from the specification. ``None`` leaves
            naming to the spawning annotation, which overrides it with its own name.
        :param scale: The scale used to generate the region area geometry. ``None`` yields a
            geometry-less specification, matching the bare region of the base factory.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A region specification.
        """
        if scale is None:
            return RegionSpecification(
                name=name, connection_specification=connection_specification
            )
        return RegionSpecification.from_event(
            name,
            scale.to_simple_event().as_composite_set(),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class PartWholeRelationship(HasRootKinematicStructureEntity, Mixin):
    """
    Base for annotations that have structural *parts* (the part-whole relation).

    Each part mixin (``HasHandle``, ``HasDoors``, ...) declares a typed part-whole relationship
    field. The unified :meth:`add` routes a part to the field whose element type matches it and lets
    the part mount itself (:meth:`HasRootKinematicStructureEntity._mount_strategy`).
    """

    @synchronized_attribute_modification
    def add(
        self, part: HasRootKinematicStructureEntity, *, field_name: str = ""
    ) -> None:
        """
        Add ``part`` as a structural part, routing it to the matching part-whole
        relationship field by type.

        :param part: The part to add.
        :param field_name: Optional name of the target part-whole relationship field.
            When given, only that field is considered (and ``part`` must still match its
            element type), which resolves the ambiguity when ``type(part)`` matches
            several fields. When empty (default), the field is resolved by type alone.
        :raises UnknownPartWholeRelationshipField: If ``field_name`` is given but is not
            a part-whole relationship field of this annotation.
        :raises CannotBeAPartOf: If no part-whole relationship field of this annotation
            accepts ``type(part)``.
        :raises AmbiguousPart: If ``type(part)`` matches more than one part-whole
            relationship field.
        """
        candidate_fields = WrappedClass(clazz=type(self)).fields_with_metadata(
            metadata_type=IsPartWholeRelationship
        )
        if field_name:
            named_fields = [
                wrapped_part_whole_relationship_field
                for wrapped_part_whole_relationship_field in candidate_fields
                if wrapped_part_whole_relationship_field.field.name == field_name
            ]
            if not named_fields:
                raise UnknownPartWholeRelationshipField(
                    type(self),
                    field_name,
                    [
                        wrapped_part_whole_relationship_field.field.name
                        for wrapped_part_whole_relationship_field in candidate_fields
                    ],
                )
            candidate_fields = named_fields
        matches = [
            wrapped_part_whole_relationship_field
            for wrapped_part_whole_relationship_field in candidate_fields
            if isinstance(part, wrapped_part_whole_relationship_field.type_endpoint)
        ]
        if not matches:
            raise CannotBeAPartOf(self, part)
        if len(matches) > 1:
            raise AmbiguousPart(self, part, [match.field for match in matches])

        [match] = matches
        part._mount_strategy(
            self, IsPartWholeRelationship.of_wrapped_field(wrapped_field=match)
        )
        if match.is_many_to_many_relationship:
            getattr(self, match.field.name).append(part)
        else:
            setattr(self, match.field.name, part)

    @property
    def parts(self) -> list[HasRootKinematicStructureEntity]:
        """
        :return: The structural parts of this annotation, from all of its part-whole
            relationship fields.
        """
        parts = []
        for wrapped_part_whole_relationship_field in WrappedClass(
            clazz=type(self)
        ).fields_with_metadata(metadata_type=IsPartWholeRelationship):
            value = getattr(self, wrapped_part_whole_relationship_field.field.name)
            if wrapped_part_whole_relationship_field.is_many_to_many_relationship:
                parts.extend(value)
            elif value is not None:
                parts.append(value)
        return parts


@dataclass(eq=False)
class HasApertures(HasRootBody, PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that have apertures.

    An aperture is an opening through this annotation's material, so mounting one
    removes its volume from the annotation's collision and visual geometry.
    """

    apertures: list[Aperture] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship(
            removes_part_geometry_from_whole=True
        ).as_dict(),
    )
    """
    The apertures of the semantic annotation.
    """


@dataclass(eq=False)
class HasMovableJoint(HasRootBody, Mixin, ABC):
    """
    A mixin class for semantic annotations whose root moves relative to its parent, such
    as a door on its hinge or a drawer on its rails.
    """

    @property
    def movable_joint(self) -> ActiveConnection1DOF | None:
        """
        The connection the root hangs from, which moves the annotation relative to its
        parent.

        ``None`` while the root cannot move, such as a door fixed to its parent before
        it is mounted on a hinge.
        """
        parent_connection = self.root.parent_connection
        return (
            parent_connection
            if isinstance(parent_connection, ActiveConnection1DOF)
            else None
        )

    @property
    def opening_ratio(self) -> float:
        """
        How far the root stands along the travel of :attr:`movable_joint`: ``0`` at its
        lower limit, where a door or drawer stands closed, and ``1`` at its upper limit.
        """
        if self.movable_joint is None:
            raise MissingMovableJointError(self)
        limits = self.movable_joint.dof.limits
        return (self.movable_joint.position - limits.lower.position) / (
            limits.upper.position - limits.lower.position
        )

    @abstractmethod
    def calculate_self_T_movable_joint(
        self, axis: Vector3
    ) -> HomogeneousTransformationMatrix:
        """
        Calculate where the joint that moves this annotation sits, relative to its root.

        :param axis: The axis the joint moves along or about, in the root frame.
        :return: The pose of the joint frame in the root frame.
        """

    def mount_on_movable_joint(
        self, connection_specification: ActiveConnection1DOFSpecification
    ) -> ActiveConnection1DOF:
        """
        Replace the connection the root hangs from with one built from
        ``connection_specification``, placed where
        :meth:`calculate_self_T_movable_joint` puts the joint, keeping the annotation
        where it is.

        :param connection_specification: The joint to mount on. Its own
            ``connection_T_child`` is ignored, and the specification is left unchanged.
        :return: The connection the root now hangs from.
        """
        placed_specification = dataclasses.replace(
            connection_specification,
            connection_T_child=self.calculate_self_T_movable_joint(
                connection_specification.axis
            ).inverse(),
        )
        return placed_specification.reconnect(self._world, self.root)


@dataclass(eq=False)
class HasDrawers(PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that have drawers.
    """

    drawers: list[Drawer] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The drawers of the semantic annotation.
    """


@dataclass(eq=False)
class HasShelfLayers(PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that have shelf layers.
    """

    shelf_layers: list[ShelfLayer] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The shelf layers of the semantic annotation.
    """


@dataclass(eq=False)
class HasDoors(PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that have doors.
    """

    doors: list[Door] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The doors of the semantic annotation.
    """


@dataclass(eq=False)
class HasHandle(IsGraspable, PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that have a handle, and are grasped at it
    when they have one.
    """

    handle: Handle | None = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The handle of the semantic annotation.
    """

    def grasped_part(self) -> IsGraspable:
        """
        :return: The handle; without a handle, the part the annotation is grasped at
            otherwise.
        """
        if self.handle is None:
            return super().grasped_part()
        return self.handle


@dataclass(eq=False)
class HasRim(HasDrawnGraspsByDefault, PartWholeRelationship, Mixin):
    """
    A mixin class for open containers that have a rim, and are grasped at it when they
    have one.
    """

    rim: Rim | None = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The rim of the container.
    """

    def grasped_part(self) -> IsGraspable:
        """
        :return: The rim; without a rim, the part the annotation is grasped at
            otherwise.
        """
        if self.rim is None:
            return super().grasped_part()
        return self.rim

    @classproperty
    def grasp_strategy(cls) -> GraspStrategy:
        """
        :return: Pinching the top of the container's own wall, as its rim would be.
        """
        return PinchTheTopOfAWall()


@dataclass(eq=False)
class HasHandles(HasRootBody, PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that may have more than one handle.
    """

    handles: list[Handle] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The handles of the semantic annotation.
    """


THasRootBody = TypeVar("THasRootBody", bound=HasRootBody)
"""
A type variable for HasRootBody.
"""


@dataclass(eq=False)
class HasLegs(PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that have legs.
    """

    legs: list[Leg] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The legs of the semantic annotation.
    """


@dataclass(eq=False)
class HasSink(PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that have a sink.
    """

    sink: Sink | None = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The sink of the semantic annotation.
    """


@dataclass(eq=False)
class IsStorageSpace(HasRootBody, Generic[THasRootBody], SubClassSafeGeneric):
    """
    A mixin class for semantic annotations that represent storage spaces.

    Used to afterthefact add object for example to a table, and have those objects move
    with the table when it is moved.
    """

    objects: list[THasRootBody] = field(default_factory=list, hash=False, kw_only=True)
    """
    The occupants currently contained in/on this annotation.
    """

    @synchronized_attribute_modification
    def add_object(self, object: HasRootBody):
        self._world.move_branch(
            object.root, self.root, enable_unsafe_inside_world_block=True
        )
        self.objects.append(object)

    def get_objects_of_type(
        self, object_type: type[SemanticAnnotation]
    ) -> list[HasRootBody]:
        """
        Returns all objects of a given type in the semantic annotation.

        ..warning:: object_type does not have to be a subclass of HasRootBody, as some semantic concepts, for example
        Food may not necessarily inherit from HasRootBody, but some objects stored in here may inherit from Food as well
        as HasRootBody.

        :param object_type: The type of the semantic annotations to return.

        :return: A list of HasRootBody objects of the given type.
        """
        return [obj for obj in self.objects if isinstance(obj, object_type)]


@dataclass(eq=False)
class HasSupportingSurface(IsStorageSpace, Mixin):
    """
    A semantic annotation that represents a supporting surface.
    """

    supporting_surface: Region = field(default=None)
    """
    The supporting surface region of the semantic annotation.
    """

    def calculate_supporting_surface(
        self,
        upward_threshold: float = 0.95,
        clearance_threshold: float = 0.5,
        min_surface_area: float = 0.0225,  # 15cm x 15cm
    ) -> Region | None:
        """
        Calculate the supporting surface region for the semantic annotation, add it to
        the world, and set it as the supporting surface of self.

        :param upward_threshold: The threshold for the face normal to be considered
            upward-facing.
        :param clearance_threshold: The threshold for the vertical clearance above the
            surface.
        :param min_surface_area: The minimum area for a surface to be considered a
            supporting surface.
        :return: The supporting surface region, or None if no suitable region could be
            found.
        """
        mesh = self.root.combined_mesh
        if mesh is None:
            return None
        # --- Find upward-facing faces ---
        normals = mesh.face_normals
        upward_mask = normals[:, 2] > upward_threshold

        if not upward_mask.any():
            return None

        # --- Find connected upward-facing regions ---
        upward_face_indices = np.nonzero(upward_mask)[0]
        submesh_up = mesh.submesh([upward_face_indices], append=True)
        face_groups = submesh_up.split(only_watertight=False)

        # Compute total area for each group
        large_groups = [g for g in face_groups if g.area >= min_surface_area]

        if not large_groups:
            return None

        # --- Merge qualifying upward-facing submeshes ---
        candidates = trimesh.util.concatenate(large_groups)

        # --- Check vertical clearance using ray casting ---
        face_centers = candidates.triangles_center
        ray_origins = face_centers + np.array([0, 0, 0.01])  # small upward offset
        ray_dirs = np.tile([0, 0, 1], (len(ray_origins), 1))

        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs
        )

        # Compute distances to intersections (if any)
        distances = np.full(len(ray_origins), np.inf)
        distances[index_ray] = np.linalg.norm(
            locations - ray_origins[index_ray], axis=1
        )

        # Filter faces with enough space above
        clear_mask = (distances > clearance_threshold) | np.isinf(distances)

        if not clear_mask.any():
            return None

        candidates_filtered = candidates.submesh([clear_mask], append=True)

        # --- Build the region ---
        # The region is placed where the surface was found, relative to the root's
        # origin, so that it lies on top of the root wherever that origin is
        vertices = candidates_filtered.vertices
        self_P_supporting_surface = vertices.mean(axis=0)
        points_3d = [
            Point3(
                x,
                y,
                z,
                reference_frame=self.root,
            )
            for x, y, z in vertices - self_P_supporting_surface
        ]
        supporting_surface = Region.from_3d_points(
            name=PrefixedName(
                f"{self.root.name.name}_supporting_surface_region",
                self.root.name.prefix,
            ),
            points_3d=points_3d,
        )

        x, y, z = self_P_supporting_surface
        self_C_supporting_surface = FixedConnection(
            parent=self.root,
            child=supporting_surface,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x, y=y, z=z, reference_frame=self.root
            ),
        )
        self._world.add_region(supporting_surface)
        self._world.add_connection(self_C_supporting_surface)
        self.add_supporting_surface(supporting_surface)
        return supporting_surface

    def infer_objects_on_surface(self):
        """
        Infer and add objects that are supported by this surface to the storage space.

        This method queries the world for bodies that are supported by this annotation's
        root body, finds their corresponding semantic annotations, and adds them to the
        objects list if they are not already present.
        """
        bodies = variable_from(self._world.bodies_with_collision)
        body = entity(bodies).where(
            SupportedBy(
                supported=bodies,
                supporting=self.root,
            )
        )
        objects = an(
            entity(
                semantic_annotation := variable(
                    HasRootBody, domain=self._world.semantic_annotations
                )
            ).where(semantic_annotation.root == body)
        ).evaluate()
        for obj in objects:
            if obj in self.objects:
                continue
            self.add_object(obj)

    @synchronized_attribute_modification
    def add_supporting_surface(self, region: Region):
        self._world.move_branch(
            region, self.root, enable_unsafe_inside_world_block=True
        )
        self.supporting_surface = region

    def sample_points_from_surface(
        self,
        body_to_sample_for: HasRootBody | None = None,
        category_of_interest: type[SemanticAnnotation] | None = None,
        amount: int = 100,
    ) -> list[Point3]:
        """
        Samples points from a surface around the semantic annotation.

        The surface is determined by the supporting surface of the semantic annotation and is truncated by the objects on the surface. The points are sampled
        using a Gaussian mixture model.

        ..warning:: Calling this method when the self.supporting_surface is None will cause the method to calculate the
            surface and add it to the world, resulting in model updates being published if the synchronizer is running.

        :param body_to_sample_for: The physical object to sample points for.
        :param category_of_interest: The type of object sample points around.
        :param amount: The number of points to sample.

        :return: A list of sampled points, sorted by distance to the around_object.
        """
        if self.supporting_surface is None:
            with self._world.modify_world():
                supporting_surface = self.calculate_supporting_surface()
            if supporting_surface is None:
                return []

        largest_xy_object_dimension = 0.1
        z_object_dimension = 0.0
        if body_to_sample_for:
            largest_xy_object_dimension = body_to_sample_for.root.combined_mesh.extents[
                :2
            ].max()
            z_object_dimension = body_to_sample_for.root.combined_mesh.extents[2]

        self_max_z = self.supporting_surface.area.max_point.z
        z_coordinate = np.full(
            (amount, 1),
            self_max_z + (z_object_dimension / 2),
        )

        surface_circuit = self._build_surface_sampler(
            category_of_interest=category_of_interest,
            object_bloat=largest_xy_object_dimension,
        )

        if surface_circuit is None:
            return []

        samples = surface_circuit.sample(amount)
        samples = samples[
            np.argsort(surface_circuit.log_likelihood(events=samples))[::-1]
        ]
        samples = np.concatenate((samples, z_coordinate), axis=1)

        if category_of_interest:
            return [
                Point3(*s[1:], reference_frame=self.supporting_surface) for s in samples
            ]
        return [Point3(*s, reference_frame=self.supporting_surface) for s in samples]

    def _build_surface_sampler(
        self,
        category_of_interest: type[SemanticAnnotation] | None = None,
        object_bloat: float = 0.1,
    ):
        """
        Build a probabilistic circuit representing the supporting surface, truncated by
        the objects on the surface, and with Gaussian mixtures around the objects of
        interest.

        :param category_of_interest: The type of object sample points around.
        :param object_bloat: The amount of bloat to apply to the object event.
        """
        truncated_event_2d = self._2d_surface_sample_space_excluding_objects(
            object_bloat
        )

        objects_of_interest = (
            self.get_objects_of_type(category_of_interest)
            if category_of_interest
            else []
        )
        if objects_of_interest:
            return self._2d_gaussian_sampler_from_2d_sample_space(
                objects_of_interest=objects_of_interest,
                # using values too low makes sampling from truncated gaussians very unstable
                variance=1,
                sample_space=truncated_event_2d,
            )
        else:
            return uniform_measure_of_event(truncated_event_2d)

    def _2d_surface_sample_space_excluding_objects(self, object_bloat: float) -> Event:
        """
        Compute a 2D event representing the supporting surface, truncated by the objects
        on the surface.

        :param object_bloat: The amount of bloat to apply to the object events.
        """
        area_of_self = BoundingBoxCollection.from_shapes(self.supporting_surface.area)
        area_of_self.transform_all_shapes_to_own_frame()
        event = area_of_self.event

        event_2d = event.marginal(SpatialVariables.xy)
        for obj in self.objects:
            bounding_box = obj.root.collision.as_bounding_box_collection_in_frame(
                self.supporting_surface
            ).bounding_box()
            bounding_box.enlarge_all(object_bloat)
            object_event = bounding_box.simple_event.as_composite_set()
            object_event_2d = object_event.marginal(SpatialVariables.xy)
            event_2d = event_2d - object_event_2d
        return event_2d

    def _2d_gaussian_sampler_from_2d_sample_space(
        self,
        objects_of_interest: list[HasRootBody],
        variance: float,
        sample_space: Event,
    ) -> ProbabilisticCircuit | None:
        """
        Create a Gaussian mixture model from a list of points, truncated by an event.

        :param objects_of_interest: Objects of interest to sample around. The Gaussian
            mixtures will be centered around the positions of these objects on the
            surface.
        :param variance: The standard deviation to use for the Gaussian mixtures.
        :param sample_space: The event to truncate the Gaussian mixture model with.
        :return: A probabilistic circuit representing the Gaussian mixture model
            truncated by the event, or None if the event has zero measure.
        """
        surface_circuit = self._untruncated_2d_gaussian_sampler(
            objects_of_interest=objects_of_interest,
            variance=variance,
        )
        sample_space.fill_missing_variables(surface_circuit.variables)
        surface_circuit.log_truncated_in_place(sample_space)

        return surface_circuit

    def _untruncated_2d_gaussian_sampler(
        self,
        objects_of_interest: list[HasRootBody],
        variance: float,
    ) -> ProbabilisticCircuit:
        """
        Create a Gaussian mixture model from a list of points, without truncation.

        This method is extracted from the `_2d_gaussian_sampler_from_2d_sample_space`
        method so that the generated distribution can be tested properly, which cannot
        be done after truncation.
        """
        surface_circuit = ProbabilisticCircuit()
        surface_circuit_root = SumUnit(probabilistic_circuit=surface_circuit)

        objects_of_interest_variable = Symbolic(
            name="objects_of_interest",
            domain=EventSet.from_iterable(objects_of_interest),
        )

        for object_of_interest in objects_of_interest:
            surface_P_obj = self._world.transform(
                object_of_interest.root.global_transform, self.supporting_surface
            )

            p_object_root = ProductUnit(probabilistic_circuit=surface_circuit)
            surface_circuit_root.add_subcircuit(p_object_root, 1.0)

            object_of_interest_p = make_dirac(
                objects_of_interest_variable, object_of_interest
            )

            x_p = GaussianDistribution(
                variable=SpatialVariables.x.value,
                location=float(surface_P_obj.x),
                scale=variance,
            )
            y_p = GaussianDistribution(
                variable=SpatialVariables.y.value,
                location=float(surface_P_obj.y),
                scale=variance,
            )

            p_object_root.add_subcircuit(leaf(object_of_interest_p, surface_circuit))
            p_object_root.add_subcircuit(leaf(x_p, surface_circuit))
            p_object_root.add_subcircuit(leaf(y_p, surface_circuit))

        return surface_circuit

    def spawn_bounding_boxes_as_region(
        self,
        boxes: BoundingBoxCollection[VolumetricBoundingBox, Point3],
        name: PrefixedName | None = None,
        color: Color | None = None,
    ) -> Region:
        """
        Spawn a collection of bounding boxes as a region, connected to this
        annotation's root with a fixed connection.

        :param boxes: The bounding boxes to spawn, e.g. the free space of a graph of
            convex sets.
        :param name: The name of the region. Defaults to "region".
        :param color: The color of the region. Defaults to a translucent green.
        :return: The region.
        """
        if name is None:
            name = PrefixedName("region")
        if color is None:
            color = Color.from_rgba(red=0.5, green=1.0, blue=0.5, opacity=0.5)

        shapes = boxes.as_shapes()
        shapes.dye_shapes(color)
        region = Region.from_shape_collection(name, shapes)

        with self._world.modify_world():
            self._world.add_region(region)
            self._world.add_connection(FixedConnection(parent=self.root, child=region))
        return region

    def planar_free_space(
        self,
        max_height: float = 2.0,
        tolerance: float = 0.001,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
        semantic_wall_annotation: Wall | None = None,
        obstacle_height_clearance: float = 0.01,
    ) -> PlanarGraphOfBoundingBoxes:
        """
        Build a graph of the free space above this supporting surface, from the
        surface's own top up to ``max_height``.

        The search space is derived from :attr:`supporting_surface`'s own area -- its
        x,y extent bounds the navigable region, and the height range determines which
        obstacles in the world count as blocking.

        ..warning:: Calling this method when :attr:`supporting_surface` is None will
            cause the method to calculate the surface and add it to the world, resulting
            in model updates being published if the synchronizer is running.

        :param max_height: The height of the free space above the surface.
        :param tolerance: The tolerance for the intersection when calculating the
            connectivity.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat wall obstacles.
        :param semantic_wall_annotation: An optional wall annotation to be considered
            as an obstacle.
        :param obstacle_height_clearance: The amount every obstacle bounding box gets
            expanded by in z, regardless of ``bloat_obstacles``/``bloat_walls``. The
            search space starts comfortably above that many times over above the
            surface's own top, so the surface's own body never registers as an
            obstacle to the free space built over it.
        :raises NoSupportingSurfaceError: If no surface is attached and none can be
            derived from this annotation's geometry.
        :return: The graph of the free space above this surface.
        """
        from semantic_digital_twin.semantic_annotations.semantic_annotations import (
            SemanticEnvironmentAnnotation,
        )
        from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
            PlanarGraphOfBoundingBoxes,
        )

        world = self._world
        if self.supporting_surface is None:
            with world.modify_world():
                if self.calculate_supporting_surface() is None:
                    raise NoSupportingSurfaceError(self)

        # The surface's own frame lies on its top, so the free space and the paths
        # through it are expressed relative to what stands on it.
        origin = HomogeneousTransformationMatrix(
            reference_frame=self.supporting_surface
        )
        surface_box = self.supporting_surface.area.as_bounding_box_collection_at_origin(
            origin
        ).bounding_box()

        surface_top = surface_box.max_z + 2 * obstacle_height_clearance
        search_space = BoundingBoxCollection(
            [
                VolumetricBoundingBox(
                    surface_box.min_x,
                    surface_box.min_y,
                    surface_top,
                    surface_box.max_x,
                    surface_box.max_y,
                    surface_top + max_height,
                    origin,
                )
            ],
            self.supporting_surface,
        )

        semantic_obstacle_annotation = SemanticEnvironmentAnnotation(
            root=world.root, _world=world
        )

        return PlanarGraphOfBoundingBoxes.free_space_from_semantic_annotation(
            search_space=search_space,
            semantic_obstacle_annotation=semantic_obstacle_annotation,
            semantic_wall_annotation=semantic_wall_annotation,
            tolerance=tolerance,
            bloat_obstacles=bloat_obstacles,
            bloat_walls=bloat_walls,
            obstacle_height_clearance=obstacle_height_clearance,
        )


@dataclass(eq=False)
class HasCaseAsRootBody(HasSupportingSurface, Mixin):
    """
    A mixin class for semantic annotations that have a case as root body.
    """

    @classproperty
    @abstractmethod
    def _hole_direction_axis(cls) -> Vector3:
        """
        The unit vector along the direction of the physical hole of the geometry, without
        a reference frame.

        Used to build this type's default geometry before any instance/root body exists to
        serve as a reference frame. Use :attr:`hole_direction` instead once an instance
        exists.
        """

    @property
    def hole_direction(self) -> Vector3:
        """
        The direction of the physical hole of the geometry.

        For a drawer for example, this would always be Z.
                ..warning:: This does not describe the axis along, for example, a drawer opens. Its the physical opening where
                you can put something into the drawer.
        """
        return Vector3.from_iterable(
            self._hole_direction_axis.to_np(), reference_frame=self.root
        )

    @classmethod
    def _create_container_event(cls, scale: Scale, wall_thickness: float) -> Event:
        """
        Return an event representing a container with walls of a specified thickness.

        :param scale: The scale of the container.
        :param wall_thickness: The thickness of the walls.
        :return: The event representing the container.
        """
        outer_box = scale.to_simple_event()
        inner_box = Scale(
            scale.x - wall_thickness,
            scale.y - wall_thickness,
            scale.z - wall_thickness,
        ).to_simple_event(cls._hole_direction_axis, wall_thickness)

        container_event = outer_box.as_composite_set() - inner_box.as_composite_set()

        return container_event

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
        *,
        wall_thickness: float = 0.01,
    ) -> BodySpecification:
        """
        Build the default body specification for a container/case, matching the hollow
        geometry generated by :meth:`create_with_new_body_in_world`.

        :param name: The name of bodies created from the specification.
        :param scale: The outer scale of the case. Defaults to a unit :class:`Scale`
            when omitted.
        :param wall_thickness: The thickness of the case walls.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with hollow container geometry.
        """
        scale = scale if scale is not None else Scale()
        return BodySpecification.from_event(
            name,
            cls._create_container_event(scale, wall_thickness),
            connection_specification=connection_specification,
        )


# %% furniture built from other furniture


@dataclass(eq=False)
class HasCabinets(PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations built from cabinets.
    """

    cabinets: list[Cabinet] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The cabinets the semantic annotation is built from.
    """


@dataclass(eq=False)
class HasCounterTop(PartWholeRelationship, Mixin):
    """
    A mixin class for semantic annotations that carry a counter top.

    Distinct from :class:`HasSupportingSurface`, which is the bare region something can
    be put on: this is the worktop as a thing in its own right, which is what a scan
    sees and what in turn carries the sink.
    """

    counter_top: CounterTop = field(
        default=None,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The counter top of the semantic annotation.
    """


# %% liquid source


class LiquidSource(ABC):
    """A source of liquid that can pour into a container.

    Decouples :meth:`HasFillLevel.receive_outflow_from` from any
    particular kind of source: a tilting cup and a static faucet differ
    only in how they answer these questions.
    """

    @abstractmethod
    def outflow_volume_rate(self, world: World) -> sm.Scalar:
        """Symbolic volume rate of liquid leaving the source, before gating.

        :param world: The world providing the forward kinematics.
        :return: Symbolic outflow volume rate, positive while pouring.
        """

    @abstractmethod
    def liquid_exit_point(self, world: World) -> Point3:
        """World-frame point at which liquid departs the source.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit point in the world frame.
        """

    @abstractmethod
    def liquid_exit_direction(self, world: World) -> Vector3:
        """World-frame direction the departing liquid initially travels in.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit direction in the world frame.
        """

    def liquid_exit_velocity(self, world: World, exit_speed: sm.ScalarData) -> Vector3:
        """World-frame velocity the departing liquid initially travels with.

        Liquid leaving over a rim slides off without a vertical launch,
        so by default the speed acts along the horizontal part of the
        exit direction only.

        :param world: The world providing the forward kinematics.
        :param exit_speed: Speed of the liquid leaving the source, in
            m/s.
        :return: Symbolic exit velocity in the world frame.
        """
        exit_direction = self.liquid_exit_direction(world)
        return Vector3(
            x=exit_speed * exit_direction.x,
            y=exit_speed * exit_direction.y,
            z=sm.Scalar(0.0),
            reference_frame=world.root,
        )

    @property
    @abstractmethod
    def pour_tilt_expression(self) -> sm.Scalar:
        """Symbolic tilt angle of the source while pouring; zero for a non-
        tilting source."""

    def current_outflow_velocity(self, world: World) -> sm.Scalar | None:
        """Live horizontal speed of the liquid leaving the source, in m/s, or
        ``None`` if unavailable.

        A source whose pour dynamics expose a head can report the
        current exit speed; sources without such a model return ``None``
        so callers fall back to a nominal speed.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit speed, or ``None``.
        """
        return None

    @abstractmethod
    def couple_drain_to_gate(self, gate: sm.Scalar, world: World) -> None:
        """Make the source drain only while ``gate`` is open, conserving the
        transferred volume.

        :param gate: The shared transfer gate in ``[0, 1]``.
        :param world: The world the source lives in.
        """

    @abstractmethod
    def validate_can_pour(self) -> None:
        """Raise if the source cannot currently pour, e.g. uninitialized or
        already coupled."""


# %% fill-level container


@dataclass(eq=False)
class HasFillLevel(HasRootBody, LiquidSource):
    """Mixin that adds a virtual fill-level DOF to any semantic annotation.

    The fill level is represented as a virtual :class:`LiquidConnection`
    whose position encodes fill in the range ``[0, 1]``. Call
    :meth:`initialize_fill_level` explicitly after the collision
    geometry has been set.

    A filled container is itself a :class:`LiquidSource`: it can pour
    its contents into another container by tilting.
    """

    RIM_EXIT_TILT_EPSILON: ClassVar[float] = 1e-2
    """Softening term for normalizing the pour direction, in units of
    horizontal tilt magnitude.

    Blended into the normalization as ``sqrt(tilt**2 + epsilon**2)`` so
    the exit point moves smoothly from the rim centre (upright) to the
    rim edge (tilted) without a kink or a division by zero at the near-
    vertical configuration, which keeps the predictive controller
    stable.
    """

    fill_connection: LiquidConnection | None = field(default=None)
    """The virtual connection whose position encodes fill level in [0, 1]."""

    fill_equation: PouringEquation | None = field(default=None)
    """Differential equation governing how this container drains when
    tilted."""

    inflow_equation: InflowEquation | None = field(
        default=None, metadata=JSONMetadata(serialize=False).as_dict()
    )
    """Differential equation governing how this container fills from an
    external source.

    Bound to the world it was built in, so it never leaves the process;
    other worlds rebuild it from :attr:`inflow_coupling`.
    """

    inflow_coupling: LiquidTransferCoupling | None = field(default=None)
    """Serializable description of the transfer coupling.

    Unlike :attr:`inflow_equation`, whose symbolic expressions cannot
    cross a process boundary, this survives synchronization and lets
    :meth:`ensure_inflow_coupling` rebuild the symbolic coupling in
    another world.
    """

    measured_capacity: float | None = field(default=None, kw_only=True)
    """The volume this container holds at full fill, in cubic metres, where it
    was measured, or ``None`` to read it off the collision geometry."""

    @synchronized_attribute_modification
    def set_inflow_coupling(self, coupling: LiquidTransferCoupling | None) -> None:
        """Record the serializable inflow coupling descriptor and synchronize
        it to other worlds.

        :param coupling: The coupling descriptor, or ``None`` to clear
            it.
        """
        self.inflow_coupling = coupling

    @synchronized_attribute_modification
    def set_fill_connection(self, connection: LiquidConnection | None) -> None:
        """Set the fill-level connection.

        :param connection: The LiquidConnection to track as the fill-
            level DOF.
        """
        self.fill_connection = connection

    @synchronized_attribute_modification
    def add_fill_equation(self, fill_equation: PouringEquation | None) -> None:
        """Add a fill equation to the semantic annotation.

        Unlike :attr:`inflow_equation`, a :class:`PouringEquation`
        carries no live symbolic state —
        :meth:`~PouringEquation.symbolic_velocity` builds its expression
        from plain parameters on demand — so it is fully JSON-
        serializable and can be recorded in the JSON-diffed modification
        history.

        :param fill_equation: The fill equation to add.
        """
        self.fill_equation = fill_equation
        if self.fill_connection is not None:
            self.fill_connection.outflow_equation = fill_equation

    def add_inflow_equation(self, inflow_equation: InflowEquation | None) -> None:
        """Add an inflow equation to the semantic annotation.

        Unlike :meth:`set_inflow_coupling`, this is not synchronized:
        the equation is a symbolic expression that cannot cross a
        process boundary, so it cannot be recorded in the JSON-diffed
        modification history.

        :param inflow_equation: The inflow equation to add.
        """
        self.inflow_equation = inflow_equation
        if self.fill_connection is not None:
            self.fill_connection.inflow_equation = inflow_equation

    def initialize_fill_level(
        self,
        world: World,
        initial_fill: float = 1.0,
        outflow_rate_constant: float = 1.0,
        discharge_coefficient: float = DEFAULT_DISCHARGE_COEFFICIENT,
        repose_angle: float = 0.0,
    ) -> None:
        """Create the virtual fill-level DOF, attach it to the world, and wire
        up the pouring equation.

        Must be called after the annotation's collision geometry is set.

        :param world: The world to add the fill-level DOF to.
        :param initial_fill: Starting fill level in [0, 1].
        :param outflow_rate_constant: Outflow rate constant for the
            articulated pouring equation.
        :param discharge_coefficient: Scales the Torricelli exit speed
            to a realistic pour range.
        :param repose_angle: Tilt past the geometric onset before the
            contents flow, in radians.
        :raises FillLevelAlreadyInitializedError: if this container's
            fill level is already initialized.
        """
        if self.fill_connection is not None:
            raise FillLevelAlreadyInitializedError(container=self)
        fill_equation = self.create_pouring_equation(
            world, outflow_rate_constant, discharge_coefficient
        )
        fill_equation.repose_angle = repose_angle
        phantom = Body(name=PrefixedName(f"{self.root.name.name}_fill_level_phantom"))
        with world.modify_world():
            world.add_body(phantom)
            connection = LiquidConnection.create_with_dofs(
                world=world,
                parent=self.root,
                child=phantom,
                axis=Vector3.Z(),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=0.0, velocity=-1.0),
                    upper=DerivativeMap(position=1.0, velocity=1.0),
                ),
            )
            connection.outflow_equation = fill_equation
            world.add_connection(connection)
            self.set_fill_connection(connection)
            self.add_fill_equation(fill_equation)

        with world.modify_world():
            JointState.from_mapping({connection: initial_fill}).apply_to(world)

    def receive_outflow_from(
        self,
        source: LiquidSource,
        world: World,
        exit_speed: float = DEFAULT_POUR_EXIT_SPEED,
        height_gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
        overlap_gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
    ) -> None:
        """Couple this container's inflow to the outflow of a liquid source.

        The source's outflow volume rate is gated by geometry so liquid only enters this container
        while the liquid's projectile lands in its opening.  The same gate is handed back to the
        source so it only drains while it is actually pouring into this container: no liquid is
        spilled and the transfer is volume conserving.

        The symbolic gate and inflow cannot be serialized, so a parametric
        :class:`~semantic_digital_twin.world_description.connections.LiquidTransferCoupling` is
        recorded on the fill connection whenever the source is a world entity. That descriptor
        survives synchronization to another world, where :meth:`ensure_inflow_coupling` rebuilds
        the symbolic coupling locally.

        :param source: The liquid source whose outflow pours into this one.
        :param world: The world providing the forward kinematics for the geometric gate.
        :param exit_speed: Horizontal speed of the liquid leaving the source, in m/s.
        :param height_gate_sharpness: Logistic steepness of the source-above-receiver gate.
        :param overlap_gate_sharpness: Logistic steepness of the projectile-landing gate.

        ..warning:: This mutates the source via :meth:`LiquidSource.couple_drain_to_gate`.

        :raises MissingFillEquationError: if the source was never initialized with a fill level.
        :raises SourceAlreadyCoupledError: if the source's outflow is already gated onto a receiver.
        :raises ReceiverNotInitializedError: if this receiver was never initialized with a fill level.
        :raises ReceiverAlreadyCoupledError: if this receiver's inflow is already coupled to a source.
        """
        source.validate_can_pour()
        if self.fill_connection is None:
            raise ReceiverNotInitializedError(receiver=self)
        if (
            self.inflow_coupling is not None
            or self.inflow_equation is not None
            or self.fill_connection.inflow_equation is not None
        ):
            raise ReceiverAlreadyCoupledError(receiver=self)
        if isinstance(source, WorldEntityWithID):
            coupling = LiquidTransferCoupling(
                source_id=source.id,
                exit_speed=exit_speed,
                height_gate_sharpness=height_gate_sharpness,
                overlap_gate_sharpness=overlap_gate_sharpness,
            )
            with world.modify_world():
                self.set_inflow_coupling(coupling)
        self._establish_inflow_coupling(
            source, world, exit_speed, height_gate_sharpness, overlap_gate_sharpness
        )

    def recouple_outflow_from(
        self, source: HasFillLevel, world: World, fill_equation: PouringEquation
    ) -> None:
        """Replace the source's drain with ``fill_equation`` and (re-)establish
        the coupling from it.

        Unlike :meth:`receive_outflow_from`, this accepts a source and a receiver that are already
        coupled: the previous gated drain and inflow are discarded while the coupling's exit speed
        and gate sharpnesses are kept, so a client can switch the source's head model (for example
        analytic to learned) against a live Giskard process. The equation swap is published, which
        marks the coupling stale in every synchronized world, and :meth:`ensure_inflow_coupling`
        rebuilds it there from the new equation.

        :param source: The liquid source whose drain is replaced.
        :param world: The world providing the forward kinematics for the geometric gate.
        :param fill_equation: The new, ungated drain equation for the source.
        """
        previous_coupling = self.inflow_coupling
        with world.modify_world():
            source.add_fill_equation(fill_equation)
        self._clear_inflow_coupling(world)
        if previous_coupling is None:
            self.receive_outflow_from(source=source, world=world)
            return
        self.receive_outflow_from(
            source=source,
            world=world,
            exit_speed=previous_coupling.exit_speed,
            height_gate_sharpness=previous_coupling.height_gate_sharpness,
            overlap_gate_sharpness=previous_coupling.overlap_gate_sharpness,
        )

    def _clear_inflow_coupling(self, world: World) -> None:
        """Detach the coupling descriptor and the local symbolic inflow so this
        receiver can be coupled anew.

        :param world: The world the coupling was built against.
        """
        with world.modify_world(publish_changes=False):
            self.add_inflow_equation(None)
            self.set_inflow_coupling(None)

    def _establish_inflow_coupling(
        self,
        source: LiquidSource,
        world: World,
        exit_speed: float,
        height_gate_sharpness: float,
        overlap_gate_sharpness: float,
    ) -> None:
        """Build the symbolic inflow equation and the source's gated outflow
        against ``world``.

        The symbolic expressions are bound to ``world`` and are local to it, so the changes are not
        published: they are reconstructed independently in every world that holds the coupling.

        :param source: The liquid source whose outflow pours into this one.
        :param world: The world providing the forward kinematics for the geometric gate.
        :param exit_speed: Horizontal speed of the liquid leaving the source, in m/s.
        :param height_gate_sharpness: Logistic steepness of the source-above-receiver gate.
        :param overlap_gate_sharpness: Logistic steepness of the projectile-landing gate.
        """
        source_volume_rate = source.outflow_volume_rate(world)
        effective_exit_speed = source.current_outflow_velocity(world)
        if effective_exit_speed is None:
            effective_exit_speed = exit_speed
        landing_point = self.projectile_landing_point(
            source, world, effective_exit_speed
        )
        gate = self._geometric_transfer_gate(
            source, world, landing_point, height_gate_sharpness, overlap_gate_sharpness
        )
        inflow_equation = GatedInflowEquation(
            container_height=self.root.collision.height,
            container_width=self.root.collision.width,
            capacity=self.capacity,
            inflow=source_volume_rate,
            gate=gate.product,
            height_gate=gate.height,
            overlap_gate=gate.overlap,
            source_tilt_expression=source.pour_tilt_expression,
            exit_speed=exit_speed,
        )
        with world.modify_world(publish_changes=False):
            self.add_inflow_equation(inflow_equation)
            source.couple_drain_to_gate(gate.product, world)
        self._record_coupling_provenance(source)

    def _record_coupling_provenance(self, source: LiquidSource) -> None:
        """Remember which source drain the just-built symbolic coupling was
        derived from.

        Recorded after the drain was gated, so an unchanged coupling
        compares equal to the source's current fill equation and only a
        synchronized equation swap reads as stale.

        :param source: The liquid source the coupling was built from.
        """
        if not isinstance(source, HasFillLevel):
            return
        self.fill_connection.coupled_source_equation_json = to_json(
            source.fill_equation
        )

    def _inflow_coupling_is_stale(self, source: LiquidSource) -> bool:
        """Whether the local symbolic coupling was built from a different
        source drain than the source currently carries, e.g. after a client
        switched head models and the swap was synchronized into this world.

        :param source: The coupling's liquid source, resolved in this
            world.
        :return:``True`` if the coupling must be rebuilt from the
            source's current equation.
        """
        if not isinstance(source, HasFillLevel):
            return False
        recorded = self.fill_connection.coupled_source_equation_json
        if recorded is None:
            # No provenance means the inflow equation was not built in this world (e.g. it
            # arrived via synchronization), so it cannot be trusted and must be rebuilt.
            return True
        return recorded != to_json(source.fill_equation)

    def ensure_inflow_coupling(self, world: World) -> None:
        """Rebuild the symbolic inflow coupling from the stored descriptor if
        it is missing or stale.

        A world synchronized from another process carries the coupling
        descriptor but not the symbolic inflow equation, which cannot be
        serialized. This reconstructs the symbolic coupling against
        ``world`` so a transfer task can read it. It is a no-op when the
        symbolic inflow equation is already present and still derived
        from the source's current fill equation, or when no coupling
        descriptor was recorded.

        :param world: The world the coupling must be rebuilt against.
        """
        coupling = self.inflow_coupling
        if coupling is None:
            return
        source = world.get_semantic_annotation_by_id(coupling.source_id)
        self._reattach_fill_connection(world)
        if isinstance(source, HasFillLevel):
            source._reattach_fill_connection(world)
        if (
            self.fill_connection.inflow_equation is not None
            and not self._inflow_coupling_is_stale(source)
        ):
            return
        self._establish_inflow_coupling(
            source,
            world,
            coupling.exit_speed,
            coupling.height_gate_sharpness,
            coupling.overlap_gate_sharpness,
        )

    def _reattach_fill_connection(self, world: World) -> None:
        """Point :attr:`fill_connection` at the connection resident in
        ``world``.

        Synchronizing an annotation to another world deserializes its
        fill connection by value, so the reference is a detached copy
        with no world. Re-resolving it against ``world`` restores a
        connection whose forward kinematics and physics equations can be
        evaluated.

        :param world: The world whose resident fill connection this
            annotation must track.
        """
        if self.fill_connection._world is world:
            return
        self.fill_connection = world.get_connection(
            self.fill_connection.parent, self.fill_connection.child
        )

    def outflow_volume_rate(self, world: World) -> sm.Scalar:
        """Volume rate leaving this cup as it tilts, converting its normalised
        drain to a volume rate.

        :param world: The world providing the forward kinematics.
        :return: Symbolic outflow volume rate, positive while pouring.
        """
        normalised_drain = self.fill_equation.symbolic_velocity(self.fill_connection)
        return -normalised_drain * self.fill_equation.capacity

    def create_pouring_equation(
        self, world: World, outflow_rate_constant: float, discharge_coefficient: float
    ) -> ArticulatedPouringEquation:
        """The pouring equation modelling this container's drain.

        The liquid stands across the collision geometry's width and
        pours over its top corner once it reaches the collision
        geometry's height.

        :param world: The world the container lives in.
        :param outflow_rate_constant: Outflow rate constant of the
            drain.
        :param discharge_coefficient: Scales the Torricelli exit speed.
        :return: The ungated drain of this container.
        """
        return ArticulatedPouringEquation(
            container_width=self.root.collision.width,
            container_height=self.root.collision.height,
            capacity=self.capacity,
            outflow_rate_constant=outflow_rate_constant,
            discharge_coefficient=discharge_coefficient,
        )

    def current_outflow_velocity(self, world: World) -> sm.Scalar | None:
        """Discharge-scaled Torricelli exit speed from the current pour, or
        ``None`` without a model.

        Delegates to the pour equation, which leaves the lip at ``C_d *
        sqrt(2 g h_head)``: a fuller or more tilted cup pours faster,
        and the discharge coefficient tunes the pour range.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit speed, or ``None`` if the fill equation
            exposes no pour head.
        """
        if not isinstance(self.fill_equation, ArticulatedPouringEquation):
            return None
        return self.fill_equation.exit_velocity(self.fill_connection)

    def liquid_exit_point(self, world: World) -> Point3:
        """The point on the rim over which liquid pours, in the world frame.

        While the cup is upright the exit point coincides with the rim
        centre; as it tilts the exit point moves to the rim edge on the
        pour side, which is where the liquid actually spills.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit point in the world frame.
        """
        world_T_cup = world.compose_forward_kinematics_expression(world.root, self.root)
        world_P_rim_exit = world_T_cup @ self._rim_exit_point(world)
        world_P_rim_exit.reference_frame = world.root
        return world_P_rim_exit

    def _rim_exit_point(self, world: World) -> Point3:
        """The rim exit point in the cup frame: the rim edge along the
        horizontal pour direction.

        The pour spills over the lowest rim edge, which lies in the
        direction opposite the horizontal component of the world-up axis
        as seen from the cup.  The centre-to-edge offset follows the
        rim's half-extents, so a wider cup exits farther from its
        centre.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit point in the cup frame.
        """
        collision = self.root.collision
        lower = collision.min_point
        upper = collision.max_point
        cup_R_world = world.compose_forward_kinematics_expression(
            self.root, world.root
        ).rotation_matrix
        cup_V_up = cup_R_world @ Vector3.Z()
        tilt_magnitude_squared = cup_V_up.x * cup_V_up.x + cup_V_up.y * cup_V_up.y
        normalization = sm.sqrt(tilt_magnitude_squared + self.RIM_EXIT_TILT_EPSILON**2)
        half_extent_x = (upper.x - lower.x) / 2
        half_extent_y = (upper.y - lower.y) / 2
        cup_P_rim_exit = Point3(
            x=(lower.x + upper.x) / 2 - cup_V_up.x / normalization * half_extent_x,
            y=(lower.y + upper.y) / 2 - cup_V_up.y / normalization * half_extent_y,
            z=upper.z,
            reference_frame=self.root,
        )
        return cup_P_rim_exit

    def liquid_exit_direction(self, world: World) -> Vector3:
        """The direction the tilted opening faces in the world frame.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit direction in the world frame.
        """
        world_R_cup = world.compose_forward_kinematics_expression(
            world.root, self.root
        ).rotation_matrix
        world_V_pour = world_R_cup @ Vector3.Z()
        return world_V_pour

    @property
    def pour_tilt_expression(self) -> sm.Scalar:
        """Symbolic tilt angle of this cup, taken from its fill connection."""
        return self.fill_connection.tilt_expression

    def couple_drain_to_gate(self, gate: sm.Scalar, world: World) -> None:
        """Gate this cup's own outflow so it drains only while ``gate`` is
        open.

        Delegates to the fill equation so its drain model is preserved
        when the coupling is rebuilt, for example after synchronization
        to another process.

        :param gate: The shared transfer gate in ``[0, 1]``.
        :param world: The world the cup lives in.
        """
        self.add_fill_equation(self.fill_equation.with_gate(gate))

    def validate_can_pour(self) -> None:
        """
        :raises MissingFillEquationError: if this cup was never initialized with a fill level.
        :raises SourceAlreadyCoupledError: if this cup's outflow is already gated onto a receiver.
        """
        if self.fill_equation is None or self.fill_connection is None:
            raise MissingFillEquationError(source=self)
        if isinstance(self.fill_equation, GatedArticulatedPouringEquation):
            raise SourceAlreadyCoupledError(source=self)

    def projectile_landing_point(
        self,
        source: LiquidSource,
        world: World,
        exit_speed: sm.ScalarData,
        gravity: float = STANDARD_GRAVITY,
    ) -> Point3:
        """Where liquid poured from the source lands on this container's
        opening plane.

        The liquid leaves the source's exit point with the source's exit
        velocity and then follows projectile motion under gravity; the
        returned point is where that arc crosses this container's
        opening plane. A stream launched upwards therefore flies longer
        and lands farther out than one released horizontally.

        :param source: The pouring liquid source.
        :param world: The world providing the forward kinematics.
        :param exit_speed: Speed of the liquid leaving the source, in
            m/s.
        :param gravity: Gravitational acceleration in metres per second
            squared.
        :return: The symbolic landing point in the world frame, on this
            container's opening plane.
        """
        exit_point = source.liquid_exit_point(world)
        exit_velocity = source.liquid_exit_velocity(world, exit_speed)
        plane_height = self.opening_point(world).z
        drop_height = sm.max(
            sm.Scalar(MINIMUM_DROP_HEIGHT), exit_point.z - plane_height
        )
        flight_time = (
            exit_velocity.z + sm.sqrt(exit_velocity.z**2 + 2 * gravity * drop_height)
        ) / gravity
        return Point3(
            x=exit_point.x + exit_velocity.x * flight_time,
            y=exit_point.y + exit_velocity.y * flight_time,
            z=plane_height,
            reference_frame=world.root,
        )

    def _geometric_transfer_gate(
        self,
        source: LiquidSource,
        world: World,
        landing_point: Point3,
        height_gate_sharpness: float,
        overlap_gate_sharpness: float,
    ) -> TransferGate:
        """Build the differentiable gate that is open only while the source
        pours into this container.

        The gate multiplies a vertical term (source exit point above
        this container's opening) and a horizontal term (the liquid's
        projectile lands within this container's opening radius), each a
        smooth logistic so the optimizer sees a non-zero gradient when
        aiming the pour. Because the landing point moves forward as the
        source tilts, the optimizer must position the source upstream
        and tilt it so the arc lands in this container.

        Both terms are measured against the opening rather than the
        container's origin: liquid enters over the rim, so a lip below
        the rim cannot fill the container however far above its base it
        sits.

        :param source: The pouring liquid source.
        :param world: The world providing the forward kinematics.
        :param landing_point: The projectile landing point on this
            container's opening plane.
        :param height_gate_sharpness: Logistic steepness of the vertical
            term.
        :param overlap_gate_sharpness: Logistic steepness of the
            landing-in-opening term.
        :return: The gate's vertical and horizontal factors.
        """
        source_exit = source.liquid_exit_point(world)
        receiver_opening = self.opening_point(world)

        height_gate = self._logistic(
            source_exit.z - receiver_opening.z, height_gate_sharpness
        )
        landing_distance = sm.sqrt(
            (landing_point.x - receiver_opening.x) ** 2
            + (landing_point.y - receiver_opening.y) ** 2
        )
        overlap_gate = self._logistic(
            (self.opening_radius - landing_distance) / self.opening_radius,
            overlap_gate_sharpness,
        )
        return TransferGate(height=height_gate, overlap=overlap_gate)

    @property
    def opening_radius(self) -> float:
        """Radius of this container's opening, in metres."""
        return self.root.collision.width / 2

    @property
    def capacity(self) -> float:
        """Volume this container holds at full fill, in cubic metres.

        The measured capacity where there is one; otherwise read from
        the collision geometry as the box its extents bound. The shape
        factor this leaves out is the same for containers of the same
        shape, so it cancels in a transfer between them.
        """
        if self.measured_capacity is not None:
            return self.measured_capacity
        collision = self.root.collision
        return collision.depth * collision.width * collision.height

    def opening_point(self, world: World) -> Point3:
        """The centre of this container's opening, in the world frame.

        The container's own origin sits at the base of its collision
        geometry, so the opening is a container height above it;
        anything that reasons about where liquid enters or leaves must
        use this rather than the origin.

        :param world: The world providing the forward kinematics.
        :return: Symbolic opening centre in the world frame.
        """
        world_T_cup = world.compose_forward_kinematics_expression(world.root, self.root)
        cup_P_rim = self.rim_point()
        world_P_opening = world_T_cup @ cup_P_rim
        world_P_opening.reference_frame = world.root
        return world_P_opening

    def rim_point(self) -> Point3:
        """The centre of this container's opening, expressed in the container's
        own frame.

        :return: A point at the horizontal centre and top of the
            collision bounding box.
        """
        lower = self.root.collision.min_point
        upper = self.root.collision.max_point
        return Point3(
            x=(lower.x + upper.x) / 2,
            y=(lower.y + upper.y) / 2,
            z=upper.z,
            reference_frame=self.root,
        )

    @staticmethod
    def _logistic(value: sm.Scalar, sharpness: float) -> sm.Scalar:
        """Smooth ``[0, 1]`` step that approaches ``1`` as ``value`` grows
        positive.

        :param value: The symbolic margin being gated.
        :param sharpness: Steepness of the transition around ``value ==
            0``.
        :return: Symbolic logistic of ``sharpness * value``.
        """
        return sm.Scalar(1.0) / (sm.Scalar(1.0) + sm.exp(-sharpness * value))

    @property
    def fill_level(self) -> float:
        """Current fill level in ``[0, 1]``."""
        return float(self.fill_connection.position)
