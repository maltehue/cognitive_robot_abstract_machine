from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import trimesh.sample
from typing_extensions import Self

from krrood.entity_query_language.factories import (
    entity,
    variable,
    contains,
    an,
    the,
)
from krrood.entity_query_language.predicate import (
    RenderedFields,
    symbolic_function,
    Triple,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    clause,
    Copula,
    Noun,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.collision_checking.collision_detector import (
    ClosestPoints,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
    AllowSelfCollisions,
)
from semantic_digital_twin.reasoning.predicates import PlaceIsOccupied
from semantic_digital_twin.robots.robot_part_mixins import HasFingers
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Camera,
    EndEffector,
    TCamera,
)
from semantic_digital_twin.semantic_annotations.mixins import TKinematicStructureEntity
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@symbolic_function
def robot_in_collision(
    robot: AbstractRobot,
    ignore_collision_with: Optional[List[Body]] = None,
    threshold: float = 0.001,
) -> List[ClosestPoints]:
    """
    Check if the robot collides with any object in the world at the given pose.

    :param robot: The robot object
    :param ignore_collision_with: A list of objects to ignore collision with
    :param threshold: The threshold for contact detection
    :return: True if the robot collides with any object, False otherwise
    """
    if ignore_collision_with is None:
        ignore_collision_with = []

    world = robot._world

    with world.modify_world():
        world.collision_manager.clear_temporary_rules()
        world.collision_manager.add_temporary_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=threshold,
                robot=robot,
            )
        )
        world.collision_manager.add_temporary_rule(AllowSelfCollisions(robot=robot))
        world.collision_manager.add_temporary_rule(
            AllowCollisionBetweenGroups(
                body_group_a=robot.bodies, body_group_b=ignore_collision_with
            )
        )
    world.collision_manager.update_collision_matrix()

    collisions = world.collision_manager.compute_collisions()

    return collisions.contacts


@symbolic_function
def robot_holds_body(robot: AbstractRobot, body: Body) -> bool:
    """
    Check if a robot is holding an object.

    :param robot: The robot object
    :param body: The body to check if it is picked
    :return: True if the robot is holding the object, False otherwise
    """
    g = variable(EndEffector, robot._world.semantic_annotations)
    grippers = an(
        entity(g).where(
            g._robot == robot,
        )
    )

    return any(
        [is_body_in_gripper(body, gripper) > 0.0 for gripper in grippers.evaluate()]
    )


@symbolic_function
def blocking(
    pose: HomogeneousTransformationMatrix,
    root: Body,
    tip: Body,
) -> List[ClosestPoints]:
    """
    Get the bodies that are blocking the robot from reaching a given position.

    The blocking are all bodies that are in collision with the robot when reaching for
    the pose.

    :param pose: The pose to reach
    :param root: The root of the kinematic chain.
    :param tip: The threshold between the end effector and the position.
    :return: A list of bodies the robot is in collision with when reaching for the
        specified object or None if the pose or object is not reachable.
    """
    result = root._world.compute_inverse_kinematics(
        root=root, tip=tip, target=pose, max_iterations=1000
    )
    with root._world.modify_world():
        for dof, state in result.items():
            root._world.state[dof.id].position = state

    r = variable(AbstractRobot, root._world.semantic_annotations)
    robot = the(
        entity(r).where(
            contains(r.bodies, tip),
        )
    )
    return robot_in_collision(robot.first(), [])


# %% sampled grasp contacts


@dataclass
class GripperRaySamples:
    """
    Distinct sampled rays hitting each body between a gripper's fingers.
    """

    sample_size: int
    """
    The total number of sampled rays, including misses.
    """

    hit_rays: dict[Body, set[int]] = field(default_factory=dict)
    """
    The distinct indices of rays intersecting each non-finger body.
    """

    @classmethod
    def sample(cls, gripper: HasFingers, sample_size: int) -> Self:
        """
        Sample each thumb-opposed finger pair and retain distinct hit indices.
        """
        thumb_mesh = gripper.thumb.tip.collision.combined_mesh.copy()
        thumb_mesh.apply_transform(gripper.thumb.tip.global_transform.to_np())
        opposing_fingers = [
            finger for finger in gripper.fingers if finger is not gripper.thumb
        ]
        finger_bodies = {body for finger in gripper.fingers for body in finger.bodies}
        samples = cls(sample_size=sample_size * len(opposing_fingers))
        ray_tracer = gripper._world.ray_tracer
        ray_tracer.update_scene()
        for finger_index, finger in enumerate(opposing_fingers):
            finger_mesh = finger.tip.collision.combined_mesh.copy()
            finger_mesh.apply_transform(finger.tip.global_transform.to_np())
            finger_points = trimesh.sample.sample_surface(finger_mesh, sample_size)[0]
            thumb_points = trimesh.sample.sample_surface(thumb_mesh, sample_size)[0]
            _, ray_indices, bodies = ray_tracer.ray_test(finger_points, thumb_points)
            for ray_index, body in zip(ray_indices, bodies):
                if body not in finger_bodies:
                    samples.hit_rays.setdefault(body, set()).add(
                        finger_index * sample_size + int(ray_index)
                    )
        return samples

    def compute_fraction(self, body: Body) -> float:
        """
        Return the fraction of sampled rays intersecting the requested body.
        """
        return len(self.hit_rays.get(body, set())) / self.sample_size


@symbolic_function
def bodies_in_gripper(gripper: HasFingers, sample_size: int = 100) -> list[Body]:
    """
    Return the distinct non-finger bodies intersecting rays between the fingers.

    :param gripper: The gripper to inspect.
    :param sample_size: The number of rays sampled per thumb-opposed finger pair.
    """
    return list(GripperRaySamples.sample(gripper, sample_size).hit_rays)


@symbolic_function
def is_body_in_gripper(
    body: Body, gripper: EndEffector, sample_size: int = 100
) -> float:
    """
    Check if the body in the gripper.

    This method samples random rays between the finger and the thumb and returns the
    marginal probability that the rays intersect.

    :param body: The body for which the check should be done.
    :param gripper: The gripper for which the check should be done.
    :param sample_size: The number of rays sampled per thumb-opposed finger pair.
    :return: The percentage of rays between the fingers that hit the body.
    """
    return GripperRaySamples.sample(gripper, sample_size).compute_fraction(body)


@symbolic_function
def is_body_gripped(
    body: Body, gripper: EndEffector, threshold: float = 0.9, sample_size: int = 100
) -> bool:
    """
    Check if the body is held by the gripper with at least the given confidence.

    :param body: The body for which the check should be done.
    :param gripper: The gripper for which the check should be done.
    :param threshold: Minimum fraction of sampled rays that must hit ``body`` (see
        :func:`is_body_in_gripper`) for it to count as held.
    :param sample_size: The number of rays sampled per thumb-opposed finger pair.
    :return: Whether ``body`` is held by ``gripper``.
    """
    return is_body_in_gripper(body, gripper, sample_size) > threshold


@symbolic_function
def is_gripper_holding_something(gripper: EndEffector) -> bool:
    """
    Check if the gripper is holding something.

    :param gripper: The gripper for which the check should be done.
    :return: True if there is a body mounted beneath the gripper in the kinematic chain.
    """
    return bool(gripper.tool_frame.child_kinematic_structure_entities)


@symbolic_function
def is_pose_free_for_robot(robot: AbstractRobot, pose: Pose) -> bool:
    return not PlaceIsOccupied(
        robot.mobile_base.bounding_box,
        pose,
        robot._world,
        robot.bodies_with_collision
        + [
            kse
            for annotation in robot._world.get_semantic_annotations_by_type(Floor)
            for kse in annotation.kinematic_structure_entities
        ],
    )()


@symbolic_function
def get_visible_bodies(camera: Camera) -> List[KinematicStructureEntity]:
    """
    Get all bodies and regions that are visible from the given camera using a
    segmentation mask.

    :param camera: The camera for which the visible objects should be returned
    :return: A list of bodies/regions that are visible from the camera
    """
    rt = RayTracer(camera._world)
    rt.update_scene()

    seg = rt.create_segmentation_mask(
        camera.root_T_forward_view,
        resolution=CameraResolution(width=256, height=256),
        min_distance=0.2,
        field_of_view=camera.field_of_view,
    )
    indices = np.unique(seg)
    indices = indices[indices > -1]
    bodies = [camera._world.kinematic_structure[i] for i in indices]

    return bodies


@dataclass(eq=False)
class VisibleTo(Triple[TKinematicStructureEntity, TCamera]):
    """
    Whether a camera can see something.
    """

    entity: TKinematicStructureEntity
    """
    The thing that may be in view.
    """

    camera: TCamera
    """
    The camera looking.
    """

    @property
    def subject(self) -> TKinematicStructureEntity:
        return self.entity

    @property
    def object(self) -> TCamera:
        return self.camera

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the entity is visible to the camera"*.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(
            Noun(fields["entity"]),
            Copula(),
            Adjective("visible"),
            Prepositions.TO,
            Noun(fields["camera"]),
        )

    def __call__(self) -> bool:
        return self.entity in get_visible_bodies(self.camera)


@symbolic_function
def occluding_bodies(camera: Camera, body: Body) -> List[Body]:
    """
    Determines the bodies that occlude a given body in the scene as seen from a
    specified camera.

    This function uses a ray-tracing approach to check occlusion. Every body that hides
    anything from the target body is an occluding body.

    :param camera: The camera for which the occluding bodies should be returned
    :param body: The body for which the occluding bodies should be returned
    :return: A list of bodies that are occluding the given body.
    """
    camera_pose = camera.root_T_forward_view

    # create a world only containing the target body
    world_without_occlusion = deepcopy(body._world)
    root = Body(name=PrefixedName("root"))
    copied_body = Body.from_json(body.to_json())
    root_T_body = body.global_transform
    root_T_body.reference_frame = root
    root_to_copied_body = FixedConnection(
        parent=root,
        child=copied_body,
        parent_T_connection_expression=root_T_body,
    )
    with world_without_occlusion.modify_world():
        world_without_occlusion.clear()
        world_without_occlusion.add_body(root)
        world_without_occlusion.add_connection(root_to_copied_body)

    # get segmentation mask without occlusion
    ray_tracer_without_occlusion = RayTracer(world_without_occlusion)
    ray_tracer_without_occlusion.update_scene()
    segmentation_mask_without_occlusion = (
        ray_tracer_without_occlusion.create_segmentation_mask(
            camera_pose,
            resolution=CameraResolution(width=256, height=256),
            min_distance=0.1,
            field_of_view=camera.field_of_view,
        )
    )

    # get segmentation mask with occlusion
    ray_tracer_with_occlusion = RayTracer(camera._world)
    ray_tracer_with_occlusion.update_scene()
    segmentation_mask_with_occlusion = (
        ray_tracer_with_occlusion.create_segmentation_mask(
            camera_pose,
            resolution=CameraResolution(width=256, height=256),
            min_distance=0.1,
            field_of_view=camera.field_of_view,
        )
    )

    # pixels where the target body is visible when nothing else is in the scene
    target_pixels = segmentation_mask_without_occlusion == copied_body.index

    # whatever covers those pixels in the real scene (except the target itself)
    # is occluding the target
    indices = np.unique(segmentation_mask_with_occlusion[target_pixels])
    indices = indices[(indices > -1) & (indices != body.index)]
    bodies = [camera._world.kinematic_structure[i] for i in indices]
    return bodies
