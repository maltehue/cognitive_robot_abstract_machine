from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from itertools import combinations

from lxml import etree
from typing_extensions import List, Protocol, TYPE_CHECKING, runtime_checkable, Self

from .collision_matrix import (
    CollisionRule,
    CollisionMatrix,
    CollisionCheck,
)
from ..robots.abstract_robot import AbstractRobot

if TYPE_CHECKING:
    from ..world import World
    from ..world_description.world_entity import Body


@dataclass
class AvoidCollisionRule(CollisionRule, ABC):
    buffer_zone_distance: float = field(default=0.05)
    """
    Distance defining a buffer zone around the entity. The buffer zone represents a soft boundary where
    proximity should be monitored but minor violations are acceptable.
    """

    violated_distance: float = field(default=0.0)
    """
    Critical distance threshold that must not be violated. Any proximity below this threshold represents
    a severe collision risk requiring immediate attention.
    """

    added_collision_checks: set[CollisionCheck] = field(default_factory=set)

    def applies_to(self, body_a: Body, body_b: Body) -> bool:
        """
        Returns True if the rule configures collision distances for the given body.
        """
        raise NotImplementedError

    def buffer_zone_distance_for(self, body_a: Body, body_b: Body) -> float | None:
        """
        Returns the configured buffer-zone distance for the body or None if not applicable.
        """
        return self.buffer_zone_distance if self.applies_to(body_a, body_b) else None

    def violated_distance_for(self, body_a: Body, body_b: Body) -> float | None:
        """
        Returns the configured violated distance for the body or None if not applicable.
        """
        return self.violated_distance if self.applies_to(body_a, body_b) else None

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_matrix.add_collision_checks(self.added_collision_checks)


@dataclass
class AllowCollisionRule(CollisionRule, ABC):
    allowed_collision_pairs: set[CollisionCheck] = field(default_factory=set)
    allowed_collision_bodies: set[Body] = field(default_factory=set)

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_matrix.remove_collision_checks(self.allowed_collision_pairs)
        collision_matrix.collision_checks = {
            collision_check
            for collision_check in collision_matrix.collision_checks
            if collision_check.body_a not in self.allowed_collision_bodies
            and collision_check.body_b not in self.allowed_collision_bodies
        }

    def update(self, world: World):
        self.allowed_collision_pairs = set()
        self.allowed_collision_bodies = set()
        self._update(world)

    @abstractmethod
    def _update(self, world: World): ...


@dataclass
class AvoidCollisionBetweenGroups(AvoidCollisionRule):
    """
    Adds collision checks between all pairs of bodies in the given groups to the collision matrix.
    """

    body_group_a: List[Body] = field(default_factory=list)
    body_group_b: List[Body] = field(default_factory=list)

    def applies_to(self, body_a: Body, body_b: Body) -> bool:
        """
        Returns True if the body is a member of any group handled by this rule.
        """
        return (body_a in self.body_group_a and body_b in self.body_group_b) or (
            body_a in self.body_group_b and body_b in self.body_group_a
        )

    def update(self, world: World):
        self.added_collision_checks = set()
        for body_a in self.body_group_a:
            for body_b in self.body_group_b:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
                )
                self.added_collision_checks.add(collision_check)


@dataclass
class AvoidAllCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all body pairs of the world managed by the rule to the collision matrix.
    """

    world: World = field(kw_only=True)

    def applies_to(self, body_a: Body, body_b: Body) -> bool:
        return (
            body_a in self.world.bodies_with_collision
            or body_b in self.world.bodies_with_collision
        )

    def update(self, world: World):
        self.added_collision_checks = set()
        for body_a, body_b in combinations(self.world.bodies_with_collision, 2):
            collision_check = CollisionCheck.create_and_validate(
                body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
            )
            self.added_collision_checks.add(collision_check)


@dataclass
class AvoidExternalCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all bodies managed by the rule and all other bodies in the world
    that are not managed by the rule.
    """

    bodies: List[Body] = field(default_factory=list)
    world: World = field(kw_only=True)

    def applies_to(self, body_a: Body, body_b: Body) -> bool:
        return (
            body_a in self.bodies and body_b in self.world.bodies_with_collision
        ) or (body_b in self.bodies and body_a in self.world.bodies_with_collision)

    def update(self, world: World):
        self.added_collision_checks = set()
        for body_a in self.bodies:
            for body_b in self.world.bodies_with_collision:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
                )
                self.added_collision_checks.add(collision_check)


@dataclass
class AvoidSelfCollisions(AvoidCollisionRule):
    """
    Adds collision checks between all body pairs of the robot managed by the rule.
    """

    robot: AbstractRobot = field(kw_only=True)

    def applies_to(self, body_a: Body, body_b: Body) -> bool:
        return (
            body_a in self.robot.bodies_with_collision
            and body_b in self.robot.bodies_with_collision
        )

    def update(self, world: World):
        self.added_collision_checks = set()
        for body_a, body_b in combinations(self.robot.bodies_with_collision, 2):
            collision_check = CollisionCheck.create_and_validate(
                body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
            )
            self.added_collision_checks.add(collision_check)


@dataclass
class AllowAllCollisions(AllowCollisionRule):
    world: World = field(kw_only=True)

    def _update(self, world: World):
        self.allowed_collision_bodies = set(world.bodies_with_collision)


@dataclass
class AllowCollisionBetweenGroups(AllowCollisionRule):
    body_group_a: List[Body] = field(default_factory=list)
    body_group_b: List[Body] = field(default_factory=list)

    def update(self, world: World):
        self.allowed_collision_pairs = set()
        for body_a in self.body_group_a:
            for body_b in self.body_group_b:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b
                )
                self.allowed_collision_pairs.add(collision_check)


@dataclass
class AllowNonRobotCollisions(AllowCollisionRule):

    def _update(self, world: World):
        """
        Disable collision checks between bodies that do not belong to any robot.
        """
        # Bodies that are part of any robot and participate in collisions
        robot_bodies: set[Body] = {
            body
            for robot in world.get_semantic_annotations_by_type(AbstractRobot)
            for body in robot.bodies_with_collision
        }

        # Bodies with collisions that are NOT part of a robot
        non_robot_bodies: set[Body] = set(world.bodies_with_collision) - robot_bodies
        if not non_robot_bodies:
            return

        # Disable every unordered pair (including self-collisions) exactly once
        for a, b in combinations(non_robot_bodies, 2):
            self.allowed_collision_pairs.add(
                CollisionCheck.create_and_validate(body_a=a, body_b=b, distance=0)
            )


@dataclass
class AllowSelfCollisions(AllowCollisionRule):
    robot: AbstractRobot = field(kw_only=True)

    def _update(self, world: World):
        self.allowed_collision_bodies = set(self.robot.bodies_with_collision)


@dataclass
class AllowCollisionForAdjacentPairs(AllowCollisionRule):

    def _update(self, world: World):
        for body_a, body_b in combinations(world.bodies_with_collision, 2):
            if (
                not world.is_controlled_connection_in_chain(body_a, body_b)
                or body_a == body_b.parent_kinematic_structure_entity
                or body_b == body_a.parent_kinematic_structure_entity
            ):
                self.allowed_collision_pairs.add(
                    CollisionCheck.create_and_validate(body_a, body_b)
                )


@dataclass
class SelfCollisionMatrixRule(AllowCollisionRule):

    def _update(self, world: World): ...

    def update(self, world: World): ...

    def compute_collision_matrix(self, world: World) -> set[CollisionCheck]:
        """
        Parses the collision requrests and (temporary) collision configs in the world
        to create a set of collision checks.
        """
        collision_matrix: set[CollisionCheck] = set()
        for collision_request in self.collision_requests:
            if collision_request.all_bodies_for_group1():
                view_1_bodies = world.bodies_with_collision
            else:
                view_1_bodies = collision_request.body_group_a
            if collision_request.all_bodies_for_group2():
                view2_bodies = world.bodies_with_collision
            else:
                view2_bodies = collision_request.body_group_b
            for body1 in view_1_bodies:
                for body2 in view2_bodies:
                    collision_check = CollisionCheck.create_and_validate(
                        body_a=body1, body_b=body2, distance=0
                    )
                    distance = collision_request.distance
                    if not collision_request.is_allow_collision():
                        collision_check.distance = distance
                        collision_check._validate()
                    if collision_request.is_allow_collision():
                        if collision_check in collision_matrix:
                            collision_matrix.remove(collision_check)
                    if collision_request.is_avoid_collision():
                        if collision_request.is_distance_set():
                            collision_matrix.add(collision_check)
                        else:
                            collision_matrix.add(collision_check)
        return collision_matrix

    @classmethod
    def from_collision_srdf(cls, file_path: str, world: World) -> Self:
        """
        Creates a CollisionConfig instance from an SRDF file.

        Parse an SRDF file to configure disabled collision pairs or bodies for a given world.
        Process SRDF elements like `disable_collisions`, `disable_self_collision`,
        or `disable_all_collisions` to update collision configuration
        by referencing bodies in the provided `world`.

        :param file_path: The path to the SRDF file used for collision configuration.
        """
        self = cls()
        SRDF_DISABLE_ALL_COLLISIONS: str = "disable_all_collisions"
        SRDF_DISABLE_SELF_COLLISION: str = "disable_self_collision"
        SRDF_MOVEIT_DISABLE_COLLISIONS: str = "disable_collisions"

        srdf = etree.parse(file_path)
        srdf_root = srdf.getroot()

        children_with_tag = [child for child in srdf_root if hasattr(child, "tag")]

        child_disable_collisions = [
            c for c in children_with_tag if c.tag == SRDF_DISABLE_ALL_COLLISIONS
        ]

        for c in child_disable_collisions:
            body = world.get_body_by_name(c.attrib["link"])
            self.allowed_collision_bodies.add(body)

        child_disable_moveit_and_self_collision = [
            c
            for c in children_with_tag
            if c.tag in {SRDF_MOVEIT_DISABLE_COLLISIONS, SRDF_DISABLE_SELF_COLLISION}
        ]

        disabled_collision_pairs = [
            (body_a, body_b)
            for child in child_disable_moveit_and_self_collision
            if (body_a := world.get_body_by_name(child.attrib["link1"])).has_collision()
            and (
                body_b := world.get_body_by_name(child.attrib["link2"])
            ).has_collision()
        ]

        for body_a, body_b in disabled_collision_pairs:
            if body_a == body_b:
                continue
            self.allowed_collision_pairs.add(
                CollisionCheck.create_and_validate(body_a, body_b)
            )
        return self
