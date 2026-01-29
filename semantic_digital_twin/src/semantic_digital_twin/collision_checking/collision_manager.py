from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations_with_replacement

from typing_extensions import Optional, List, Dict, Any, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from .collision_detector import CollisionMatrix, CollisionCheck, CollisionRule
from .collision_rules import Updatable
from ..adapters.world_entity_kwargs_tracker import WorldEntityWithIDKwargsTracker
from ..callbacks.callback import ModelChangeCallback
from ..world import World
from ..world_description.world_entity import Body


@dataclass
class CollisionManager(ModelChangeCallback):
    """
    static disables:
        Disable non-robot collisions.
        Disable adjacent body collisions, including ones with no hardware interface.
        self collision matrix
    """

    self_collision_matrix: CollisionRule
    default_collision_rules: List[CollisionRule] = field(default_factory=list)
    temporary_collision_rules: List[CollisionRule] = field(default_factory=list)
    updatable_rules: List[Updatable] = field(default_factory=list)

    def _notify(self):
        for rule in self.updatable_rules:
            rule.update(self.world)

    def create_collision_matrix(self) -> CollisionMatrix:
        collision_matrix = CollisionMatrix()
        for rule in self.default_collision_rules:
            rule.apply_to_collision_matrix(collision_matrix)
        for rule in self.temporary_collision_rules:
            rule.apply_to_collision_matrix(collision_matrix)
        self.self_collision_matrix.apply_to_collision_matrix(collision_matrix)
        return collision_matrix

    def add_disabled_collision_pair(self, body_a: Body, body_b: Body):
        """
        Disable collision checking between two bodies
        """
        pair = tuple(sorted([body_a, body_b], key=lambda body: body.id))
        self._disabled_collision_pairs.add(pair)

    def disable_collisions_for_adjacent_bodies(self):
        """
        Computes pairs of bodies that should not be collision checked because they have no controlled connections
        between them.

        When all connections between two bodies are not controlled, these bodies cannot move relative to each
        other, so collision checking between them is unnecessary.

        :return: Set of body pairs that should have collisions disabled
        """

        body_combinations = combinations_with_replacement(
            self.world.bodies_with_collision, 2
        )
        for body_a, body_b in body_combinations:
            if not self.world.is_controlled_connection_in_chain(body_a, body_b):
                pass
