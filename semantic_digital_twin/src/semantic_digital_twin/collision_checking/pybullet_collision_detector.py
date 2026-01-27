from dataclasses import dataclass, field
from typing import Dict, Tuple, DefaultDict, List, Set, Optional

import giskardpy_bullet_bindings as bpb

from giskardpy.middleware import get_middleware
from semantic_digital_twin.collision_checking.bpb_wrapper import (
    create_shape_from_link,
    create_collision,
)
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionDetector,
    CollisionCheck,
)
from semantic_digital_twin.collision_checking.collisions import GiskardCollision
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class BulletCollisionDetector(CollisionDetector):
    kineverse_world: bpb.KineverseWorld = field(
        default_factory=bpb.KineverseWorld, init=False
    )
    body_to_bpb_obj: Dict[Body, bpb.CollisionObject] = field(
        default_factory=dict, init=False
    )
    query: Optional[
        DefaultDict[PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]
    ] = field(default=None, init=False)

    def sync_world_model(self) -> None:
        self.reset_cache()
        get_middleware().logdebug("hard sync")
        for o in self.kineverse_world.collision_objects:
            self.kineverse_world.remove_collision_object(o)
        self.body_to_bpb_obj = {}
        self.objects_in_order = []

        for body in sorted(
            self.world.bodies_with_enabled_collision, key=lambda b: b.id
        ):
            self.add_body(body)
            self.objects_in_order.append(self.body_to_bpb_obj[body])

    def sync_world_state(self) -> None:
        bpb.batch_set_transforms(
            self.objects_in_order,
            self.world.compute_forward_kinematics_of_all_collision_bodies(),
        )

    def add_body(self, body: Body):
        if not body.has_collision() or body.get_collision_config().disabled:
            return
        o = create_shape_from_link(link=body, tmp_folder=self.tmp_folder)
        self.kineverse_world.add_collision_object(o)
        self.body_to_bpb_obj[body] = o

    def reset_cache(self):
        self.query = None

    def cut_off_distances_to_query(
        self, collision_matrix: Set[CollisionCheck], buffer: float = 0.05
    ) -> DefaultDict[PrefixedName, Set[Tuple[bpb.CollisionObject, float]]]:
        if self.query is None:
            self.query = {
                (
                    self.body_to_bpb_obj[check.body_a],
                    self.body_to_bpb_obj[check.body_b],
                ): check.distance
                + buffer
                for check in collision_matrix
            }
        return self.query

    def check_collisions(
        self,
        collision_matrix: Optional[Set[CollisionCheck]] = None,
        buffer: float = 0.05,
    ) -> List[GiskardCollision]:

        query = self.cut_off_distances_to_query(collision_matrix, buffer=buffer)
        result: List[bpb.Collision] = (
            self.kineverse_world.get_closest_filtered_map_batch(query)
        )
        return [create_collision(collision, self.world) for collision in result]
