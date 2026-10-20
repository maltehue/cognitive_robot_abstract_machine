from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AllowCollisionForEndEffector,
)

if TYPE_CHECKING:
    from semantic_digital_twin.collision_checking.collision_rules import (
        AllowCollisionRule,
    )
    from semantic_digital_twin.robots.robot_parts import EndEffector
    from semantic_digital_twin.world_description.world_entity import Body

# %% explicit motion contact rules


@dataclass
class HasContactRules:
    """
    Attach explicit collision allowances to a manipulation motion.
    """

    allowed_contacts: list[AllowCollisionRule] = field(
        default_factory=list, kw_only=True
    )
    """
    Pair-specific allowances active during the motion.
    """

    def create_contact_nodes(
        self, gripper: EndEffector, allow_environment_contacts: bool
    ) -> list[UpdateTemporaryCollisionRules]:
        """
        Build one temporary policy including an optional broad hand allowance.
        """
        rules = list(self.allowed_contacts)
        if allow_environment_contacts:
            rules.append(AllowCollisionForEndEffector(end_effector=gripper))
        if not rules:
            return []
        return [UpdateTemporaryCollisionRules(temporary_rules=rules)]


# %% manipulation object and support contacts


@dataclass
class HasManipulationContacts:
    """
    Select the object and support contacts allowed during manipulation.
    """

    allow_environment_contacts: bool = field(default=True, kw_only=True)
    """
    Whether to preserve the broad hand allowance instead of selected pairs.
    """

    support_bodies: list[Body] = field(default_factory=list, kw_only=True)
    """
    Supports that the manipulated object may touch while lifting or placing.
    """

    def create_contact_rules(
        self, gripper: EndEffector, object_body: Body, *, include_supports: bool = True
    ) -> list[AllowCollisionRule]:
        """
        Build selected pairs when broad hand contact is disabled.
        """
        if self.allow_environment_contacts:
            return []
        rules = [
            AllowCollisionBetweenGroups(
                body_group_a=gripper.bodies_with_collision, body_group_b=[object_body]
            )
        ]
        if include_supports and self.support_bodies:
            rules.append(
                AllowCollisionBetweenGroups(
                    body_group_a=[object_body], body_group_b=self.support_bodies
                )
            )
        return rules
