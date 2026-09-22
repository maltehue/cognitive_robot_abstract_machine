"""
Collision failures identify the same indexed contact as their controller task.
"""

import pytest

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    _CancelBecauseExternalCollisionViolated,
    _ExternalCollisionAvoidanceTask,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World


# %% indexed external contacts
@pytest.mark.parametrize("collision_index", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_external_failure_reports_the_indexed_controller_contact(
    cylinder_bot_world: World, collision_index: int, reverse: bool
) -> None:
    """
    Detector ordering cannot change which violating pair is reported.

    :param cylinder_bot_world: Existing robot with two external obstacles.
    :param collision_index: Closest-contact slot whose task reports a violation.
    :param reverse: Whether the detector delivers contacts farthest first.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(MinimalRobot)[0]
    obstacles = [
        world.get_body_by_name(name) for name in ("environment", "environment2")
    ]
    manager = world.collision_manager
    manager.temporary_rules.extend(
        AvoidCollisionBetweenGroups(
            body_group_a=[robot.root],
            body_group_b=[obstacle],
            buffer_zone_distance=1.0,
            violated_distance=1.0,
        )
        for obstacle in obstacles
    )
    manager.max_avoided_bodies_rules.append(
        MaxAvoidedCollisionsOverride(2, {robot.root})
    )
    manager.update_collision_matrix()
    context = MotionStatechartContext(world)
    contacts = context.external_collision_manager
    contacts.register_group_of_body(robot.root)
    group = contacts.get_collision_group(robot.root)
    detected = manager.compute_collisions()
    expected = sorted(
        (
            contact if contact.body_a is robot.root else contact.reverse()
            for contact in detected.contacts
        ),
        key=lambda contact: contact.distance,
    )
    assert len(expected) == 2
    contacts.on_compute_collisions(
        CollisionCheckingResult(list(reversed(expected)) if reverse else expected)
    )
    task = _ExternalCollisionAvoidanceTask(
        collision_group=group,
        collision_index=collision_index,
        external_collision_manager=contacts,
    )
    statechart = MotionStatechart()
    statechart.add_node(task)
    statechart.observation_state[task] = ObservationStateValues.FALSE
    cancellation = _CancelBecauseExternalCollisionViolated(tasks=[task])
    try:
        with pytest.raises(CollisionViolatedError) as failure:
            cancellation.on_tick(context)
        [reported] = failure.value.violated_collisions
        assert reported.body_a is expected[collision_index].body_a
        assert reported.body_b is expected[collision_index].body_b
        assert reported.distance == task.contact_distance.evaluate()[0]
        assert failure.value.thresholds == [task.violated_distance.evaluate()[0]]
        assert contacts.last_closest_contacts[group] == expected
    finally:
        context.cleanup()
