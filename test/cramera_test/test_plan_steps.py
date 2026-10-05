"""
A plan as the Plan Builder writes it: reading it from the builder's form, writing it
back, and turning it into the coraplex actions that carry it out.
"""

from __future__ import annotations

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from krrood.adapters.json_serializer import from_json, to_json
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table

from cramera.live.placement_surface import PlacementSurface
from cramera.model_catalog import BuilderStep
from cramera.plan_steps import (
    BuilderPlan,
    LevelPose,
    LookAt,
    MalformedPlanError,
    MoveTorso,
    ObjectStepInPlanError,
    ParkArms,
    Pick,
    PickUpWithDefaultGraspAction,
    Place,
    StepField,
    SurfaceTarget,
    Transport,
)

from .test_live_bridge import shaped_body, world_with


def context_on(world) -> Context:
    """
    The context a step is resolved against, for a scene with no robot in it.
    """
    return Context(world=world, robot=None)


def step(step_type: BuilderStep, **parameters) -> dict:
    """
    One step in the builder's form.
    """
    return {StepField.TYPE: step_type.value, StepField.PARAMETERS: parameters}


def read(*steps) -> BuilderPlan:
    return BuilderPlan.from_payload(list(steps))


# %% reading a plan


class TestReadingAPlan:
    def test_a_plan_keeps_the_order_its_steps_were_written_in(self):
        plan = read(
            step(BuilderStep.PARK_ARMS, arm="BOTH"),
            step(BuilderStep.MOVE_TORSO, torso="HIGH"),
        )
        assert [type(found) for found in plan.steps] == [ParkArms, MoveTorso]
        assert plan.steps[0].arm is Arms.BOTH
        assert plan.steps[1].torso_state is TorsoState.HIGH

    def test_a_plan_without_steps_is_a_plan_of_nothing(self):
        assert BuilderPlan.from_payload([]).steps == []

    def test_a_step_of_an_unknown_type_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read({StepField.TYPE: "make_coffee", StepField.PARAMETERS: {}})

    def test_a_step_acting_on_an_object_names_it(self):
        plan = read(
            step(BuilderStep.PICK, object="milk.stl", arm="LEFT"),
            step(
                BuilderStep.PLACE,
                object="milk.stl",
                arm="LEFT",
                targetMode="pose",
                x=2.4,
                y=1.8,
                z=0.8,
                yaw=0.0,
            ),
            step(
                BuilderStep.TRANSPORT,
                object="milk.stl",
                arm="RIGHT",
                targetMode="semantic",
                surfaceType="Table",
                surfaceName="",
            ),
        )
        assert [type(found) for found in plan.steps] == [Pick, Place, Transport]
        assert plan.steps[0] == Pick(object_name="milk.stl", arm=Arms.LEFT)
        assert plan.steps[1].target == LevelPose(2.4, 1.8, 0.8, 0.0)
        assert plan.steps[2].target == SurfaceTarget("Table", None)
        assert plan.steps[2].arm is Arms.RIGHT

    def test_a_place_without_a_target_mode_is_at_a_pose(self):
        [place] = read(
            step(BuilderStep.PLACE, object="milk.stl", arm="LEFT", x=1, y=2, z=3, yaw=0)
        ).steps
        assert place.target == LevelPose(1.0, 2.0, 3.0, 0.0)

    def test_a_step_acting_on_an_object_must_name_one(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.PICK, object="", arm="LEFT"))

    def test_a_surface_nothing_is_put_down_on_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(
                step(
                    BuilderStep.PLACE,
                    object="milk.stl",
                    arm="LEFT",
                    targetMode="semantic",
                    surfaceType="Ceiling",
                )
            )

    def test_a_target_mode_that_is_neither_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(
                step(
                    BuilderStep.TRANSPORT,
                    object="milk.stl",
                    arm="LEFT",
                    targetMode="roughly",
                )
            )

    def test_a_step_looking_for_an_object_is_refused(self):
        with pytest.raises(ObjectStepInPlanError):
            read(step(BuilderStep.DETECT, object="cheeze_it.obj"))

    def test_an_arm_the_robot_does_not_have_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.PARK_ARMS, arm="THIRD"))

    def test_a_torso_state_that_is_not_one_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.MOVE_TORSO, torso="SLIGHTLY_UP"))

    def test_a_coordinate_that_is_not_a_number_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.NAVIGATE, x="over there", y=1.0, z=0.0, yaw=0.0))

    def test_a_point_to_look_at_needs_all_three_coordinates(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.LOOK_AT, x=1.0, y=2.0))


# %% writing a plan back


@pytest.mark.parametrize(
    "written",
    [
        step(BuilderStep.PARK_ARMS, arm="LEFT"),
        step(BuilderStep.MOVE_TORSO, torso="LOW"),
        step(BuilderStep.NAVIGATE, x=2.6, y=1.8, z=0.0, yaw=1.57),
        step(BuilderStep.LOOK_AT, x=1.0, y=-0.5, z=1.2),
        step(BuilderStep.PICK, object="milk.stl", arm="RIGHT"),
        step(
            BuilderStep.PLACE,
            object="milk.stl",
            arm="LEFT",
            targetMode="pose",
            x=2.4,
            y=1.8,
            z=0.8,
            yaw=0.5,
        ),
        step(
            BuilderStep.TRANSPORT,
            object="milk.stl",
            arm="LEFT",
            targetMode="semantic",
            surfaceType="CounterTop",
            surfaceName="apartment/island_countertop",
        ),
    ],
)
def test_a_step_reads_back_as_it_was_written(written):
    plan = read(written)

    assert read(*plan.to_payload()) == plan
    assert from_json(to_json(plan)) == plan


# %% the actions a plan performs


class TestActionsAPlanPerforms:
    def test_parking_arms_parks_the_named_arm(self):
        action = (
            read(step(BuilderStep.PARK_ARMS, arm="LEFT"))
            .steps[0]
            .action(context_on(world_with()))
        )
        assert isinstance(action, ParkArmsAction)
        assert action.arm is Arms.LEFT

    def test_moving_the_torso_moves_it_to_the_named_state(self):
        action = (
            read(step(BuilderStep.MOVE_TORSO, torso="LOW"))
            .steps[0]
            .action(context_on(world_with()))
        )
        assert isinstance(action, MoveTorsoAction)
        assert action.torso_state is TorsoState.LOW

    def test_navigating_goes_to_the_given_place(self):
        world = world_with()
        action = (
            read(step(BuilderStep.NAVIGATE, x=2.6, y=1.8, z=0.0, yaw=1.57))
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, NavigateAction)
        assert action.target_location.to_position().to_np()[:3] == pytest.approx(
            [2.6, 1.8, 0.0]
        )

    def test_looking_at_a_point_aims_at_that_point(self):
        world = world_with()
        [look] = read(step(BuilderStep.LOOK_AT, x=1.0, y=-0.5, z=1.2)).steps
        action = look.action(context_on(world))

        assert isinstance(action, LookAtAction)
        assert isinstance(look, LookAt)
        assert action.target.to_position().to_np()[:3] == pytest.approx(
            [look.target.x, look.target.y, look.target.z]
        )
        assert action.target.reference_frame is world.root

    def test_picking_takes_the_named_object_with_the_named_arm(self):
        milk = shaped_body("demo", "milk.stl")
        world = world_with(milk)
        action = (
            read(step(BuilderStep.PICK, object="milk.stl", arm="RIGHT"))
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, PickUpWithDefaultGraspAction)
        assert action.object_designator.root is milk
        assert action.arm is Arms.RIGHT

    def test_an_object_the_world_does_not_hold_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.PICK, object="milk.stl", arm="LEFT")).steps[
                0
            ].action(context_on(world_with()))

    def test_placing_at_a_pose_puts_the_object_there(self):
        milk = shaped_body("demo", "milk.stl")
        world = world_with(milk)
        action = (
            read(
                step(
                    BuilderStep.PLACE,
                    object="milk.stl",
                    arm="LEFT",
                    targetMode="pose",
                    x=2.4,
                    y=1.8,
                    z=0.8,
                    yaw=0.0,
                )
            )
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, PlaceAction)
        assert action.object_designator is milk
        assert action.target_location.to_position().to_np()[:3] == pytest.approx(
            [2.4, 1.8, 0.8]
        )

    def test_placing_on_a_surface_leaves_the_pose_to_be_found(self):
        milk = shaped_body("demo", "milk.stl")
        table = shaped_body("demo", "table")
        world = world_with(milk, table)
        with world.modify_world():
            world.add_semantic_annotation(Table(root=table))
        action = (
            read(
                step(
                    BuilderStep.PLACE,
                    object="milk.stl",
                    arm="LEFT",
                    targetMode="semantic",
                    surfaceType="Table",
                    surfaceName="demo/table",
                )
            )
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, Match)

    def test_transporting_carries_the_object_to_a_surface(self):
        milk = shaped_body("demo", "milk.stl")
        world = world_with(milk)
        action = (
            read(
                step(
                    BuilderStep.TRANSPORT,
                    object="milk.stl",
                    arm="LEFT",
                    targetMode="semantic",
                    surfaceType="Table",
                )
            )
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, TransportAction)
        assert action.object_designator.root is milk
        assert isinstance(action.target_location, PlacementSurface)
        assert action.target_location.surface_type is Table
        assert action.target_location.surface_name is None
