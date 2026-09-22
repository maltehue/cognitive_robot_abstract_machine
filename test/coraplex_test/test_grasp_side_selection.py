"""
Which side of an object the default grasp approaches from.

A robot that drives to the object and one that reaches from a fixed stand have to answer
this differently: the driving one has not arrived yet when the side is chosen, so its
current position must not decide it.
"""

import numpy as np

from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    AxisIdentifier,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription, PreferredGraspAlignment
from coraplex.view_manager import ViewManager
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% the object the robot is asked to grasp

OBJECT_BESIDE_THE_ROBOT = Pose.from_xyz_rpy(0.7, 2.0, 0.8)
"""
A pose well off to the robot's side, where the object's y axis, not its x axis, points
at a robot standing near the origin.
"""


def end_effector_of(world, robot_type, arm=Arms.LEFT):
    """
    One arm's end effector, as the grasp defaults are asked for it.

    :param world: The world holding the robot.
    :param robot_type: The robot's semantic annotation type.
    :param arm: Which arm to take the end effector of.
    """
    robot = world.get_semantic_annotations_by_type(robot_type)[0]
    return ViewManager.get_end_effector_view(arm, robot)


def pose_in(world, pose):
    """
    A pose expressed against the world's root, as the grasp defaults expect it.

    :param world: The world whose root the pose is expressed in.
    :param pose: The pose to re-reference.
    """
    return Pose(pose.to_position(), pose.to_quaternion(), reference_frame=world.root)


# %% a robot that drives to the object


class TestSideForADrivingRobot:
    """
    A mobile base navigates to the object after the side is chosen, so the side follows
    the object rather than the pose the robot happens to be standing in.
    """

    def test_the_object_beside_the_robot_is_still_approached_from_its_front(
        self, pr2_world_copy
    ):
        """
        Standing to the object's side, the side facing the robot is the object's y face.
        Choosing that one sends the robot around the object instead of to the front it
        could have driven to.
        """
        end_effector = end_effector_of(pr2_world_copy, PR2)

        grasp = GraspDescription.robot_relative_default(
            end_effector, pose_in(pr2_world_copy, OBJECT_BESIDE_THE_ROBOT)
        )

        assert grasp.approach_direction is ApproachDirection.FRONT
        assert grasp.vertical_alignment is VerticalAlignment.NoAlignment

    def test_a_rotated_object_is_approached_from_its_own_front(self, pr2_world_copy):
        """
        The side is the object's, so it turns with the object and the robot drives to
        wherever that is.
        """
        end_effector = end_effector_of(pr2_world_copy, PR2)
        turned_around = Pose.from_xyz_rpy(0.7, 2.0, 0.8, yaw=np.pi)

        grasp = GraspDescription.robot_relative_default(
            end_effector, pose_in(pr2_world_copy, turned_around)
        )

        assert grasp.approach_direction is ApproachDirection.FRONT

    def test_an_explicit_alignment_still_decides_the_side(self, pr2_world_copy):
        """
        A caller naming the axis to grasp along is answered from that axis, driving
        robot or not.
        """
        end_effector = end_effector_of(pr2_world_copy, PR2)
        along_the_y_axis = PreferredGraspAlignment(
            preferred_axis=AxisIdentifier.Y,
            with_vertical_alignment=False,
            with_rotated_gripper=False,
        )

        grasp = GraspDescription.robot_relative_default(
            end_effector,
            pose_in(pr2_world_copy, OBJECT_BESIDE_THE_ROBOT),
            grasp_alignment=along_the_y_axis,
        )

        assert grasp.approach_direction.axis is AxisIdentifier.Y


# %% a robot that reaches from where it stands


class TestSideForAStandingRobot:
    """
    A robot without a mobile base can only reach the side already facing it.
    """

    def test_the_side_facing_the_robot_is_approached(self, tracy_world):
        """
        The object lies off to the side, so the side facing this robot is its y face --
        the only one it can reach without a base to drive.
        """
        end_effector = end_effector_of(tracy_world, Tracy)

        grasp = GraspDescription.robot_relative_default(
            end_effector, pose_in(tracy_world, OBJECT_BESIDE_THE_ROBOT)
        )

        assert grasp.approach_direction.axis is AxisIdentifier.Y
