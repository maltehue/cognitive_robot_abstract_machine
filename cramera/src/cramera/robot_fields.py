"""
Robot-instance fields shared by live APIs and saved scene files.
"""

from enum import StrEnum


class RobotField(StrEnum):
    """
    Stable names of robot metadata and independent root trajectories.
    """

    IDENTIFIER = "identifier"
    """
    Persistent native robot namespace.
    """

    LABEL = "label"
    """
    Display name of the semantic robot annotation.
    """

    MODEL = "model"
    """
    Robot annotation class available in the model catalog.
    """

    POSE = "pose"
    """
    World pose of the robot root.
    """

    JOINT_POSITIONS = "joint_positions"
    """
    Native one-degree-of-freedom connection values.
    """

    ROBOTS = "robots"
    """
    Every articulated instance in the scene.
    """

    ACTIVE_IDENTIFIER = "active_identifier"
    """
    Robot selected for live controls.
    """

    ACTIVE_ROBOT = "activeRobot"
    """
    Selected instance in a saved scene.
    """

    NAME = "name"
    """
    Legacy robot model name.
    """

    PREFIX = "prefix"
    """
    Native namespace of this model's links and joints.
    """

    PARTS = "parts"
    """
    Robot parts mapped to their link names.
    """

    PART_ANNOTATIONS = "partAnnotations"
    """
    Structured native arm and gripper annotations.
    """

    ROBOT = "robot"
    """
    Legacy selected robot metadata.
    """

    MODEL_BASES = "modelBases"
    """
    Per-instance root poses in a trajectory frame.
    """
