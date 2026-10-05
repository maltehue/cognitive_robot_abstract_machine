"""
What moving each of a robot's joints costs the controller.

Giskard spreads a motion over every joint that can contribute to it, each at the same
price unless told otherwise. A robot's parts say where that is wrong: a torso that has
to keep a humanoid balanced costs more to bend than an arm costs to reach, see
:attr:`~semantic_digital_twin.robots.robot_parts.Torso.motion_cost`. Every controller
coraplex builds for a robot takes its configuration from here, so a part's cost holds
in a motion, in a reachability check and in a validation alike.
"""

from __future__ import annotations

from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from typing_extensions import Any, Optional


def controller_config(robot: Optional[AbstractRobot], **settings: Any) -> QPControllerConfig:
    """
    :param robot: The robot the controller moves, or ``None`` for a controller that
        moves no robot in particular.
    :param settings: The controller configuration's own settings, the target frequency
        and prediction horizon among them.
    :return: The configuration, weighing the robot's joints as its parts say.
    """
    config = QPControllerConfig(**settings)
    if robot is None:
        return config
    torso = robot.get_torso_if_specified()
    if torso is None or torso.motion_cost == 1.0:
        return config
    for connection in torso.connections:
        for dof in getattr(connection, "active_dofs", []):
            config.set_dof_weight(
                dof.name,
                Derivatives.velocity,
                config.get_degree_of_freedom_weight(dof.name, Derivatives.velocity)
                * torso.motion_cost,
            )
    return config
