from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any, Callable, Mapping

from coraplex.datastructures.dataclasses import Context
from coraplex_mcp.exceptions import MalformedWorldEntryPoint
from semantic_digital_twin.world import World

# %% provider

_WORLD_ENTRY_POINT_VARIABLE = "CORAPLEX_MCP_WORLD"
"""
The environment variable naming the ``module:function`` that supplies the belief a
session designs against.
"""


class WorldProvider(ABC):
    """
    Supplies the world and robot a new session operates on.

    Abstracting the world source keeps the server independent of any particular robot,
    environment, or asset-loading mechanism.
    """

    @abstractmethod
    def create_context(self) -> Context:
        """
        :return: A fresh context holding the world and robot for a new session.
        """


@dataclass
class Pr2WorldProvider(WorldProvider):
    """
    Builds a PR2 in an empty world, loaded from the robot's URDF.
    """

    def create_context(self) -> Context:
        # ``URDFParser`` pulls in the ROS ``xacro`` toolchain; importing it here keeps
        # that dependency at the deployment boundary so the rest of the package imports
        # without a ROS installation.
        from semantic_digital_twin.adapters.urdf import URDFParser
        from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
        from semantic_digital_twin.robots.pr2 import PR2
        from semantic_digital_twin.spatial_types.spatial_types import (
            HomogeneousTransformationMatrix,
        )
        from semantic_digital_twin.world_description.connections import (
            Connection6DoF,
            OmniDrive,
        )
        from semantic_digital_twin.world_description.world_entity import Body

        world = URDFParser.from_file(file_path=PR2.get_ros_file_path()).parse()
        robot = PR2.from_world(world)
        with world.modify_world():
            robot_root = world.root
            map_body = Body(name=PrefixedName("map"))
            localization_body = Body(name=PrefixedName("odom_combined"))
            world.add_connection(
                Connection6DoF.create_with_dofs(world, map_body, localization_body)
            )
            drive = OmniDrive.create_with_dofs(
                parent=localization_body, child=robot_root, world=world
            )
            world.add_connection(drive)
            drive.has_hardware_interface = True
        return Context(world=world, robot=robot)


@dataclass
class CallableWorldProvider(WorldProvider):
    """
    Binds sessions to an existing belief supplied by a callable.

    The callable may return a :class:`Context` or a :class:`World`; a world is wrapped
    in a context around its first robot. This is how a session designs against an
    already-built or fetched semantic digital twin rather than a fresh one.
    """

    context_factory: Callable[[], Any]
    """
    The callable that returns the belief, invoked once per opened session.
    """

    def create_context(self) -> Context:
        result = self.context_factory()
        if isinstance(result, Context):
            return result
        if isinstance(result, World):
            return Context.from_world(result)
        raise MalformedWorldEntryPoint(
            repr(self.context_factory),
            f"returned {type(result).__name__}, expected World or Context",
        )


def world_provider_from_environment(environment: Mapping[str, str]) -> WorldProvider:
    """
    Resolve the world provider from the environment.

    When the world entry-point variable names a ``module:function`` returning a belief,
    sessions are bound to that belief; otherwise a fresh PR2 world is used.

    :param environment: The process environment to read the entry point from.
    :return: The resolved world provider.
    :raises MalformedWorldEntryPoint: If the entry point is set but not importable.
    """
    entry_point = environment.get(_WORLD_ENTRY_POINT_VARIABLE)
    if not entry_point:
        return Pr2WorldProvider()
    return CallableWorldProvider(_resolve_entry_point(entry_point))


def _resolve_entry_point(entry_point: str) -> Callable[[], Any]:
    """
    :param entry_point: A ``module:function`` reference.
    :return: The referenced callable.
    :raises MalformedWorldEntryPoint: If the reference is not a resolvable callable.
    """
    if entry_point.count(":") != 1:
        raise MalformedWorldEntryPoint(entry_point, "not of the form 'module:function'")
    module_name, attribute_name = entry_point.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as missing_module:
        raise MalformedWorldEntryPoint(entry_point, str(missing_module))
    if not hasattr(module, attribute_name):
        raise MalformedWorldEntryPoint(
            entry_point, f"module {module_name!r} has no {attribute_name!r}"
        )
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise MalformedWorldEntryPoint(entry_point, "does not name a callable")
    return factory
