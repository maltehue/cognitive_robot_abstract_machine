from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.designator import Designator
from coraplex_mcp.authoring import CompositeCapabilityFactory
from coraplex_mcp.catalogue import CapabilityCatalogue
from coraplex_mcp.exceptions import UnknownSession
from coraplex_mcp.marshaling import ValueMarshaller

# %% session


@dataclass
class RobotSession:
    """
    A live workspace an agent builds a robot program against.

    Each session owns its capability catalogue and authoring factory, so capabilities
    authored in one session do not leak into another.
    """

    identifier: str
    """
    The identifier the session is addressed by.
    """

    context: Context
    """
    The world and robot the session's plans are built and performed against.
    """

    catalogue: CapabilityCatalogue
    """
    The capabilities this session can construct, including the ones authored in it.
    """

    factory: CompositeCapabilityFactory
    """
    The factory that authors new capabilities into this session's catalogue.
    """

    marshaller: ValueMarshaller
    """
    The marshaller that converts tool arguments against this session's world.
    """

    def construct_capability(
        self, capability_name: str, parameters: Dict[str, Any]
    ) -> Designator:
        """
        Build a capability instance from tool arguments against this session's world.

        :param capability_name: The name of the capability to construct.
        :param parameters: The tool arguments keyed by field name.
        :return: The constructed capability, ready to perform.
        :raises UnknownCapability: If the capability is not in this session's catalogue.
        """
        capability = self.catalogue.capability_type(capability_name)
        arguments = self.marshaller.marshal_parameters(
            capability, parameters, self.context.world
        )
        return capability(**arguments)


# %% registry


@dataclass
class SessionRegistry:
    """
    Holds the open robot sessions and hands out their capabilities.
    """

    base_catalogue: CapabilityCatalogue = field(
        default_factory=CapabilityCatalogue.from_installed_capabilities
    )
    """
    The catalogue of built-in capabilities each new session starts from.
    """

    _sessions: Dict[str, RobotSession] = field(default_factory=dict)
    """
    The open sessions keyed by identifier.
    """

    def open_session(self, identifier: str, context: Context) -> RobotSession:
        """
        Open a session with its own copy of the built-in catalogue.

        :param identifier: The identifier the session is addressed by.
        :param context: The world and robot the session operates on.
        :return: The opened session.
        """
        catalogue = self.base_catalogue.copy()
        marshaller = ValueMarshaller()
        factory = CompositeCapabilityFactory(catalogue=catalogue, marshaller=marshaller)
        session = RobotSession(identifier, context, catalogue, factory, marshaller)
        self._sessions[identifier] = session
        return session

    def session(self, identifier: str) -> RobotSession:
        """
        :param identifier: The identifier of the session to return.
        :return: The open session with that identifier.
        :raises UnknownSession: If no session is open under that identifier.
        """
        if identifier not in self._sessions:
            raise UnknownSession(identifier)
        return self._sessions[identifier]

    def identifiers(self) -> List[str]:
        """
        :return: The identifiers of the open sessions.
        """
        return list(self._sessions)

    def close_session(self, identifier: str) -> None:
        """
        :param identifier: The identifier of the session to discard.
        :raises UnknownSession: If no session is open under that identifier.
        """
        if identifier not in self._sessions:
            raise UnknownSession(identifier)
        del self._sessions[identifier]
