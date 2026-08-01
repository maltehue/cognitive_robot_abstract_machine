import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex_mcp.catalogue import CapabilityCatalogue
from coraplex_mcp.sessions import SessionRegistry
from coraplex_mcp.world_provider import Pr2WorldProvider


@pytest.fixture(scope="session")
def catalogue() -> CapabilityCatalogue:
    """
    The catalogue of installed capabilities, built once for the whole test session.
    """
    return CapabilityCatalogue.from_installed_capabilities()


@pytest.fixture
def simulated_context() -> Context:
    """
    A context holding a PR2 in a fresh world.

    Building the world loads the robot URDF through the ROS ``xacro`` toolchain; the
    fixture skips when that toolchain is not installed, so the world-dependent tests run
    in an environment with ROS and are skipped elsewhere.
    """
    try:
        return Pr2WorldProvider().create_context()
    except ModuleNotFoundError as missing_dependency:
        pytest.skip(f"world building requires ROS: {missing_dependency}")


@pytest.fixture
def registry() -> SessionRegistry:
    """
    A session registry seeded with the installed capabilities.
    """
    return SessionRegistry()
