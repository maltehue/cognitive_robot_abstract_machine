"""
The Plan Builder reads the model inventory through the existing HTTP server.
"""

from cramera.model_catalog import ModelCatalog

from .test_server import server, get_json

# %% catalog delivery


def test_catalog_endpoint(server) -> None:
    """
    The authoring endpoint serves the installed catalog without creating a world.

    :param server: Existing server fixture bound to an ephemeral port.
    """
    assert (
        get_json(server + "/api/plan/catalog") == ModelCatalog.installed().to_payload()
    )
