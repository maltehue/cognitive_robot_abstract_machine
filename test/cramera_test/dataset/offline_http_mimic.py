"""An HTTP process exposing the result of its isolated network probe."""

from __future__ import annotations

import json
import socket
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from typing_extensions import ClassVar

from cramera.offline_network import (
    OfflineNetworkArgument,
    OfflineNetworkLauncher,
    server_from_listener,
)


# %% isolated HTTP process
class ProbeArgument(StrEnum):
    """Arguments shared by the launcher and its HTTP child."""

    REPORT = "--report"
    """File announcing the bound HTTP address."""
    EXIT_CODE = "--exit-code"
    """Exit status after answering one request."""


@dataclass(init=False)
class IsolatedProbeHandler(BaseHTTPRequestHandler):
    """Answer with the result of opening a socket in the current namespace."""

    TEST_NETWORK_ADDRESS: ClassVar[str] = "192.0.2.1"
    """Reserved TEST-NET address, contacted only inside the isolated namespace."""
    PROBE_PORT: ClassVar[int] = 443
    """TCP port used to verify the absence of an external route."""
    READY_MESSAGE: ClassVar[str] = "isolated HTTP ready"
    """Child output proving stdout remains inherited across isolation."""

    def do_GET(self) -> None:
        """Report the outbound connection error through the inherited listener."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            result = connection.connect_ex((self.TEST_NETWORK_ADDRESS, self.PROBE_PORT))
        body = json.dumps(result).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(arguments: list[str]) -> int:
    """Launch an isolated child or answer one local HTTP request.

    :param arguments: Command-line arguments without the script name.
    :return: The requested child exit status.
    """
    parser = ArgumentParser()
    parser.add_argument(ProbeArgument.REPORT, type=Path, required=True)
    parser.add_argument(ProbeArgument.EXIT_CODE, type=int, default=0)
    parser.add_argument(OfflineNetworkArgument.LISTEN_FD, type=int)
    options = parser.parse_args(arguments)
    if options.listen_fd is None:
        return OfflineNetworkLauncher(
            port=0, command=[sys.executable, __file__, *arguments]
        ).run()
    with server_from_listener(options.listen_fd, IsolatedProbeHandler) as server:
        server.daemon_threads = False
        options.report.write_text(json.dumps(server.server_address))
        print(IsolatedProbeHandler.READY_MESSAGE, flush=True)
        server.handle_request()
    return options.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
