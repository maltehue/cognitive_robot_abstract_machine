"""Run a local HTTP viewer in a process without external network routes."""

from __future__ import annotations

import signal
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from socketserver import BaseRequestHandler, ThreadingTCPServer

from typing_extensions import ClassVar


# %% network namespace ownership
class OfflineNetworkArgument(StrEnum):
    """Namespace and listener arguments shared across the process boundary."""

    USER = "--user"
    """Create an isolated user namespace."""
    MAP_ROOT = "--map-root-user"
    """Map the current user to the namespace's root identity."""
    NETWORK = "--net"
    """Create a network namespace without external interfaces."""
    LISTEN_FD = "--listen-fd"
    """Pass the existing host-loopback listener to the child."""


@dataclass
class OfflineNetworkLauncher:
    """Own an isolated process reachable through one inherited localhost listener."""

    ISOLATOR: ClassVar[str] = "unshare"
    """Linux command creating the isolated user and network namespaces."""
    LOOPBACK_ADDRESS: ClassVar[str] = "127.0.0.1"
    """Host address exposed to the local browser."""
    SHUTDOWN_TIMEOUT: ClassVar[float] = 5.0
    """Seconds allowed for each graceful child shutdown attempt."""

    port: int
    """Host TCP port, with zero requesting an available ephemeral port."""
    command: list[str]
    """Child command, which accepts the appended inherited-listener argument."""
    process: subprocess.Popen[bytes] | None = field(
        default=None, init=False, repr=False
    )
    """The child process owned by this launcher."""
    started: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    """Signal that the listener is bound and the isolated child was created."""

    def run(self) -> int:
        """Run the isolated child with inherited output and release it on every exit.

        :return: The child's exit status, including after an interrupted wait.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.LOOPBACK_ADDRESS, self.port))
            listener.listen()
            descriptor = listener.fileno()
            self.process = subprocess.Popen(
                [
                    self.ISOLATOR,
                    OfflineNetworkArgument.USER,
                    OfflineNetworkArgument.MAP_ROOT,
                    OfflineNetworkArgument.NETWORK,
                    *self.command,
                    OfflineNetworkArgument.LISTEN_FD,
                    str(descriptor),
                ],
                pass_fds=(descriptor,),
                start_new_session=True,
            )
            self.started.set()
            try:
                return self.process.wait()
            except KeyboardInterrupt:
                self.process.send_signal(signal.SIGINT)
                try:
                    return self.process.wait(timeout=self.SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self.stop()
                    return self.process.wait()
            finally:
                self.stop()

    def stop(self) -> None:
        """Terminate and reap this launcher's child with a bounded graceful wait."""
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=self.SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


# %% inherited HTTP listener
def server_from_listener(
    descriptor: int, handler: type[BaseRequestHandler]
) -> ThreadingTCPServer:
    """Adopt a listening socket without opening another network endpoint.

    :param descriptor: Open listener whose ownership transfers to the returned server.
    :param handler: Request handler serving the inherited connections.
    :return: A ready server that closes the inherited descriptor when closed.
    """
    listener = socket.socket(fileno=descriptor)
    server = ThreadingTCPServer(
        listener.getsockname(), handler, bind_and_activate=False
    )
    server.socket.close()
    server.socket = listener
    server.daemon_threads = True
    return server
