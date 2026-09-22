"""Owned offline processes retain incoming localhost HTTP and reject outbound TCP."""

from __future__ import annotations

import errno
import json
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import pytest

from cramera.offline_network import (
    OfflineNetworkArgument,
    OfflineNetworkLauncher,
    server_from_listener,
)

from .conftest import DATASET_DIR
from .dataset.offline_http_mimic import IsolatedProbeHandler, ProbeArgument


# %% deterministic child lifecycle
@dataclass
class ChildLifecycle:
    """A child whose wait outcomes make shutdown races reproducible."""

    outcomes: list[int | BaseException]
    """Successes or interruptions delivered by successive waits."""
    returncode: int | None = None
    """The recorded exit status, or None while running."""
    signals: list[int] = field(default_factory=list)
    """Signals explicitly sent to this child."""
    terminated: bool = False
    """Whether graceful termination was requested."""
    killed: bool = False
    """Whether forced termination was requested."""

    def wait(self, timeout: float | None = None) -> int:
        """Deliver the next wait outcome.

        :param timeout: The bounded wait requested by the owner.
        :return: This child's exit status.
        """
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.returncode = outcome
        return outcome

    def poll(self) -> int | None:
        """Report whether the child has exited."""
        return self.returncode

    def send_signal(self, requested_signal: int) -> None:
        """Record a signal addressed only to this child.

        :param requested_signal: Signal selected by the launcher.
        """
        self.signals.append(requested_signal)

    def terminate(self) -> None:
        """Record graceful termination."""
        self.terminated = True

    def kill(self) -> None:
        """Record forced termination."""
        self.killed = True


@dataclass
class ListenerHandoff:
    """Observe the actual bound listener passed to a deterministic child."""

    child: ChildLifecycle
    """The child returned to the owner."""
    command: list[str] = field(default_factory=list)
    """The complete namespace command."""
    address: tuple[str, int] | None = None
    """The host address owned by the inherited socket."""
    descriptor: int | None = None
    """The descriptor forwarded through the subprocess boundary."""

    def __call__(
        self,
        command: list[str],
        pass_fds: tuple[int, ...],
        start_new_session: bool,
    ) -> ChildLifecycle:
        """Record handoff without redirecting either output stream.

        :param command: Namespace command to execute.
        :param pass_fds: Descriptors explicitly inherited by the child.
        :param start_new_session: Whether the child owns its terminal signal session.
        :return: The controlled child lifecycle.
        """
        assert start_new_session
        [self.descriptor] = pass_fds
        with socket.fromfd(
            self.descriptor, socket.AF_INET, socket.SOCK_STREAM
        ) as listener:
            self.address = listener.getsockname()
            assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        self.command = command
        return self.child


@pytest.fixture
def isolated_namespaces() -> None:
    """Skip the integration test unless this Linux host permits network namespaces."""
    if sys.platform != "linux" or shutil.which("unshare") is None:
        pytest.skip("Linux unshare is unavailable")
    probe = subprocess.run(
        ["unshare", "--user", "--map-root-user", "--net", "true"],
        capture_output=True,
        timeout=3,
    )
    if probe.returncode:
        pytest.skip("This host does not permit isolated user/network namespaces")


# %% process ownership
def test_run_passes_only_the_listener_and_forwards_exit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child inherits one loopback listener and the command's existing arguments.

    :param monkeypatch: Restores the observed subprocess constructor.
    """
    child = ChildLifecycle([7])
    handoff = ListenerHandoff(child)
    monkeypatch.setattr(subprocess, "Popen", handoff)
    command = [sys.executable, "-m", "cramera.offline", "recording", "--no-browser"]
    launcher = OfflineNetworkLauncher(port=0, command=command)

    assert not launcher.started.is_set()
    assert launcher.run() == child.returncode == 7
    assert launcher.started.is_set()
    assert handoff.address[0] == "127.0.0.1"
    assert handoff.address[1] > 0
    assert handoff.command[-len(command) - 2 :] == [
        *command,
        OfflineNetworkArgument.LISTEN_FD,
        str(handoff.descriptor),
    ]
    assert not child.terminated
    assert not child.killed


def test_ctrl_c_reaches_only_the_owned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interruption requests graceful child shutdown and forwards its result.

    :param monkeypatch: Restores the deterministic child constructor.
    """
    child = ChildLifecycle([KeyboardInterrupt(), 0])
    monkeypatch.setattr(subprocess, "Popen", ListenerHandoff(child))
    assert OfflineNetworkLauncher(port=0, command=[sys.executable]).run() == 0
    assert child.signals == [signal.SIGINT]
    assert not child.terminated


def test_unresponsive_child_is_terminated_then_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child ignoring Ctrl-C is bounded by the owner's shutdown timeout.

    :param monkeypatch: Restores the deterministic child constructor.
    """
    timeout = subprocess.TimeoutExpired(sys.executable, 1)
    child = ChildLifecycle([KeyboardInterrupt(), timeout, timeout, 9, 9])
    monkeypatch.setattr(subprocess, "Popen", ListenerHandoff(child))
    assert OfflineNetworkLauncher(port=0, command=[sys.executable]).run() == 9
    assert child.terminated
    assert child.killed


def test_exception_during_wait_releases_the_owned_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected wait failure cannot leave this launcher's child running.

    :param monkeypatch: Restores the deterministic child constructor.
    """
    failure = OSError(errno.EIO, "wait failed")
    child = ChildLifecycle([failure, 0])
    monkeypatch.setattr(subprocess, "Popen", ListenerHandoff(child))
    with pytest.raises(OSError) as caught:
        OfflineNetworkLauncher(port=0, command=[sys.executable]).run()
    assert caught.value is failure
    assert child.terminated
    assert not child.killed


def test_stopping_an_unstarted_launcher_does_nothing() -> None:
    """A launcher without a child has no process to signal."""
    OfflineNetworkLauncher(port=0, command=[sys.executable]).stop()


def test_listener_conflict_never_announces_startup() -> None:
    """An occupied host port cannot release the browser readiness gate."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen()
        launcher = OfflineNetworkLauncher(
            port=existing.getsockname()[1], command=[sys.executable]
        )
        with pytest.raises(OSError) as error:
            launcher.run()
        assert error.value.errno == errno.EADDRINUSE
        assert not launcher.started.is_set()
        assert launcher.process is None


def test_child_creation_failure_never_announces_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing isolation command leaves the readiness gate closed.

    :param monkeypatch: Restore the unavailable namespace command.
    """
    monkeypatch.setattr(OfflineNetworkLauncher, "ISOLATOR", "/nonexistent/unshare")
    launcher = OfflineNetworkLauncher(port=0, command=[sys.executable])
    with pytest.raises(FileNotFoundError):
        launcher.run()
    assert not launcher.started.is_set()
    assert launcher.process is None


# %% inherited listener ownership
def test_server_takes_ownership_of_the_supplied_listener() -> None:
    """Adopting an existing socket preserves its bound address and closes it on exit."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    address = listener.getsockname()
    descriptor = listener.detach()
    with server_from_listener(descriptor, IsolatedProbeHandler) as server:
        assert server.server_address == address
        assert server.socket.fileno() == descriptor
        assert server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
    assert server.socket.fileno() == -1


# %% actual network isolation
def test_isolated_child_answers_localhost_but_cannot_open_external_route(
    tmp_path: Path, isolated_namespaces: None
) -> None:
    """HTTP remains reachable from the host while outbound TCP has no route.

    :param tmp_path: Temporary readiness-report directory.
    :param isolated_namespaces: Confirms Linux namespace support without credentials.
    """
    report = tmp_path / "listener.json"
    command = [
        sys.executable,
        str(DATASET_DIR / "offline_http_mimic.py"),
        ProbeArgument.REPORT,
        str(report),
        ProbeArgument.EXIT_CODE,
        "7",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while (
            not report.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert report.exists(), process.communicate(timeout=1)
        address, port = json.loads(report.read_text())
        with urlopen(f"http://{address}:{port}/", timeout=2) as response:
            assert json.load(response) == errno.ENETUNREACH
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 7
        assert output == (IsolatedProbeHandler.READY_MESSAGE + "\n").encode()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
