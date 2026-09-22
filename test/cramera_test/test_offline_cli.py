"""
Offline command dispatch owns its serving process and browser startup.
"""

from __future__ import annotations

import os
import sys
import threading
from http import HTTPStatus
from pathlib import Path
from socketserver import ThreadingTCPServer
from unittest.mock import MagicMock, Mock

import pytest

from cramera import offline
from cramera.offline import OfflineBrowser, OfflineHandler, OfflineOptions, OfflineTour

from .conftest import DATASET_DIR


# %% prepared recording
@pytest.fixture
def prepared_tour(fixture_scene: Path) -> OfflineTour:
    """
    Copy the existing recorded fixture into an independent presentation.

    :param fixture_scene: Existing miniature recording and architecture.
    :return: Validated presentation with the dataset storyboard.
    """
    return OfflineTour.prepare(
        fixture_scene / "scenes",
        fixture_scene / "presentation",
        DATASET_DIR / "offline_storyboard.json",
    )


def test_check_only_uses_process_arguments_without_starting_services(
    prepared_tour: OfflineTour,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Validation exits before constructing a server, child or browser.

    :param prepared_tour: Complete miniature recorded presentation.
    :param monkeypatch: Restore process arguments and observed service constructors.
    :param capsys: Capture the validation summary.
    """
    server = Mock()
    launcher = Mock()
    browser = Mock()
    monkeypatch.setattr(offline.socketserver, "ThreadingTCPServer", server)
    monkeypatch.setattr(offline, "OfflineNetworkLauncher", launcher)
    monkeypatch.setattr(offline.webbrowser, "open", browser)
    monkeypatch.setattr(
        sys, "argv", ["cramera.offline", str(prepared_tour.directory), "--check-only"]
    )

    offline.main()

    assert str(len(prepared_tour.chapters)) in capsys.readouterr().out
    server.assert_not_called()
    launcher.assert_not_called()
    browser.assert_not_called()


def test_prepare_and_check_command_exports_a_usable_tour(fixture_scene: Path) -> None:
    """
    The preparation command creates a validated presentation before returning.

    :param fixture_scene: Existing miniature recording and architecture.
    """
    destination = fixture_scene / "exported"
    offline.main(
        [
            str(destination),
            "--prepare-from",
            str(fixture_scene / "scenes"),
            "--storyboard",
            str(DATASET_DIR / "offline_storyboard.json"),
            "--check-only",
        ]
    )
    tour = OfflineTour(destination)
    tour.validate()
    assert (
        tour.storyboard_path.read_bytes()
        == (DATASET_DIR / "offline_storyboard.json").read_bytes()
    )


# %% serving ownership
@pytest.mark.parametrize("inherited_descriptor", [None, 17])
def test_server_selection_preserves_bundle_and_closes_on_interrupt(
    prepared_tour: OfflineTour,
    inherited_descriptor: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Normal and inherited listeners use the selected bundle and close on Ctrl-C.

    :param prepared_tour: Complete miniature recorded presentation.
    :param inherited_descriptor: Optional listener provided by an isolated parent.
    :param monkeypatch: Restore listener constructors and browser operations.
    """
    server = MagicMock(spec=ThreadingTCPServer)
    server.__enter__.return_value = server
    server.serve_forever.side_effect = KeyboardInterrupt
    normal_server = Mock(return_value=server)
    inherited_server = Mock(return_value=server)
    browser = Mock()
    monkeypatch.setattr(offline.socketserver, "ThreadingTCPServer", normal_server)
    monkeypatch.setattr(offline, "server_from_listener", inherited_server)
    monkeypatch.setattr(offline.webbrowser, "open", browser)
    arguments = [str(prepared_tour.directory), "--no-browser"]
    if inherited_descriptor is not None:
        arguments.extend(["--listen-fd", str(inherited_descriptor)])
    options = OfflineOptions.parse(arguments)

    offline.main(arguments)

    if inherited_descriptor is None:
        normal_server.assert_called_once_with(
            ("127.0.0.1", options.port), OfflineHandler
        )
        inherited_server.assert_not_called()
    else:
        inherited_server.assert_called_once_with(inherited_descriptor, OfflineHandler)
        normal_server.assert_not_called()
    assert Path(os.environ["CRAMERA_DATA"]) == prepared_tour.directory.resolve()
    assert Path(os.environ["CRAMERA_SCENES"]) == (
        prepared_tour.directory.resolve() / "scenes"
    )
    assert os.environ["CRAMERA_SCENE"] == prepared_tour.scene_names[0]
    server.serve_forever.assert_called_once_with()
    server.__exit__.assert_called_once_with(None, None, None)
    browser.assert_not_called()


def test_normal_server_opens_the_tour_and_preserves_serve_errors(
    prepared_tour: OfflineTour, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A serving error closes the owned listener and retains the original exception.

    :param prepared_tour: Complete miniature recorded presentation.
    :param monkeypatch: Restore serving and desktop browser operations.
    """
    failure = OSError("listener failed")
    server = MagicMock(spec=ThreadingTCPServer)
    server.__enter__.return_value = server
    server.serve_forever.side_effect = failure
    monkeypatch.setattr(
        offline.socketserver, "ThreadingTCPServer", Mock(return_value=server)
    )
    browser = Mock()
    monkeypatch.setattr(offline.webbrowser, "open", browser)
    arguments = [str(prepared_tour.directory)]
    options = OfflineOptions.parse(arguments)

    with pytest.raises(OSError) as error:
        offline.main(arguments)

    assert error.value is failure
    browser.assert_called_once_with(f"http://localhost:{options.port}/tour.html")
    assert server.__exit__.call_args.args[1] is failure


# %% isolated child dispatch
@pytest.mark.parametrize("open_browser", [False, True])
def test_isolated_child_receives_only_serving_options_and_forwards_exit_status(
    prepared_tour: OfflineTour,
    open_browser: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Isolation starts one child and cancels its browser waiter when the child exits.

    :param prepared_tour: Complete miniature recorded presentation.
    :param open_browser: Whether the parent should schedule browser readiness.
    :param monkeypatch: Restore process, thread and browser constructors.
    """
    arguments = [str(prepared_tour.directory), "--isolated"]
    if not open_browser:
        arguments.append("--no-browser")
    options = OfflineOptions.parse(arguments)
    browser = OfflineBrowser(f"http://localhost:{options.port}/tour.html")
    browser_factory = Mock(return_value=browser)
    worker = Mock()
    child = Mock()
    child.started = threading.Event()
    child.run.return_value = 7
    launcher = Mock(return_value=child)
    normal_server = Mock()
    monkeypatch.setattr(offline, "OfflineBrowser", browser_factory)
    monkeypatch.setattr(offline.threading, "Thread", worker)
    monkeypatch.setattr(offline, "OfflineNetworkLauncher", launcher)
    monkeypatch.setattr(offline.socketserver, "ThreadingTCPServer", normal_server)

    with pytest.raises(SystemExit) as exit_status:
        offline.main(arguments)

    assert exit_status.value.code == child.run.return_value
    port, command = launcher.call_args.args
    assert port == options.port
    assert command[:3] == [sys.executable, "-m", "cramera.offline"]
    child_options = OfflineOptions.parse(command[3:])
    assert child_options.directory == prepared_tour.directory.resolve()
    assert child_options.port == options.port
    assert child_options.isolated is False
    assert child_options.open_browser is False
    assert child_options.prepare_from is None
    assert child_options.listen_descriptor is None
    child.run.assert_called_once_with()
    normal_server.assert_not_called()
    assert browser.stopped.is_set()
    browser_factory.assert_called_once_with(browser.url, start_signal=child.started)
    if open_browser:
        worker.assert_called_once_with(target=browser.open_when_ready, daemon=True)
        worker.return_value.start.assert_called_once_with()
    else:
        worker.assert_not_called()


def test_failed_isolated_launch_cancels_browser_startup(
    prepared_tour: OfflineTour, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A namespace startup failure cannot leave a pending browser-open request.

    :param prepared_tour: Complete miniature recorded presentation.
    :param monkeypatch: Restore child and browser startup operations.
    """
    browser = OfflineBrowser("http://localhost/tour.html")
    failure = FileNotFoundError("unshare")
    child = Mock()
    child.run.side_effect = failure
    monkeypatch.setattr(offline, "OfflineBrowser", Mock(return_value=browser))
    monkeypatch.setattr(offline.threading, "Thread", Mock())
    monkeypatch.setattr(offline, "OfflineNetworkLauncher", Mock(return_value=child))

    with pytest.raises(FileNotFoundError) as error:
        offline.main([str(prepared_tour.directory), "--isolated"])

    assert error.value is failure
    assert browser.stopped.is_set()


# %% readiness deadlines
@pytest.mark.parametrize(
    "status", [HTTPStatus.NOT_FOUND, HTTPStatus.SERVICE_UNAVAILABLE]
)
def test_nonready_http_responses_expire_without_opening_browser(
    status: HTTPStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A responding endpoint must serve the actual page before a browser can open.

    :param status: HTTP response indicating that the presentation is unavailable.
    :param monkeypatch: Restore the readiness clock, request and desktop browser.
    """
    browser = OfflineBrowser("http://localhost/tour.html", timeout_seconds=1)
    response = MagicMock()
    response.__enter__.return_value.status = status
    request = Mock(return_value=response)
    browser_open = Mock()
    monkeypatch.setattr(offline, "urlopen", request)
    monkeypatch.setattr(offline.webbrowser, "open", browser_open)
    monkeypatch.setattr(offline.time, "monotonic", Mock(side_effect=[0, 0, 1]))
    monkeypatch.setattr(browser.stopped, "wait", Mock(return_value=False))

    assert browser.open_when_ready() is False

    request.assert_called_once()
    assert request.call_args.args[0].method == "HEAD"
    browser_open.assert_not_called()


@pytest.mark.parametrize("listener_owned", [False, True])
def test_readiness_probe_waits_until_this_launch_owns_the_listener(
    listener_owned: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An existing server cannot satisfy readiness before this process binds.

    :param listener_owned: Whether the launcher has created its listening child.
    :param monkeypatch: Restore HTTP, browser and readiness-clock operations.
    """
    started = threading.Event()
    if listener_owned:
        started.set()
    browser = OfflineBrowser(
        "http://localhost/tour.html", timeout_seconds=1, start_signal=started
    )
    response = MagicMock()
    response.__enter__.return_value.status = HTTPStatus.OK
    request = Mock(return_value=response)
    browser_open = Mock()
    monkeypatch.setattr(offline, "urlopen", request)
    monkeypatch.setattr(offline.webbrowser, "open", browser_open)
    monkeypatch.setattr(offline.time, "monotonic", Mock(side_effect=[0, 0, 1]))
    monkeypatch.setattr(browser.stopped, "wait", Mock(return_value=False))

    assert browser.open_when_ready() is listener_owned

    if listener_owned:
        request.assert_called_once()
        browser_open.assert_called_once_with(browser.url)
    else:
        request.assert_not_called()
        browser_open.assert_not_called()
