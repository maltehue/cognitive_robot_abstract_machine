"""
Self-contained recorded tours and local-only HTTP presentation.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from socketserver import ThreadingTCPServer
from unittest.mock import MagicMock, Mock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from cramera import offline
from cramera.offline import OfflineTour, OfflineHandler, OfflineOptions, OfflineBrowser
from cramera.onboard.scene_index import InvalidSceneName

# %% tour bundles


def test_prepare_copies_recordings_out_of_temporary_storage(
    fixture_scene: Path,
) -> None:
    """
    A prepared tour remains valid without its source recordings.

    :param fixture_scene: Existing miniature recorded scene.
    """
    source = fixture_scene / "scenes"
    destination = fixture_scene / "presentation"
    storyboard = Path(__file__).parent / "dataset" / "offline_storyboard.json"
    tour = OfflineTour.prepare(source, destination, storyboard)
    assert tour.directory == destination
    assert tour.scene_names == ["fixture"]
    assert (destination / "scenes" / "fixture" / "robot.urdf").read_bytes() == (
        source / "fixture" / "robot.urdf"
    ).read_bytes()
    (source / "fixture" / "robot.urdf").unlink()
    tour.validate()
    manifest = json.loads(tour.storyboard_path.read_text())
    assert manifest["chapters"][0]["scene"] == "fixture"


def test_prepare_never_replaces_an_existing_tour(fixture_scene: Path) -> None:
    """
    Existing presentation files remain untouched during repeated export.

    :param fixture_scene: Existing miniature recorded scene.
    """
    destination = fixture_scene / "presentation"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        OfflineTour.prepare(
            fixture_scene / "scenes",
            destination,
            Path(__file__).parent / "dataset" / "offline_storyboard.json",
        )
    assert sentinel.read_text() == "keep"


def test_offline_options_preserve_isolation_and_bundle_path(tmp_path: Path) -> None:
    """
    The documented command accepts a bundle and isolated network mode.

    :param tmp_path: Independent presentation directory.
    """
    options = OfflineOptions.parse(
        [str(tmp_path), "--isolated", "--no-browser", "--port", "8713"]
    )
    assert options.directory == tmp_path
    assert options.isolated is True
    assert options.open_browser is False
    assert options.port == 8713


@pytest.mark.parametrize(
    "arguments",
    [["--port", "0"], ["--port", "65536"], ["--isolated", "--listen-fd", "4"]],
)
def test_invalid_launch_options_are_rejected(
    tmp_path: Path, arguments: list[str]
) -> None:
    """
    Invalid ports and ambiguous isolation ownership fail before serving.

    :param tmp_path: Independent presentation directory.
    :param arguments: Conflicting or out-of-range command options.
    """
    with pytest.raises(SystemExit) as error:
        OfflineOptions.parse([str(tmp_path), *arguments])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("view", "unknown"),
        ("speed", 0),
        ("durationSeconds", -1),
        ("scene", "../elsewhere"),
    ],
)
def test_invalid_chapters_fail_before_copying(
    fixture_scene: Path, field: str, value: object
) -> None:
    """
    Invalid presentation actions do not leave a partial export.

    :param fixture_scene: Existing miniature recording.
    :param field: Chapter option made invalid.
    :param value: Unsupported value to reject.
    """
    payload = json.loads(
        (Path(__file__).parent / "dataset" / "offline_storyboard.json").read_text()
    )
    payload["chapters"][0][field] = value
    manifest = fixture_scene / "invalid.json"
    manifest.write_text(json.dumps(payload))
    destination = fixture_scene / "invalid-export"
    with pytest.raises((ValueError, InvalidSceneName)):
        OfflineTour.prepare(fixture_scene / "scenes", destination, manifest)
    assert not destination.exists()


# %% browser isolation


def test_offline_headers_allow_local_assets_and_block_external_connections(
    fixture_scene: Path,
) -> None:
    """
    Every response restricts the presentation to its own local origin.

    :param fixture_scene: Existing miniature scene and knowledge sources.
    """
    with ThreadingTCPServer(("127.0.0.1", 0), OfflineHandler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        with urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/index.html"
        ) as response:
            policy = response.headers["Content-Security-Policy"]
            assert "connect-src 'self'" in policy
            assert "default-src 'self'" in policy
            assert "https:" not in policy
            assert response.headers["Permissions-Policy"] == "microphone=(), camera=()"
        server.shutdown()
        worker.join()


def test_offline_viewer_rejects_execution_requests(fixture_scene: Path) -> None:
    """
    A presentation server cannot start a robot plan.

    :param fixture_scene: Existing miniature recorded scene.
    """
    with ThreadingTCPServer(("127.0.0.1", 0), OfflineHandler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/plan/run",
            data=b"{}",
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 405
        server.shutdown()
        worker.join()


def test_browser_waits_for_http_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An open socket without a responding viewer does not open a broken page.

    :param monkeypatch: Isolate HTTP readiness and the desktop browser.
    """
    response = MagicMock()
    response.__enter__.return_value.status = 200
    request = Mock(side_effect=[URLError("starting"), response])
    browser_open = Mock()
    monkeypatch.setattr(offline, "urlopen", request)
    monkeypatch.setattr(offline.webbrowser, "open", browser_open)
    browser = OfflineBrowser("http://localhost:8713/tour.html")
    assert browser.open_when_ready()
    assert request.call_count == 2
    browser_open.assert_called_once_with(browser.url)


def test_stopped_startup_never_opens_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A failed or cancelled launch cannot later open an unavailable page.

    :param monkeypatch: Isolate desktop browser operations.
    """
    browser_open = Mock()
    monkeypatch.setattr(offline.webbrowser, "open", browser_open)
    browser = OfflineBrowser("http://localhost:8713/tour.html")
    browser.stopped.set()
    assert not browser.open_when_ready()
    browser_open.assert_not_called()
