"""Prepare and present recorded CRAM stories without a running robot process."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socketserver
import sys
import threading
import time
import webbrowser
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPMethod, HTTPStatus
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from typing_extensions import Any, ClassVar

from cramera import paths
from cramera.offline_assets import OfflineSceneAssets
from cramera.offline_network import (
    OfflineNetworkArgument,
    OfflineNetworkLauncher,
    server_from_listener,
)
from cramera.onboard.scene_index import validate_scene_name, write_scene_index
from cramera.server import NO_BROWSER_FLAG, Handler, ServerOptions

# %% presentation data


class TourFile(StrEnum):
    """Files and directories defining a recorded presentation."""

    SCENES = "scenes"
    """Scene bundles copied into the presentation."""
    TOUR = "offline_tour"
    """Directory holding the presentation's chapter descriptions."""
    STORYBOARD = "storyboard.json"
    """Chapter manifest consumed by the viewer."""
    INDEX = "index.json"
    """Existing CRAMERA scene index."""
    TEMPLATE = "offline_storyboard.json"
    """Packaged reference story for the validated recordings."""


class TourField(StrEnum):
    """Fields in the authored presentation manifest."""

    CHAPTERS = "chapters"
    """Ordered chapters forming the complete presentation."""
    TITLE = "title"
    """Short chapter heading."""
    NARRATION = "narration"
    """Presenter notes accompanying the recording."""
    SCENE = "scene"
    """Recorded scene selected for the chapter."""
    VIEW = "view"
    """Graph panel shown alongside the scene."""
    QUERY = "query"
    """Optional recorded-world question."""
    PLAYBACK = "playback"
    """Whether recorded trajectory playback starts."""
    SPEED = "speed"
    """Recorded trajectory playback multiplier."""
    DURATION = "durationSeconds"
    """Minimum chapter duration in seconds."""


class TourView(StrEnum):
    """Existing graph panels available in a recorded presentation."""

    KNOWLEDGE = "knowledge"
    """Objects and relationships in the recorded world."""
    PLAN = "plan"
    """Recorded plan hierarchy."""
    STATECHART = "statechart"
    """Recorded controller statechart."""


@dataclass(frozen=True)
class TourChapter:
    """A recorded scene and the presentation actions accompanying it."""

    PLAYBACK_SPEEDS: ClassVar[frozenset[float]] = frozenset({0.5, 1.0, 2.0, 4.0})
    """Playback multipliers supported by the presentation controls."""
    MAX_DURATION_SECONDS: ClassVar[float] = 3600.0
    """Exclusive upper bound on the duration of a single chapter."""

    title: str
    """Short chapter heading."""
    narration: str
    """Presenter notes displayed beside the recording."""
    scene: str
    """Saved scene selected for this chapter."""
    view: str
    """Existing graph panel to display."""
    query: str | None
    """Optional recorded-world question."""
    playback: bool
    """Whether the chapter plays the recorded trajectory."""
    speed: float
    """Playback speed multiplier."""
    duration_seconds: float
    """Minimum chapter duration during automatic presentation."""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TourChapter:
        """Read one authored chapter and validate its scene and timing.

        :param payload: Chapter object from the presentation manifest.
        :return: Validated chapter.
        """
        chapter = cls(
            title=payload[TourField.TITLE],
            narration=payload[TourField.NARRATION],
            scene=validate_scene_name(payload[TourField.SCENE]),
            view=payload[TourField.VIEW],
            query=payload.get(TourField.QUERY),
            playback=payload[TourField.PLAYBACK],
            speed=float(payload[TourField.SPEED]),
            duration_seconds=float(payload[TourField.DURATION]),
        )
        if chapter.view not in set(TourView):
            raise ValueError(f"Unknown presentation view: {chapter.view}")
        if (
            chapter.speed not in cls.PLAYBACK_SPEEDS
            or not 0 < chapter.duration_seconds < cls.MAX_DURATION_SECONDS
        ):
            raise ValueError("Chapter speed or duration is outside the supported range")
        return chapter


@dataclass(frozen=True)
class OfflineTour:
    """A local presentation with independent copies of every recorded scene."""

    ENCODING: ClassVar[str] = "utf-8"
    """Text encoding of the authored JSON storyboard."""

    directory: Path
    """Root containing the copied scenes and storyboard."""

    @property
    def storyboard_path(self) -> Path:
        """Return the manifest location served by the existing scene-file endpoint."""
        return self.directory / TourFile.SCENES / TourFile.TOUR / TourFile.STORYBOARD

    @property
    def chapters(self) -> list[TourChapter]:
        """Read the nonempty list of presentation chapters."""
        payload = json.loads(self.storyboard_path.read_text(encoding=self.ENCODING))
        chapters = [
            TourChapter.from_payload(chapter) for chapter in payload[TourField.CHAPTERS]
        ]
        if not chapters:
            raise ValueError("A presentation must contain at least one chapter")
        return chapters

    @property
    def scene_names(self) -> list[str]:
        """Return referenced scene names in first-use order."""
        return list(dict.fromkeys(chapter.scene for chapter in self.chapters))

    def validate(self) -> None:
        """Verify that the complete story can load from local scene assets."""
        for name in self.scene_names:
            OfflineSceneAssets(self.directory / TourFile.SCENES / name).validate()

    @classmethod
    def prepare(cls, source: Path, destination: Path, storyboard: Path) -> OfflineTour:
        """Copy the referenced recordings without replacing an existing presentation.

        :param source: Directory containing saved CRAMERA scene bundles.
        :param destination: New presentation directory to create.
        :param storyboard: Authored chapter manifest.
        :return: Validated local presentation.
        """
        if destination.exists():
            raise FileExistsError(destination)
        payload = json.loads(storyboard.read_text(encoding=cls.ENCODING))
        chapters = [
            TourChapter.from_payload(chapter) for chapter in payload[TourField.CHAPTERS]
        ]
        if not chapters:
            raise ValueError("A presentation must contain at least one chapter")
        names = list(dict.fromkeys(chapter.scene for chapter in chapters))
        for name in names:
            OfflineSceneAssets(source / name).validate()
        tour = cls(destination)
        tour.storyboard_path.parent.mkdir(parents=True)
        shutil.copy2(storyboard, tour.storyboard_path)
        for name in names:
            shutil.copytree(source / name, destination / TourFile.SCENES / name)
        write_scene_index(destination / TourFile.SCENES / TourFile.INDEX, names[0])
        tour.validate()
        return tour


# %% local-only viewer


@dataclass
class OfflineBrowser:
    """Open a presentation only after its HTTP server can answer requests."""

    REQUEST_TIMEOUT_SECONDS: ClassVar[float] = 0.5
    """Maximum wait for each readiness response."""
    RETRY_INTERVAL_SECONDS: ClassVar[float] = 0.1
    """Delay between readiness attempts while the browser remains unopened."""

    url: str
    """Local presentation address to open."""
    timeout_seconds: float = 60.0
    """Maximum wait for the owned server to become ready."""
    stopped: threading.Event = field(default_factory=threading.Event)
    """Cancellation signal when server startup ends or fails."""
    start_signal: threading.Event | None = None
    """Optional confirmation that this launch owns the listening port."""

    def open_when_ready(self) -> bool:
        """Wait for HTTP readiness and then open the presentation page.

        :return: Whether the browser was asked to open the ready presentation.
        """
        deadline = time.monotonic() + self.timeout_seconds
        while not self.stopped.is_set() and time.monotonic() < deadline:
            if self.start_signal is not None and not self.start_signal.is_set():
                self.stopped.wait(self.RETRY_INTERVAL_SECONDS)
                continue
            try:
                with urlopen(
                    Request(self.url, method=HTTPMethod.HEAD),
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                ) as response:
                    ready = response.status == HTTPStatus.OK
            except (URLError, TimeoutError, OSError):
                ready = False
            if ready and not self.stopped.is_set():
                webbrowser.open(self.url)
                return True
            self.stopped.wait(self.RETRY_INTERVAL_SECONDS)
        return False


class OfflineHeader(StrEnum):
    """Browser restriction headers attached to recorded presentation responses."""

    CONTENT_POLICY = "Content-Security-Policy"
    """Allowed browser resource origins and resource types."""
    PERMISSIONS_POLICY = "Permissions-Policy"
    """Available browser hardware capabilities."""


class RecordedQueryEndpoint(StrEnum):
    """Read-only query endpoints available during a recorded presentation."""

    EQL = "/api/eql"
    """Execute a recorded-world EQL query."""
    QUESTION = "/api/question"
    """Answer a supported recorded-world question."""


@dataclass(init=False)
class OfflineHandler(Handler):
    """Serve existing recording APIs with same-origin browser resource restrictions."""

    CONTENT_POLICY: ClassVar[str] = (
        "default-src 'self'; connect-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
        "font-src 'self' data:; worker-src 'self' blob:; frame-src 'self'; "
        "object-src 'none'; base-uri 'self'; form-action 'self'"
    )
    """Only local resources may be loaded by the presentation page."""
    PERMISSIONS_POLICY: ClassVar[str] = "microphone=(), camera=()"
    """Recorded playback requires no browser microphone or camera access."""

    def end_headers(self) -> None:
        """Restrict page resources and prevent browser microphone/camera use."""
        self.send_header(OfflineHeader.CONTENT_POLICY, self.CONTENT_POLICY)
        self.send_header(OfflineHeader.PERMISSIONS_POLICY, self.PERMISSIONS_POLICY)
        super().end_headers()

    def do_POST(self) -> None:
        """Allow recorded queries while refusing changes or new live executions."""
        if urlparse(self.path).path not in set(RecordedQueryEndpoint):
            self.send_error(
                HTTPStatus.METHOD_NOT_ALLOWED, "Recorded presentation is read-only"
            )
            return
        super().do_POST()


# %% command options
class OfflineArgument(StrEnum):
    """Options accepted by the recorded presentation command."""

    DIRECTORY = "directory"
    """Presentation bundle to prepare or serve."""
    PORT = "--port"
    """Local viewer TCP port."""
    ISOLATED = "--isolated"
    """Start the viewer in a network namespace without outbound routes."""
    PREPARE_FROM = "--prepare-from"
    """Directory containing recordings to export."""
    STORYBOARD = "--storyboard"
    """Authored chapter manifest for the exported presentation."""
    CHECK_ONLY = "--check-only"
    """Validate the bundle without starting a server."""


class ArgumentAction(StrEnum):
    """Boolean state changes requested by command-line switches."""

    ENABLE = "store_true"
    """Set a disabled option to true when its switch is present."""
    DISABLE = "store_false"
    """Set an enabled option to false when its switch is present."""


class OfflineEnvironment(StrEnum):
    """Existing CRAMERA paths redirected to the recorded presentation."""

    DATA = "CRAMERA_DATA"
    """Presentation data root."""
    SCENES = "CRAMERA_SCENES"
    """Directory containing the exported recordings."""
    SCENE = "CRAMERA_SCENE"
    """Initial recording selected by the presentation."""


@dataclass(frozen=True)
class OfflineOptions:
    """Command-line choices for preparation and recorded presentation."""

    DEFAULT_PORT: ClassVar[int] = 8713
    """Default local HTTP port for a recorded presentation."""
    URL_TEMPLATE: ClassVar[str] = "http://localhost:{port}/tour.html"
    """Local presentation page parameterized by the viewer port."""
    OPEN_BROWSER_DESTINATION: ClassVar[str] = "open_browser"
    """Parser attribute controlled by the existing no-browser switch."""
    MODULE_FLAG: ClassVar[str] = "-m"
    """Python option selecting an importable module as the child entry point."""
    MODULE: ClassVar[str] = "cramera.offline"
    """Importable presentation command passed to an isolated child."""
    LOG_FORMAT: ClassVar[str] = "%(message)s"
    """Console output format for the presentation command."""

    directory: Path
    """Presentation bundle to prepare or serve."""
    port: int
    """Local viewer port."""
    open_browser: bool
    """Whether to open the presentation in the user's default browser."""
    isolated: bool
    """Whether to deny outgoing network through a Linux network namespace."""
    listen_descriptor: int | None
    """Inherited localhost listener when already inside the isolated process."""
    prepare_from: Path | None
    """Source recordings directory for an optional initial export."""
    storyboard: Path
    """Manifest used when preparing a new presentation."""
    check_only: bool
    """Whether to validate the bundle and exit without serving it."""

    @classmethod
    def parse(cls, arguments: list[str]) -> OfflineOptions:
        """Parse presentation options and reject invalid port combinations.

        :param arguments: Command-line arguments without the executable name.
        :return: Options for one recorded presentation.
        """
        parser = ArgumentParser(description=__doc__)
        parser.add_argument(OfflineArgument.DIRECTORY, type=Path)
        parser.add_argument(OfflineArgument.PORT, type=int, default=cls.DEFAULT_PORT)
        parser.add_argument(
            NO_BROWSER_FLAG,
            dest=cls.OPEN_BROWSER_DESTINATION,
            action=ArgumentAction.DISABLE,
        )
        parser.add_argument(OfflineArgument.ISOLATED, action=ArgumentAction.ENABLE)
        parser.add_argument(OfflineNetworkArgument.LISTEN_FD, type=int)
        parser.add_argument(OfflineArgument.PREPARE_FROM, type=Path)
        parser.add_argument(
            OfflineArgument.STORYBOARD,
            type=Path,
            default=paths.WEB_ROOT / TourFile.TEMPLATE,
        )
        parser.add_argument(OfflineArgument.CHECK_ONLY, action=ArgumentAction.ENABLE)
        options = parser.parse_args(arguments)
        if not 1 <= options.port <= ServerOptions.MAX_PORT:
            parser.error("port must be between 1 and 65535")
        if options.isolated and options.listen_fd is not None:
            parser.error("--isolated and --listen-fd cannot be combined")
        return cls(
            options.directory,
            options.port,
            options.open_browser,
            options.isolated,
            options.listen_fd,
            options.prepare_from,
            options.storyboard,
            options.check_only,
        )


def main(arguments: list[str] | None = None) -> None:
    """Validate and serve a recorded presentation with one local command.

    :param arguments: Command-line arguments, or None to use the process arguments.
    """
    logging.basicConfig(
        level=logging.INFO, format=OfflineOptions.LOG_FORMAT, force=True
    )
    options = OfflineOptions.parse(sys.argv[1:] if arguments is None else arguments)
    tour = (
        OfflineTour.prepare(options.prepare_from, options.directory, options.storyboard)
        if options.prepare_from is not None
        else OfflineTour(options.directory)
    )
    tour.validate()
    if options.check_only:
        print(
            f"Validated {len(tour.chapters)} chapters and {len(tour.scene_names)} local scenes."
        )
        return
    url = options.URL_TEMPLATE.format(port=options.port)
    if options.isolated:
        command = [
            sys.executable,
            OfflineOptions.MODULE_FLAG,
            OfflineOptions.MODULE,
            str(tour.directory.resolve()),
            OfflineArgument.PORT,
            str(options.port),
            NO_BROWSER_FLAG,
        ]
        launcher = OfflineNetworkLauncher(options.port, command)
        browser = OfflineBrowser(url, start_signal=launcher.started)
        if options.open_browser:
            threading.Thread(target=browser.open_when_ready, daemon=True).start()
        try:
            raise SystemExit(launcher.run())
        finally:
            browser.stopped.set()
    os.environ[OfflineEnvironment.DATA] = str(tour.directory.resolve())
    os.environ[OfflineEnvironment.SCENES] = str(
        tour.directory.resolve() / TourFile.SCENES
    )
    os.environ[OfflineEnvironment.SCENE] = tour.scene_names[0]
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = (
        server_from_listener(options.listen_descriptor, OfflineHandler)
        if options.listen_descriptor is not None
        else socketserver.ThreadingTCPServer(
            (OfflineNetworkLauncher.LOOPBACK_ADDRESS, options.port), OfflineHandler
        )
    )
    with server:
        print(f"Recorded presentation ready: {url}", flush=True)
        if options.open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
