"""
Starting the live viewer: as a library call or as the demo wrapper CLI.
"""

from __future__ import annotations

import logging
import os
import runpy
import signal
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import ClassVar, TYPE_CHECKING

from cramera.live.http import DEFAULT_PORT
from cramera.logging_setup import get_logger
from cramera.server import DEFAULT_PORT as VIEWER_DEFAULT_PORT, ServerOptions

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

    from cramera.live.visualization import LiveVisualization

logger = get_logger(__name__)

VISUALIZATION_BACKEND_VARIABLE = "CORAPLEX_VISUALIZATION"
"""
The coraplex environment variable selecting the visualization backend.
"""


# %% demo bring-up
class RunnerArgument(StrEnum):
    """Arguments accepted by the live-demo launcher."""

    DEMO = "demo"
    """Python demonstration to execute."""
    VIEWER = "--viewer"
    """Start a browser viewer alongside the demonstration."""
    VIEWER_PORT = "--viewer-port"
    """HTTP port for the accompanying viewer."""


@dataclass(frozen=True)
class RunnerOptions:
    """A demonstration and its optional local browser viewer."""

    demo: Path
    """Python demonstration file."""
    viewer: bool = False
    """Whether this launcher owns a browser viewer process."""
    viewer_port: int = VIEWER_DEFAULT_PORT
    """HTTP port for the browser viewer."""

    @classmethod
    def parse(cls, arguments: list[str]) -> RunnerOptions:
        """Parse the live-demo command line.

        :param arguments: Arguments without the executable name.
        :return: The selected demonstration and viewer options.
        """
        parser = ArgumentParser(
            description="Run a CRAM demo with a live browser bridge."
        )
        parser.add_argument(
            RunnerArgument.DEMO, type=Path, help="Python demonstration file"
        )
        parser.add_argument(
            RunnerArgument.VIEWER,
            action="store_true",
            help="Start the viewer and open its browser page",
        )
        parser.add_argument(
            RunnerArgument.VIEWER_PORT,
            type=int,
            default=VIEWER_DEFAULT_PORT,
            help=f"Viewer HTTP port (default: {VIEWER_DEFAULT_PORT})",
        )
        options = parser.parse_args(arguments)
        if not 0 <= options.viewer_port <= ServerOptions.MAX_PORT:
            parser.error(f"viewer port must be between 0 and {ServerOptions.MAX_PORT}")
        return cls(
            demo=options.demo, viewer=options.viewer, viewer_port=options.viewer_port
        )


@dataclass
class ViewerProcess:
    """The browser server owned by one live-demo launcher."""

    SHUTDOWN_TIMEOUT: ClassVar[int] = 5
    """Seconds allowed for the viewer to stop before forced cleanup."""
    port: int
    """Port passed to the existing viewer command."""
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    """The viewer child process, when started."""

    def start(self) -> None:
        """Start the existing viewer command with the active Python environment."""
        self.process = subprocess.Popen(
            [sys.executable, "-m", "cramera.server", str(self.port)]
        )

    def stop(self) -> None:
        """Stop and reap the viewer child, if it is still running."""
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=self.SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


# %% visualization attachment
def start(world: World, port: int = DEFAULT_PORT) -> LiveVisualization:
    """
    Serve a world to the browser viewer.

    Prefer :func:`coraplex.testing.start_visualization` (or
    :class:`coraplex.visualization.WorldVisualization` directly) inside demos — this is
    the cramera-side entry point they delegate to.

    :param world: The world to serve.
    :param port: Port of the bridge's HTTP endpoints.
    :return: The started visualization.
    """
    from cramera.live.visualization import LiveVisualization

    return LiveVisualization(world=world, port=port).start()


def main(arguments: list[str] | None = None) -> None:
    """
    ``cramera-live path/to/demo.py`` — run a demo with the browser viewer.

    Selects the cramera backend through ``CORAPLEX_VISUALIZATION`` and runs the demo
    unchanged: the demo's own ``start_visualization`` call picks the backend up. The
    demo's directory is put on ``sys.path`` so it can import its local helper modules,
    exactly as if it were started directly. After the demo finishes the bridge stays
    up for inspection until Ctrl-C.

    :param arguments: Command-line arguments, or None to read the process arguments.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    options = RunnerOptions.parse(sys.argv[1:] if arguments is None else arguments)
    demo = options.demo.resolve()
    os.environ.setdefault(VISUALIZATION_BACKEND_VARIABLE, "cramera")
    sys.path.insert(0, str(demo.parent))
    viewer = ViewerProcess(port=options.viewer_port) if options.viewer else None
    original_arguments = sys.argv
    try:
        if viewer is not None:
            viewer.start()
        logger.info("running demo: %s", demo)
        sys.argv = [str(demo)]
        runpy.run_path(str(demo), run_name="__main__")
        logger.info("demo finished — bridge stays up for inspection (Ctrl-C to quit)")
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        sys.argv = original_arguments
        if viewer is not None:
            viewer.stop()


if __name__ == "__main__":
    main()
