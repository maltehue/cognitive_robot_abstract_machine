"""
The cramera HTTP server: static frontend + JSON API.

Serves three things from one port (default 8711):

  * the packaged web frontend (``cramera/web`` — panels, vendored libs)
  * scene bundles from :func:`cramera.paths.scenes_directory` under ``/scenes/``
  * the JSON API the panels talk to:

      GET  /api/knowledge              the knowledge-graph overview payload
      GET  /api/knowledge/view?name=   one graph tab (knowledge/kinematics/plan/chart)
      GET  /api/knowledge/expand?node= drill-down subgraph for one node
      POST /api/eql             run an EQL query string
      GET  /api/recording/status       {state: finalized|idle} of the on-disk
                                        __recording__ bundle (see
                                        cramera.live.recording_storage) — a pure
                                        filesystem check, so it answers correctly even
                                        once the demo process that made it has exited
      POST /api/recording/save         {name, destination?, firstFrame?, lastFrame?}
                                        -> promote that
                                        bundle to a permanent, locally saved scene,
                                        trimmed to the given inclusive frame range
                                        when one is sent
      POST /api/recording/discard      drop that bundle

The API needs krrood (EQL). Without it the server still serves the viewer and
answers API calls with ``{"ok": false, "error": ...}`` so the frontend can say
why the knowledge panel is empty.

    cramera            # console script
    python -m cramera.server [port]
"""

from __future__ import annotations

import http.server
import json
import logging
import mimetypes
import socketserver
import sys
import threading
import traceback
import webbrowser
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Any, Callable, ClassVar, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from cramera import paths
from cramera.live.frame_range import InvalidFrameRange
from cramera.live.recording_save_request import (
    InvalidRecordingSaveRequest,
    RecordingSaveRequest,
)
from cramera.live.recording_storage import (
    NoSavedRecording,
    SceneNameTaken,
    SharedScenesUnavailable,
    discard_recording_bundle,
    has_saveable_recording,
)
from cramera.logging_setup import get_logger
from cramera.models_workbench import (
    ModelWorkbench,
    NO_MODELS_MESSAGE,
    PROBABILISTIC_MODELS_AVAILABLE,
)
from cramera.onboard.scene_index import InvalidSceneName, merged_scene_index
from cramera.payload import CrameraPayload

logger = get_logger(__name__)

DEFAULT_PORT = 8711

try:
    import krrood  # noqa: F401  (the EQL engine)

    from cramera.model_catalog import ModelCatalog
    from cramera.knowledge.eql_session import EqlSession
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
    from cramera.knowledge.presets import Preset
    from cramera.knowledge.scene_bundle import SceneBundle
    from cramera.knowledge.question_matching import QuestionMatcher
    from cramera.knowledge.views.dispatcher import GraphPanelViews

    EQL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    EQL_AVAILABLE = False
    logger.warning("krrood not importable — serving the viewer without the EQL API")
except (
    Exception
):  # pragma: no cover - a broken knowledge base should not kill the viewer
    EQL_AVAILABLE = False
    traceback.print_exc()


_EQL_LOCK = threading.Lock()
"""
Krrood's SymbolGraph singleton is not threadsafe; queries are serialized.
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    """
    Static files from the packaged web root, plus the JSON API routes.
    """

    NO_EQL_MESSAGE: ClassVar[str] = "krrood/EQL not available in this environment"
    """
    What every API route answers with when krrood is not importable.
    """

    def __init__(self, *args, **kwargs):
        """
        Delegate to the base handler, serving from the packaged web root.

        :param args: Positional arguments forwarded to the base handler.
        :param kwargs: Keyword arguments forwarded to the base handler.
        """
        super().__init__(*args, directory=str(paths.WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        """
        Forbid caching so a rebuilt scene/frontend is never served stale.

        ..note:: ``no-cache`` would not do: it lets a browser keep its copy and
            revalidate, and the only validator this handler offers is the file's
            modification time, which an edit does not always push past the date the
            stored copy carries.
        """
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        """
        Route the per-request access log through logging.

        The page polls for the live scene while no demo runs; those misses would flood
        the console every second and are not news, so they stay out of the log.

        :param format:``printf``-style log message format.
        :param args: Values to interpolate into ``format``.
        """
        message = format % args if args else format
        if paths.LIVE_SCENE_NAME in message and " 404 " in message:
            return
        logger.info("  %s", message)

    # %% helpers
    def _send_json(self, payload: Any, code: int = 200) -> None:
        """
        Send a payload as JSON with the given status code.

        ``payload`` may be a plain JSON-able dict or a :class:`CrameraPayload` — either
        is serialized the same way.

        :param payload: The payload to serialize and send.
        :param code: HTTP status code to respond with.
        """
        if isinstance(payload, CrameraPayload):
            payload = payload.to_payload()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query_parameters(self) -> Dict[str, List[str]]:
        """
        The parsed query-string parameters of the current request.
        """
        return parse_qs(urlparse(self.path).query)

    def _requested_scene(self) -> Optional[str]:
        """
        The requested scene, or the active recording available in the viewer's index.

        The frontend switches scenes by reloading with a ``?scene=`` parameter, so every
        API route has to honour it or the panels would disagree about what is on screen.
        """
        requested = self._query_parameters().get("scene")
        return (
            requested[0] if requested else SceneBundle.active_name(merged_scene_index())
        )

    def _guarded(self, handler: Callable[[], Any]) -> None:
        """
        Run an API handler; report exceptions as a JSON error payload.

        :param handler: The handler to run, returning the payload to send on success.
        """
        if not EQL_AVAILABLE:
            return self._send_error(self.NO_EQL_MESSAGE)
        try:
            return self._send_json(handler())
        except Exception as error:
            return self._send_exception(error)

    # %% scene bundles (generated data, lives outside the package)
    def _serve_scene_file(self, url_path: str) -> None:
        """
        Serve one file of a scene bundle, with path-traversal protection.

        ``index.json`` is special-cased: the viewer's pickers must see both the shared
        scenes root and the local-only recordings root (see
        :func:`cramera.onboard.scene_index.merged_scene_index`) as one list. Every other
        file is looked up across both roots (local first), so a saved recording's
        ``scene.json``/``trajectory.json``/meshes serve exactly like any onboarded scene.

        :param url_path: The request path, starting with ``/scenes/``.
        """
        relative_path = url_path[len("/scenes/") :]
        if relative_path == "index.json":
            return self._send_json(merged_scene_index())
        for root in paths.scene_roots():
            base = root.resolve()
            target = (base / relative_path).resolve()
            if not target.is_relative_to(base):
                continue  # path traversal (".." or an injected absolute path)
            if target.is_file():
                return self._send_file(target)
        self.send_response(404)
        self.end_headers()

    def _send_file(self, target: Path) -> None:
        """
        Stream a resolved, existing file's bytes.

        :param target: The file to serve, already resolved and confirmed to exist.
        """
        content_type = (
            mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        )
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # %% routes
    def do_GET(self) -> None:
        """
        Route static files, scene bundles and the read-only API.
        """
        route = self.path.split("?")[0]
        if route.startswith("/scenes/"):
            return self._serve_scene_file(route)
        scene = (
            self._requested_scene()
            if EQL_AVAILABLE and route.startswith(("/api/knowledge", "/api/eql"))
            else None
        )
        if route == "/api/knowledge":
            return self._guarded(
                lambda: GraphPanelViews.of_scene(scene).for_tab("knowledge")
            )
        if route == "/api/knowledge/view":
            name = (self._query_parameters().get("name") or ["knowledge"])[0]
            return self._guarded(lambda: GraphPanelViews.of_scene(scene).for_tab(name))
        if route == "/api/knowledge/expand":
            node = (self._query_parameters().get("node") or [""])[0]
            return self._guarded(lambda: self._expanded_node(node, scene))
        if route == "/api/eql/vocabulary":
            return self._guarded(
                lambda: EqlSession.of_scene(scene).runner().vocabulary().to_payload()
            )
        if route == "/api/eql/members":
            name = (self._query_parameters().get("name") or [""])[0]
            return self._guarded(
                lambda: EqlSession.of_scene(scene)
                .runner()
                .vocabulary()
                .members_payload(name)
            )
        if route == "/api/models/state":
            if not PROBABILISTIC_MODELS_AVAILABLE:
                return self._send_error(NO_MODELS_MESSAGE)
            return self._guarded(lambda: ModelWorkbench.active().state())
        if route == "/api/recording/status":
            return self._send_json(
                {"state": "finalized" if has_saveable_recording() else "idle"}
            )
        if route == "/api/plan/catalog":
            return self._guarded(lambda: ModelCatalog.installed())
        if route == "/api/plan/scaffold/log":
            return self._scaffold_log()
        return super().do_GET()

    @staticmethod
    def _expanded_node(node: str, scene: Optional[str]) -> Any:
        """
        The node's subgraph, or a "not drillable" error if it has none.

        :param node: Id of the double-clicked node to expand.
        :param scene: Name of the scene the node belongs to, or None for the active one.
        """
        payload = GraphPanelViews.of_scene(scene).for_node(node)
        return payload if payload else {"ok": False, "error": "not drillable"}

    def do_POST(self) -> None:
        """
        Route the write-ish endpoints: EQL queries, asked questions and the models
        workbench.
        """
        route = self.path.split("?")[0]
        if route == "/api/eql":
            return self._run_eql()
        if route == "/api/question":
            return self._answer_asked_question()
        if route.startswith("/api/models/"):
            return self._run_models_request(route)
        if route == "/api/recording/save":
            return self._save_recording()
        if route == "/api/recording/discard":
            return self._discard_recording()
        if route == "/api/plan/save":
            return self._save_generated_plan()
        if route == "/api/plan/scaffold":
            return self._launch_scaffold()
        if route == "/api/plan/scaffold/stop":
            return self._stop_scaffold()
        return self._send_error("unknown endpoint", 404)

    def _generated_demos_directory(self):
        """
        Where the Plan Builder writes generated demos:
        ``coraplex/demos/coraplex_generated`` (overridable via
        ``CRAMERA_GENERATED_DEMOS``).

        Sits two levels under ``coraplex/`` so
        the generated ``os.path.dirname(__file__)/../../resources/objects`` mesh paths resolve.
        """
        import os
        from pathlib import Path

        override = os.environ.get("CRAMERA_GENERATED_DEMOS")
        if override:
            return Path(override)
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "coraplex" / "demos"
            if candidate.is_dir():
                return candidate / "coraplex_generated"
        return (
            here.parent / "generated_demos"
        )  # fallback: never resolves meshes, but writes

    _scaffold_proc = None  # class-level: the running Plan-Builder scaffold demo, if any
    _scaffold_log_path = None  # where that process's stdout+stderr are captured

    def _launch_scaffold(self) -> None:
        """
        Write a Plan-Builder scaffold demo (environment + objects, idle) and run it with
        cramera-live so its live world comes up on the bridge (:8765).

        The user drags objects in the Scene view; the builder captures their poses via
        /captured_objects.
        """
        import os
        import subprocess
        import sys

        body = self._request_body()
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            return self._send_json({"ok": False, "error": "empty code"}, 400)
        try:
            out_dir = self._generated_demos_directory()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "_builder_scaffold.py"
            path.write_text(code)
            repo = (
                out_dir.parent.parent.parent
            )  # coraplex_generated -> demos -> coraplex -> repo
            self._stop_scaffold(reply=False)
            env = dict(os.environ, CORAPLEX_VISUALIZATION="cramera")
            # capture stdout+stderr so the Plan Builder can show a traceback if the demo
            # fails, instead of the error vanishing into a detached process
            log_path = out_dir / "_builder_scaffold.log"
            type(self)._scaffold_log_path = log_path
            log_file = open(log_path, "wb")  # noqa: SIM115 (the child process owns it)
            # run with the SAME interpreter that runs this server (its venv/env), instead of
            # a hardcoded repo/.venv path — so a checkout with a differently placed or named
            # environment works without editing anything. `-m cramera.live.runner` is the
            # module behind the `cramera-live` console script.
            type(self)._scaffold_proc = subprocess.Popen(
                [sys.executable, "-m", "cramera.live.runner", str(path)],
                cwd=str(repo),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group, so stop can kill children too
            )
        except (OSError, ValueError) as error:
            return self._send_json({"ok": False, "error": str(error)}, 500)
        self._send_json({"ok": True, "path": str(path)})

    def _scaffold_log(self) -> None:
        """
        The running (or last) scaffold's captured output, and whether it is still alive.

        Lets the Plan Builder surface a demo's traceback instead of losing it to a
        detached process.
        """
        from pathlib import Path

        proc = type(self)._scaffold_proc
        log_path = type(self)._scaffold_log_path
        returncode = None if proc is None else proc.poll()
        text = ""
        if log_path is not None:
            try:
                text = Path(log_path).read_text(errors="replace")[-8000:]
            except OSError:
                text = ""
        self._send_json(
            {
                "ok": True,
                "alive": proc is not None and returncode is None,
                "returncode": returncode,
                "log": text,
            }
        )

    def _stop_scaffold(self, reply: bool = True) -> None:
        """
        Stop the running scaffold demo, if any.

        :param reply: Whether to send a JSON response (False for internal calls).
        """
        import os
        import signal
        import time

        proc = type(self)._scaffold_proc
        if proc is not None and proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(20):  # give it up to ~2s to exit, then SIGKILL the group
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        type(self)._scaffold_proc = None
        if reply:
            self._send_json({"ok": True})

    def _save_generated_plan(self) -> None:
        """
        Write a Plan-Builder-generated demo to the generated-demos directory and return
        its path (so the UI can show ``cramera-live <path>``).
        """
        import re
        from pathlib import Path

        body = self._request_body()
        name = str(body.get("name") or "").strip()
        code = body.get("code")
        if not name.endswith(".py"):
            name += ".py"
        if not re.match(r"^[A-Za-z0-9_\-]+\.py$", name):
            return self._send_json({"ok": False, "error": "invalid file name"}, 400)
        if not isinstance(code, str) or not code.strip():
            return self._send_json({"ok": False, "error": "empty code"}, 400)
        try:
            out_dir = self._generated_demos_directory()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / name
            path.write_text(code)
        except OSError as error:
            return self._send_json({"ok": False, "error": str(error)}, 500)
        self._send_json({"ok": True, "path": str(path)})

    def _save_recording(self) -> None:
        """
        Promote the on-disk ``__recording__`` bundle to a permanent, locally saved scene
        — independent of whether the demo process that made it is still running (see
        :mod:`cramera.live.recording_storage`).

        An optional ``firstFrame``/``lastFrame`` pair cuts the run down to that
        inclusive range before it is saved.
        """
        try:
            body = self._request_body()
            name = RecordingSaveRequest.from_json(body).save()
        except (
            InvalidSceneName,
            InvalidFrameRange,
            InvalidRecordingSaveRequest,
            json.JSONDecodeError,
        ) as error:
            return self._send_json({"ok": False, "error": str(error)}, 400)
        except (NoSavedRecording, SharedScenesUnavailable) as error:
            return self._send_json({"ok": False, "error": str(error)}, 400)
        except SceneNameTaken as error:
            return self._send_json({"ok": False, "error": str(error)}, 409)
        self._send_json({"ok": True, "scene": name})

    def _discard_recording(self) -> None:
        """
        Drop the on-disk ``__recording__`` bundle, if one exists.
        """
        discard_recording_bundle()
        self._send_json({"ok": True})

    def _run_eql(self) -> None:
        """
        Execute an EQL query.
        """
        if not EQL_AVAILABLE:
            return self._send_error(self.NO_EQL_MESSAGE)
        try:
            request_body = self._request_body()
            code = (request_body.get("code") or "").strip()
            if not code:
                return self._send_error("empty query")
            with _EQL_LOCK:
                session = EqlSession.of_scene(self._requested_scene())
                return self._send_json(session.run(code))
        except Exception as error:
            # a SyntaxError from the query is named by its own type, like any other
            return self._send_exception(error)

    def _answer_asked_question(self) -> None:
        """
        Match a natural-language question to the presets this scene can answer.

        Bundle-declared presets need a running demo, which the recorded scene does not
        have, so only the presets answerable here are on offer to match.
        """
        if not EQL_AVAILABLE:
            return self._send_error(self.NO_EQL_MESSAGE)
        try:
            text = (self._request_body().get("text") or "").strip()
            if not text:
                return self._send_error("empty question")
            with _EQL_LOCK:
                answerable = [
                    preset
                    for preset in Preset.of_scene(self._requested_scene())
                    if not preset.requires_live
                ]
                return self._send_json(QuestionMatcher(answerable).match(text))
        except Exception as error:
            return self._send_exception(error)

    def _run_models_request(self, route: str) -> None:
        """
        Answer one models-workbench request.

        :param route: The request path below ``/api/models/``.
        """
        if not PROBABILISTIC_MODELS_AVAILABLE:
            return self._send_error(NO_MODELS_MESSAGE)
        try:
            body = self._request_body()
            workbench = ModelWorkbench.active()
            if route == "/api/models/load":
                # ``model_text`` is the uploaded file verbatim: python's JSON reader
                # accepts the Infinity/NaN literals circuit files carry, which the
                # browser's own parser would reject
                model_data = body.get("model")
                if model_data is None:
                    model_data = json.loads(body.get("model_text") or "{}")
                return self._send_json(
                    workbench.load_model(model_data, name=body.get("name") or "")
                )
            if route == "/api/models/probability":
                return self._send_json(
                    workbench.probability(
                        body.get("query") or [], body.get("evidence") or []
                    )
                )
            if route == "/api/models/posterior":
                return self._send_json(
                    workbench.posterior(
                        body.get("variables") or [], body.get("evidence") or []
                    )
                )
            if route == "/api/models/mode":
                return self._send_json(workbench.mode(body.get("evidence") or []))
            return self._send_error("unknown endpoint", 404)
        except Exception as error:
            return self._send_exception(error)

    def _request_body(self) -> Dict[str, Any]:
        """
        The request's JSON body, or an empty mapping without one.
        """
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _send_error(self, message: str, code: int = 200) -> None:
        """
        Send an error payload; the panels render ``error`` as their message.

        :param message: What went wrong, in the panel's own words.
        :param code: HTTP status code to respond with.
        """
        self._send_json({"ok": False, "error": message}, code)

    def _send_exception(self, error: Exception) -> None:
        """
        Send an exception as an error payload, named by its type.

        :param error: The exception to report.
        """
        self._send_error("%s: %s" % (type(error).__name__, error))


def make_server(port: int = 0) -> socketserver.ThreadingTCPServer:
    """
    A ready-to-serve ThreadingTCPServer (port 0 = ephemeral, for tests).

    :param port: Port to listen on, or 0 for an ephemeral port.
    """
    socketserver.TCPServer.allow_reuse_address = True
    return socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)


NO_BROWSER_FLAG = "--no-browser"
"""
CLI flag that keeps the server from opening the viewer page on start.
"""


@dataclass(frozen=True)
class ServerOptions:
    """
    What the ``cramera`` command line asks for.
    """

    MAX_PORT: ClassVar[int] = 65535
    """
    Largest TCP port accepted by the server.
    """

    port: int = DEFAULT_PORT
    """
    Port the server listens on.
    """

    open_browser: bool = True
    """
    Whether the viewer page is opened in the default browser on start.
    """

    @classmethod
    def parse(cls, arguments: list[str]) -> ServerOptions:
        """
        Read viewer options and report invalid input with command-line usage.

        :param arguments: Arguments without the executable name.
        :return: Validated viewer options.
        """
        parser = ArgumentParser(
            description="Serve the CRAMERA viewer and local recordings."
        )
        parser.add_argument(
            "port",
            nargs="?",
            type=int,
            default=DEFAULT_PORT,
            help=f"HTTP port (default: {DEFAULT_PORT})",
        )
        parser.add_argument(
            NO_BROWSER_FLAG,
            dest="open_browser",
            action="store_false",
            help="Start without opening the browser",
        )
        options = parser.parse_args(arguments)
        if not 0 <= options.port <= cls.MAX_PORT:
            parser.error("port must be between 0 and 65535")
        return cls(port=options.port, open_browser=options.open_browser)


def parse_arguments(arguments: List[str]) -> ServerOptions:
    """
    Read the ``cramera`` command line: an optional port and ``--no-browser``.

    :param arguments: The command-line arguments, without the program name.
    """
    return ServerOptions.parse(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    """
    ``cramera`` — serve the viewer, the scenes and the JSON API.

    Opens the viewer page in the default browser once the server is up; demos only ever
    connect to it, so this is the one deliberate moment a page appears. Pass ``--no-
    browser`` to skip it (a headless or remote server).

    :param arguments: Command-line arguments, or None to use ``sys.argv``.
    """
    # force: an imported CRAM package may already have configured the root logger,
    # which would otherwise make this call a no-op and swallow the startup output
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    options = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    port = options.port
    if EQL_AVAILABLE:  # build the knowledge base once, before the first query
        EpisodeKnowledgeBase.of_scene(SceneBundle.active_name(merged_scene_index()))
    with make_server(port) as server:
        if port == 0:
            port = server.server_address[1]
        eql = "EQL ready (krrood)" if EQL_AVAILABLE else "EQL unavailable — static only"
        scenes = paths.scenes_directory()
        logger.info("cramera running at http://localhost:%d/ (%s)", port, eql)
        logger.info(
            "scene bundles: %s%s",
            scenes,
            "" if Path(scenes).is_dir() else "  (missing — run cramera-onboard)",
        )
        if options.open_browser:
            webbrowser.open("http://localhost:%d/" % port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
