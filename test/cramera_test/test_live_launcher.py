"""
One-command bring-up and cleanup of the browser viewer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from cramera.live import runner


# %% launcher options
class TestLiveLaunchOptions:
    """
    Viewer startup is explicit and preserves the ordinary live command.
    """

    def test_a_demo_alone_does_not_start_another_viewer(self) -> None:
        options = runner.RunnerOptions.parse(["demo.py"])
        assert options.demo == Path("demo.py")
        assert options.viewer is False

    def test_viewer_and_custom_port_are_selected(self) -> None:
        options = runner.RunnerOptions.parse(
            ["--viewer", "--viewer-port", "8123", "demo.py"]
        )
        assert options.viewer is True
        assert options.viewer_port == 8123

    def test_help_is_available_without_a_demo(self) -> None:
        with pytest.raises(SystemExit) as result:
            runner.RunnerOptions.parse(["--help"])
        assert result.value.code == 0

    @pytest.mark.parametrize("port", ["-1", "65536"])
    def test_invalid_ports_are_rejected(self, port: str) -> None:
        with pytest.raises(SystemExit) as result:
            runner.RunnerOptions.parse(["--viewer-port", port, "demo.py"])
        assert result.value.code == 2


# %% child process ownership
class TestViewerProcess:
    """
    A launcher stops only the viewer it started.
    """

    def test_start_uses_the_active_interpreter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create_process = Mock()
        monkeypatch.setattr(runner.subprocess, "Popen", create_process)
        viewer = runner.ViewerProcess(port=8123)
        viewer.start()
        assert create_process.call_args.args[0] == [
            runner.sys.executable,
            "-m",
            "cramera.server",
            "8123",
        ]

    def test_stop_terminates_its_running_viewer(self) -> None:
        process = Mock()
        process.poll.return_value = None
        viewer = runner.ViewerProcess(port=8123, process=process)
        viewer.stop()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=viewer.SHUTDOWN_TIMEOUT)

    def test_stop_leaves_an_already_finished_process_alone(self) -> None:
        process = Mock()
        process.poll.return_value = 0
        runner.ViewerProcess(port=8123, process=process).stop()
        process.terminate.assert_not_called()

    def test_stop_before_start_is_harmless(self) -> None:
        runner.ViewerProcess(port=8123).stop()

    def test_an_unresponsive_viewer_is_reaped(self) -> None:
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [runner.subprocess.TimeoutExpired("viewer", 5), 0]
        runner.ViewerProcess(port=8123, process=process).stop()
        process.kill.assert_called_once_with()

    def test_viewer_stops_when_the_demo_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        viewer = Mock()
        monkeypatch.setattr(runner, "ViewerProcess", Mock(return_value=viewer))
        monkeypatch.setattr(
            runner.runpy, "run_path", Mock(side_effect=RuntimeError("demo failed"))
        )
        with pytest.raises(RuntimeError):
            runner.main(["--viewer", "demo.py"])
        viewer.start.assert_called_once_with()
        viewer.stop.assert_called_once_with()

    def test_viewer_stops_after_normal_inspection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        viewer = Mock()
        monkeypatch.setattr(runner, "ViewerProcess", Mock(return_value=viewer))
        monkeypatch.setattr(runner.runpy, "run_path", Mock())
        monkeypatch.setattr(runner.signal, "pause", Mock())
        runner.main(["--viewer", "demo.py"])
        viewer.stop.assert_called_once_with()

    def test_launcher_flags_are_not_passed_to_the_demo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A demo's own argument parser sees only its program name.
        """
        monkeypatch.setattr(runner, "ViewerProcess", Mock())
        observed_arguments = []
        monkeypatch.setattr(
            runner.runpy,
            "run_path",
            lambda *arguments, **keywords: observed_arguments.extend(runner.sys.argv),
        )
        monkeypatch.setattr(runner.signal, "pause", Mock())
        runner.main(["--viewer", "demo.py"])
        assert observed_arguments == [str(Path("demo.py").resolve())]
