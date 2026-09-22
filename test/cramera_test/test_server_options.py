"""
Command-line feedback for starting the viewer.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from cramera import server
from cramera.server import DEFAULT_PORT, parse_arguments


# %% command-line validation
class TestViewerCommandLine:
    """
    Help and invalid arguments terminate before starting the server.
    """

    def test_help_exits_successfully(self, capsys: pytest.CaptureFixture[str]) -> None:
        """
        The conventional help flag displays usage without a traceback.
        """
        with pytest.raises(SystemExit) as result:
            parse_arguments(["--help"])
        assert result.value.code == 0
        assert str(DEFAULT_PORT) in capsys.readouterr().out

    @pytest.mark.parametrize(
        "arguments", [["--unknown"], ["invalid"], ["8123", "8124"], ["-1"], ["65536"]]
    )
    def test_invalid_arguments_are_usage_errors(self, arguments: list[str]) -> None:
        """
        Unknown flags, extra arguments and invalid ports are rejected.
        """
        with pytest.raises(SystemExit) as result:
            parse_arguments(arguments)
        assert result.value.code == 2

    @pytest.mark.parametrize("port", [0, 1, 65535])
    def test_valid_ports_are_preserved(self, port: int) -> None:
        """
        Explicit ports include the ephemeral port used for local testing.
        """
        assert parse_arguments([str(port), "--no-browser"]).port == port

    def test_ephemeral_port_opens_the_bound_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The browser receives the actual listening port when zero is requested.
        """
        with server.make_server(0) as http_server:
            browser_open = Mock()
            monkeypatch.setattr(server, "EQL_AVAILABLE", False)
            monkeypatch.setattr(server, "make_server", Mock(return_value=http_server))
            monkeypatch.setattr(server.webbrowser, "open", browser_open)
            monkeypatch.setattr(
                http_server, "serve_forever", Mock(side_effect=KeyboardInterrupt)
            )
            server.main(["0"])
            browser_open.assert_called_once_with(
                f"http://localhost:{http_server.server_address[1]}/"
            )
