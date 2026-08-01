from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps

from typing_extensions import Any, Callable, Dict

from krrood.exceptions import DataclassException

logger = logging.getLogger("coraplex_mcp")

# %% envelope


@dataclass
class ToolError:
    """
    The structured description of a tool failure returned to the client.
    """

    type: str
    """
    The error class name, so the client can branch on the kind of failure.
    """

    message: str
    """
    The human-readable description of what went wrong.
    """

    suggestion: str
    """
    Advice on how to recover, or an empty string when there is none.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: A JSON-serializable view of the error.
        """
        return {
            "type": self.type,
            "message": self.message,
            "suggestion": self.suggestion,
        }


def success(data: Any) -> Dict[str, Any]:
    """
    :param data: The payload of a successful call.
    :return: A success envelope wrapping ``data``.
    """
    return {"ok": True, "data": data}


def failure(error: ToolError) -> Dict[str, Any]:
    """
    :param error: The structured failure.
    :return: A failure envelope wrapping ``error``.
    """
    return {"ok": False, "error": error.to_dict()}


# %% boundary


def tool_boundary(operation: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
    """
    Wrap a tool operation so it always returns an envelope and never raises.

    Expected domain failures are reported with their own message and suggestion;
    unexpected failures are logged and reported without leaking internals, so malformed
    agent input can never crash the server.

    :param operation: The tool operation to wrap.
    :return: The wrapped operation returning a success or failure envelope.
    """

    @wraps(operation)
    def guarded(*arguments: Any, **keyword_arguments: Any) -> Dict[str, Any]:
        try:
            return success(operation(*arguments, **keyword_arguments))
        except DataclassException as known_failure:
            logger.warning("%s failed: %s", operation.__name__, known_failure)
            return failure(
                ToolError(
                    type(known_failure).__name__,
                    known_failure.error_message(),
                    known_failure.suggest_correction(),
                )
            )
        except Exception as unexpected_failure:
            logger.exception("%s raised an unexpected error", operation.__name__)
            return failure(
                ToolError(
                    "InternalError",
                    str(unexpected_failure),
                    "Report this with the server logs.",
                )
            )

    return guarded
