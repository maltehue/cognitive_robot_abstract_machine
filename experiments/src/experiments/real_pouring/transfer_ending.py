"""
How a demo pour's transfer ended, and a way to end it by hand without losing the bag's
files.
"""

from __future__ import annotations

import signal
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from types import FrameType

from typing_extensions import Any, Callable, Optional

# %% how the transfer ended

TRANSFER_ENDING_FILE_NAME = "transfer_ending.txt"
"""
Name of the file inside the bag that says how the transfer ended.
"""


class TransferEnding(StrEnum):
    """
    What ended a transfer.
    """

    GOAL_REACHED = "goal_reached"
    """
    The receiver held the goal fill and the stream had settled.
    """

    HELD_AT_CEILING = "held_at_ceiling"
    """
    The cup stayed at its ceiling tilt long enough to have poured all it would.
    """

    TIME_LIMIT = "time_limit"
    """
    The transfer ran out of time before either of the above.
    """

    STOPPED_BY_OPERATOR = "stopped_by_operator"
    """
    The operator stopped the transfer by hand.
    """


# %% stopping the transfer by hand


@dataclass
class CancelOnInterrupt(AbstractContextManager):
    """
    Turns the first Ctrl+C inside a ``with`` block into a call of :attr:`cancel`
    instead of an interrupt, so the script carries on after the stopped motion; a second
    Ctrl+C interrupts as usual.

    The handler that was installed before is restored on exit; one installed outside
    Python, such as rclpy's, cannot be restored, and Ctrl+C interrupts as usual after
    the block instead.
    """

    cancel: Callable[[], Any]
    """
    Stops the running motion.
    """

    interrupted: bool = field(init=False, default=False)
    """
    Whether Ctrl+C was pressed inside the block.
    """

    _original_handler: Optional[Any] = field(init=False, default=None)
    """
    The SIGINT handler installed before entering the block, ``None`` for one installed
    outside Python.
    """

    def __enter__(self) -> CancelOnInterrupt:
        self._original_handler = signal.signal(signal.SIGINT, self._cancel_once)
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self._original_handler is None:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            return
        signal.signal(signal.SIGINT, self._original_handler)

    def _cancel_once(self, signum: int, frame: Optional[FrameType]) -> None:
        """
        Stop the motion and let any further Ctrl+C interrupt the script.
        """
        self.interrupted = True
        signal.signal(signal.SIGINT, signal.default_int_handler)
        print("Stopping the transfer; press Ctrl+C again to abort the script.")
        self.cancel()
