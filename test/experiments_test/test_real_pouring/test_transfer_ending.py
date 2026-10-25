"""
Stopping a demo transfer by hand without losing the rest of the script.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass

import pytest

from experiments.real_pouring.transfer_ending import CancelOnInterrupt

# %% the first Ctrl+C stops the motion, a second one the script


@dataclass
class CountedCancel:
    """
    Stands in for stopping a motion and counts how often it was asked to.
    """

    calls: int = 0
    """
    How often the motion was asked to stop.
    """

    def __call__(self) -> None:
        self.calls += 1


@pytest.fixture
def python_handler():
    """
    A SIGINT handler installed from Python, restored after the test.
    """

    def handler(signum, frame) -> None:
        return None

    original = signal.signal(signal.SIGINT, handler)
    yield handler
    signal.signal(signal.SIGINT, original)


def test_the_first_interrupt_cancels_instead_of_interrupting(python_handler) -> None:
    """
    Ctrl+C inside the block stops the motion once and lets the script go on.
    """
    cancel = CountedCancel()
    with CancelOnInterrupt(cancel) as interruption:
        signal.raise_signal(signal.SIGINT)
    assert cancel.calls == 1
    assert interruption.interrupted


def test_a_second_interrupt_interrupts_the_script(python_handler) -> None:
    """
    Ctrl+C pressed again while the motion is being stopped aborts the script.
    """
    cancel = CountedCancel()
    with pytest.raises(KeyboardInterrupt):
        with CancelOnInterrupt(cancel):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
    assert cancel.calls == 1


def test_a_block_left_without_interrupt_cancels_nothing(python_handler) -> None:
    """
    A transfer that ends by itself is not stopped and not marked as interrupted.
    """
    cancel = CountedCancel()
    with CancelOnInterrupt(cancel) as interruption:
        pass
    assert cancel.calls == 0
    assert not interruption.interrupted


def test_the_previous_handler_is_restored(python_handler) -> None:
    """
    After the block, Ctrl+C is handled as before it.
    """
    with CancelOnInterrupt(CountedCancel()):
        signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is python_handler
