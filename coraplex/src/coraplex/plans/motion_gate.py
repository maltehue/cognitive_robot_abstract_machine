from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

from typing_extensions import Iterator

# %% keeping model changes out of running motions


@dataclass
class MotionAndModelChangeGate:
    """
    Lets any number of motions run at once, but never while the world model changes.

    Plans performed at the same time, one thread per robot, share one world. Every
    change of that world reaches every giskard, and a giskard that is executing a goal
    at that moment aborts it. So a robot attaching the object it grasped would kill the
    motion of an unrelated robot.

    A motion may be held next to other motions; a model change is held alone. A model
    change that is waiting holds back motions that arrive after it, so a stream of
    motions cannot keep it out forever.

    The gate is not re-entrant: a thread holding it for a motion must leave before it
    asks for a model change, and the other way round.
    """

    _condition: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )
    """
    Guards the counters below and wakes whoever waits for them to reach zero.
    """

    _running_motions: int = field(default=0, init=False)
    """
    How many motions hold the gate.
    """

    _running_model_change: bool = field(default=False, init=False)
    """
    Whether a model change holds the gate.
    """

    _waiting_model_changes: int = field(default=0, init=False)
    """
    How many model changes wait for the running motions to end.
    """

    @contextmanager
    def motion(self) -> Iterator[None]:
        """
        Hold the gate for a motion, next to every other motion that is running.

        Blocks while a model change is being applied or waits to be applied.
        """
        with self._condition:
            while self._running_model_change or self._waiting_model_changes:
                self._condition.wait()
            self._running_motions += 1
        try:
            yield
        finally:
            with self._condition:
                self._running_motions -= 1
                self._condition.notify_all()

    @contextmanager
    def model_change(self) -> Iterator[None]:
        """
        Hold the gate alone for a change of the shared world model.

        Blocks until every running motion ended and keeps new motions out until the
        block is left.
        """
        with self._condition:
            self._waiting_model_changes += 1
            while self._running_model_change or self._running_motions:
                self._condition.wait()
            self._waiting_model_changes -= 1
            self._running_model_change = True
        try:
            yield
        finally:
            with self._condition:
                self._running_model_change = False
                self._condition.notify_all()
