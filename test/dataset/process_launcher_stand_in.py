"""
A process that runs a second one and keeps going, standing in for a launcher such as
``ros2 run``.

It prints the started process's id on its first line and then waits until it is stopped,
so that a test can check what stopping the launcher leaves behind::

    python process_launcher_stand_in.py
"""

from __future__ import annotations

import subprocess
import sys
import time

CHILD_ARGUMENT = "--child"
"""
The argument this script passes to the copy of itself it starts.
"""

WAIT_SECONDS = 3600.0
"""
How long to wait for the signal that ends this process.
"""


def wait_to_be_stopped() -> None:
    """
    Do nothing until this process is signalled.
    """
    time.sleep(WAIT_SECONDS)


def launch_and_wait() -> None:
    """
    Start a copy of this script, report which process it became and wait alongside it.
    """
    child = subprocess.Popen([sys.executable, __file__, CHILD_ARGUMENT])
    print(child.pid, flush=True)
    wait_to_be_stopped()


def main() -> None:
    """
    Wait as the started process, or start one and wait next to it.
    """
    if CHILD_ARGUMENT in sys.argv:
        wait_to_be_stopped()
        return
    launch_and_wait()


if __name__ == "__main__":
    main()
