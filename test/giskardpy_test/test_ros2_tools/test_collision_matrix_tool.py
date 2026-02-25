import os
import signal
import subprocess
import time
from importlib.resources import files

import pytest


def test_script_launch_and_kill():
    script_path = files("giskardpy").joinpath(
        "../../scripts/ros2-tools/collision_matrix_tool.py"
    )

    # Start the process in a new process group
    process = subprocess.Popen(
        ["python3", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    try:
        # Give it enough time to initialize (e.g., 3-5 seconds)
        time.sleep(3)

        # Check if it crashed immediately
        if process.poll() is not None:
            _, stderr = process.communicate()
            pytest.fail(f"Script crashed on startup. Error: {stderr.decode()}")

        # Send SIGINT (Ctrl+C)
        os.killpg(os.getpgid(process.pid), signal.SIGINT)

        # Wait for clean shutdown
        process.communicate(timeout=10)

        # Return code 0 (Success) or -2 (SIGINT) are expected
        # Note: In headless environments, you might see -6 (SIGABRT) from Qt
        assert process.returncode in [0, -signal.SIGINT, -6]

    finally:
        if process.poll() is None:
            process.kill()
