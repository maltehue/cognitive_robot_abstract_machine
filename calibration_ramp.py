"""
Tilt the held cup about its pouring axis at a fixed rate, hold it there, and turn it
back upright: one calibration ramp for the onset angle and the drain rate.

No fill model takes part: only the left wrist's last joint turns, which tilts the cup
without moving the tool frame. Where the balls start to flow and how long the stream
lasts are read afterwards from the video and the joint states; this script writes when
each phase began, so a run's bag, video and log-sheet row can be lined up.

The run records itself: the bag covers the tilt, the hold and the return, and carries
the ramp's record, so nothing has to be lined up by hand afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from giskardpy.middleware.ros2.particle_weighing import (
    DRAINED_MASS_PROMPT,
    WeighedRamp,
    ask_for_mass,
)
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.middleware.ros2.rosbag_recording import (
    DEFAULT_BAG_DIRECTORY,
    RosbagRecording,
)
from tracy_pouring_setup import (
    connect_to_giskard,
    move_left_tool_to,
    park_arms,
    tilt_joint_position,
    turn_tilt_joint_to,
    upright_carry_pose,
)

# %% settings

RECORD_DIRECTORY = Path("experiments/results/real_pouring/ramps")
"""
Where each ramp's record is written, relative to the workspace root.
"""

RAMP_RECORD_FILE_NAME = "ramp.json"
"""
Name the ramp's record gets inside the bag, next to the recorded messages.
"""

BAG_DIRECTORY = DEFAULT_BAG_DIRECTORY / "calibration_ramps"
"""
Where the ramps' bags are written, apart from the transfer demo's bags.
"""


@dataclass
class RampSettings:
    """
    The ramp one run drives.
    """

    run_id: str
    """
    The log sheet's run id; also the record's file name.
    """

    tilt_rate_deg_s: float
    """
    Commanded tilt rate, in degrees per second.
    """

    final_tilt_deg: float
    """
    Tilt the ramp stops at, measured from upright, in degrees.
    """

    hold_s: float
    """
    Seconds the cup is held at the final tilt.
    """

    return_rate_deg_s: float
    """
    Tilt rate turning the cup back upright, in degrees per second.
    """


@dataclass
class RampRecord:
    """
    What one ramp did and when, in wall-clock seconds since the epoch.
    """

    settings: RampSettings
    """
    The ramp that was commanded.
    """

    git_commit: str
    """
    The workspace commit the ramp ran from.
    """

    bag_path: str
    """
    The bag the ramp was recorded into.
    """

    weighed: WeighedRamp
    """
    What the ramp poured out, measured on a scale.
    """

    upright_joint_position: float
    """
    The tilt joint's position with the cup upright, in radians; the tilt is the joint
    position minus this.
    """

    tilt_started_at: float
    """
    When the ramp goal was sent.
    """

    final_tilt_reached_at: float
    """
    When Giskard reported the final tilt reached.
    """

    hold_ended_at: float
    """
    When the cup started turning back.
    """

    upright_again_at: float
    """
    When Giskard reported the cup upright again.
    """


# %% motion


def run_ramp(
    giskard: GiskardWrapper, settings: RampSettings, recording: RosbagRecording
) -> RampRecord:
    """
    Bring the cup upright above the table, wait for the operator, then record the tilt,
    the hold and the return, and ask what came out.
    """
    # park_arms(giskard)
    move_left_tool_to(giskard, upright_carry_pose(giskard))
    upright = tilt_joint_position(giskard)
    input(f"[{settings.run_id}] Cup loaded and levelled? Enter to record and tilt.")
    print(f"Recording into {recording.start()}.")
    tilt_started_at = time.time()
    turn_tilt_joint_to(
        giskard,
        upright + math.radians(settings.final_tilt_deg),
        math.radians(settings.tilt_rate_deg_s),
    )
    final_tilt_reached_at = time.time()
    time.sleep(settings.hold_s)
    hold_ended_at = time.time()
    turn_tilt_joint_to(giskard, upright, math.radians(settings.return_rate_deg_s))
    upright_again_at = time.time()
    recording.stop()
    weighed = WeighedRamp(drained_mass_grams=ask_for_mass(input, DRAINED_MASS_PROMPT))
    print(f"Drained {weighed.drained_fill_level:.3f} of a full cup.")
    return RampRecord(
        settings=settings,
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        bag_path=str(recording.bag_path),
        weighed=weighed,
        upright_joint_position=upright,
        tilt_started_at=tilt_started_at,
        final_tilt_reached_at=final_tilt_reached_at,
        hold_ended_at=hold_ended_at,
        upright_again_at=upright_again_at,
    )


# %% entry point


def parse_settings() -> RampSettings:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id", help="the log sheet's run id")
    parser.add_argument("--tilt-rate", type=float, default=10.0, help="deg/s")
    parser.add_argument("--final-tilt", type=float, default=110.0, help="deg")
    parser.add_argument("--hold", type=float, default=5.0, help="s")
    parser.add_argument("--return-rate", type=float, default=30.0, help="deg/s")
    arguments = parser.parse_args()
    return RampSettings(
        run_id=arguments.run_id,
        tilt_rate_deg_s=arguments.tilt_rate,
        final_tilt_deg=arguments.final_tilt,
        hold_s=arguments.hold,
        return_rate_deg_s=arguments.return_rate,
    )


if __name__ == "__main__":
    settings = parse_settings()
    record_path = RECORD_DIRECTORY / f"{settings.run_id}.json"
    if record_path.exists():
        raise SystemExit(f"{record_path} exists; pick a new run id")
    recording = RosbagRecording(directory=BAG_DIRECTORY)
    record = run_ramp(connect_to_giskard("tracy_calibration_ramp"), settings, recording)
    record_text = json.dumps(asdict(record), indent=2)
    RECORD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    record_path.write_text(record_text)
    print(f"wrote {record_path}")
    print(f"wrote {recording.add_file(RAMP_RECORD_FILE_NAME, record_text)}")
