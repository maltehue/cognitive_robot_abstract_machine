"""
Tilt the held cup about its pouring axis at a fixed rate, hold it there, and turn it
back upright: one calibration ramp for the onset angle and the drain rate.

No fill model takes part: only the left wrist's last joint turns, which tilts the cup
without moving the tool frame. Where the balls start to flow and how long the stream
lasts are read afterwards from the video and the joint states; this script writes when
each phase began, so a run's bag, video and log-sheet row can be lined up.

Start ``ros2 bag record`` on the left arm's joint states before running it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from tracy_pouring_setup import (
    LEFT_TILT_JOINT,
    connect_to_giskard,
    move_left_tool_to,
    park_arms,
    upright_carry_pose,
)

# %% settings

RECORD_DIRECTORY = Path("experiments/results/real_pouring/ramps")
"""
Where each ramp's record is written, relative to the workspace root.
"""

STATE_SETTLE_TIME = 0.5
"""
Seconds to wait after a motion before reading the joint state Giskard synchronized.
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


def turn_tilt_joint_to(
    giskard: GiskardWrapper, position: float, rate_deg_s: float
) -> None:
    """
    Turn only the tilt joint to ``position`` (radians), no faster than ``rate_deg_s``.
    """
    joint = giskard.world.get_connection_by_name(LEFT_TILT_JOINT)
    rate = math.radians(rate_deg_s)
    task = JointPositionList(
        goal_state=JointState.from_mapping({joint: position}), max_velocity=rate
    )
    motion = MotionStatechart()
    motion.add_node(task)
    motion.add_node(JointVelocityLimit(connections=[joint], max_velocity=rate))
    motion.add_node(EndMotion.when_true(task))
    giskard.execute(motion)


def tilt_joint_position(giskard: GiskardWrapper) -> float:
    """
    :return: The tilt joint's current position, in radians, once the state has settled.
    """
    time.sleep(STATE_SETTLE_TIME)
    return float(giskard.world.get_connection_by_name(LEFT_TILT_JOINT).position)


def run_ramp(giskard: GiskardWrapper, settings: RampSettings) -> RampRecord:
    """
    Bring the cup upright above the table, wait for the operator, then tilt, hold and
    turn it back.
    """
    park_arms(giskard)
    move_left_tool_to(giskard, upright_carry_pose(giskard))
    upright = tilt_joint_position(giskard)
    input(
        f"[{settings.run_id}] Cup loaded and levelled, recording running? Enter to tilt."
    )
    tilt_started_at = time.time()
    turn_tilt_joint_to(
        giskard,
        upright + math.radians(settings.final_tilt_deg),
        settings.tilt_rate_deg_s,
    )
    final_tilt_reached_at = time.time()
    time.sleep(settings.hold_s)
    hold_ended_at = time.time()
    turn_tilt_joint_to(giskard, upright, settings.return_rate_deg_s)
    return RampRecord(
        settings=settings,
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        upright_joint_position=upright,
        tilt_started_at=tilt_started_at,
        final_tilt_reached_at=final_tilt_reached_at,
        hold_ended_at=hold_ended_at,
        upright_again_at=time.time(),
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
    record = run_ramp(connect_to_giskard("tracy_calibration_ramp"), settings)
    RECORD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(asdict(record), indent=2))
    print(f"wrote {record_path}")
