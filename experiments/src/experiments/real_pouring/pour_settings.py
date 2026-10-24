"""
What a demo pour was asked to do, kept in its bag next to what it delivered.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Optional

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from krrood.adapters.json_serializer import from_json, to_json

# %% the settings of one pour

POUR_SETTINGS_FILE_NAME = "pour_settings.json"
"""
Name of the file the settings are written to inside the bag.
"""


@dataclass
class MissingPourSettingsError(Exception):
    """
    Raised when a bag holds no pour settings, as bags recorded before they were kept do.
    """

    bag: Path
    """
    The bag read.
    """

    def __str__(self) -> str:
        return f"{self.bag} holds no {POUR_SETTINGS_FILE_NAME}."


@dataclass(frozen=True)
class RepositoryState:
    """
    The code a pour ran with.
    """

    commit: str
    """
    The commit checked out.
    """

    has_uncommitted_changes: bool
    """
    Whether the working tree differed from that commit.
    """

    @classmethod
    def of_working_tree(cls, directory: Path) -> RepositoryState:
        """
        :param directory: A directory inside the repository.
        :return: The commit checked out there, and whether anything tracked or untracked
            differs from it.
        """

        def git(*arguments: str) -> str:
            return subprocess.run(
                ["git", "-C", str(directory), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        return cls(
            commit=git("rev-parse", "HEAD"),
            has_uncommitted_changes=bool(git("status", "--porcelain")),
        )


@dataclass(frozen=True)
class TiltPacingSettings:
    """
    How the source cup's tilt was paced during the transfer, in radians and seconds.
    """

    approach_rate: float
    """
    How fast the cup kept turning where the model predicted no flow, per second.
    """

    maximum_rate: float
    """
    The fastest the cup could tilt further, per second.
    """

    maximum_return_rate: float
    """
    The fastest the cup could tilt back, per second.
    """

    maximum_acceleration: Optional[float]
    """
    The fastest the tilt rate could change, per second squared, or ``None`` without a
    limit.
    """

    maximum_tilt: float
    """
    The tilt the cup was not turned past.
    """

    ceiling_slowdown_width: float
    """
    The span below :attr:`maximum_tilt` over which tilting further eased to zero.
    """

    transfer_start_tilt: float
    """
    The tilt the cup was turned to before the transfer started.
    """

    ceiling_hold_seconds: float
    """
    How long the cup could stay at :attr:`maximum_tilt` before the transfer ended.
    """


@dataclass(frozen=True)
class PourSettings:
    """
    The goal and controller settings a demo pour ran with.

    Every field after :attr:`prediction_duration` came later; bags recorded before it
    read back with ``None`` there.
    """

    goal_fill: float
    """
    Fill of the receiving cup the pour aims for.
    """

    fill_tolerance: float
    """
    How far short of the goal fill the pour may stop.
    """

    couple_source_fill: bool
    """
    Whether the source's fill drained along in the receiver's fill prediction.
    """

    prediction_duration: float
    """
    How far ahead the fill task predicted, in seconds.
    """

    start_fill: Optional[float] = field(default=None, kw_only=True)
    """
    Fill of the source cup when the pour started.
    """

    material: Optional[PouredMaterial] = field(default=None, kw_only=True)
    """
    What was poured.
    """

    receiving_cup: Optional[ReceivingCup] = field(default=None, kw_only=True)
    """
    The cup poured into.
    """

    pouring_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    How fast the fill task could raise the receiver's predicted fill, per second.
    """

    stopping_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    How fast the fill task could take back fill predicted beyond the goal, per second.
    """

    settled_inflow: Optional[float] = field(default=None, kw_only=True)
    """
    The inflow below which the receiver counted as no longer filling, per second.
    """

    tilt_pacing: Optional[TiltPacingSettings] = field(default=None, kw_only=True)
    """
    How the tilt was paced.
    """

    drain: Optional[ModelPoint] = field(default=None, kw_only=True)
    """
    The drain the controller and the bank started from.
    """

    bank: Optional[BankSettings] = field(default=None, kw_only=True)
    """
    The live bank's settings, or ``None`` where the bank did not run.
    """

    stop_lead_time: Optional[float] = field(default=None, kw_only=True)
    """
    How long grains were taken to keep arriving after a stop, in seconds.
    """

    learned_head: Optional[bool] = field(default=None, kw_only=True)
    """
    Whether the learned head drove the transfer instead of the analytic one.
    """

    repository: Optional[RepositoryState] = field(default=None, kw_only=True)
    """
    The code the pour ran with.
    """

    def as_json_text(self) -> str:
        """
        :return: The settings as the text of their file in the bag.
        """
        return json.dumps(to_json(self), indent=2)

    @classmethod
    def read(cls, bag: Path) -> PourSettings:
        """
        :param bag: The bag directory.
        :return: The settings the bag was recorded with.
        :raises MissingPourSettingsError: If the bag holds none.
        """
        path = bag / POUR_SETTINGS_FILE_NAME
        if not path.exists():
            raise MissingPourSettingsError(bag)
        return from_json(json.loads(path.read_text()))
