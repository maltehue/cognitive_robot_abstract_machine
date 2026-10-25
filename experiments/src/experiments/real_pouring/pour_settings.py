"""
What a demo pour was asked to do, kept in its bag next to what it delivered.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
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

    corrects_drain: Optional[bool] = field(default=None, kw_only=True)
    """
    Whether the live bank corrected the controller's drain, or told it only the fills.
    """

    learned_head: Optional[bool] = field(default=None, kw_only=True)
    """
    Whether the learned head drove the transfer instead of the analytic one.
    """

    transfer_time_limit: Optional[float] = field(default=None, kw_only=True)
    """
    How long the transfer could run before it was ended, in seconds.
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


# %% the label a pour's bag is named with

LABEL_SEPARATOR = "_"
"""
Separates the parts of a pour's label.
"""

DECIMAL_MARK = "p"
"""
Stands for the decimal point in a label: a ROS node named after the bag may not hold a
dot.
"""


class PourLabelMarker(StrEnum):
    """
    Words a label adds where a run departs from the usual setup.
    """

    WITHOUT_BANK = "nobank"
    """
    The live bank did not run.
    """

    WITH_LEARNED_HEAD = "learnedhead"
    """
    The learned head drove the transfer.
    """

    FILLS_ONLY = "fillsonly"
    """
    The live bank told the controller the fills but left its drain as built.
    """


@dataclass(frozen=True)
class PourLabel:
    """
    What tells pours apart at a glance, for naming their bags.
    """

    material: PouredMaterial
    """
    What was poured.
    """

    receiving_cup: ReceivingCup
    """
    The cup poured into.
    """

    goal_fill: float
    """
    Fill of the receiving cup the pour aims for.
    """

    start_fill: float
    """
    Fill of the source cup when the pour started.
    """

    approach_tilt_rate: float
    """
    How fast the cup kept turning where the model predicted no flow, in radians per
    second.
    """

    live_bank: bool
    """
    Whether the live bank ran.
    """

    learned_head: bool
    """
    Whether the learned head drove the transfer.
    """

    corrects_drain: bool = field(default=True, kw_only=True)
    """
    Whether the live bank corrected the controller's drain as well as telling it the
    fills.
    """

    def text(self) -> str:
        """
        :return: Material, receiving cup, goal and start fill in percent, and the
            approach rate in degrees per second, followed by a marker for each
            departure from the usual setup.
        """
        approach = f"{round(math.degrees(self.approach_tilt_rate), 1):g}".replace(
            ".", DECIMAL_MARK
        )
        parts = [
            self.material,
            self.receiving_cup,
            f"goal{round(self.goal_fill * 100)}",
            f"start{round(self.start_fill * 100)}",
            f"approach{approach}",
        ]
        if not self.live_bank:
            parts.append(PourLabelMarker.WITHOUT_BANK)
        elif not self.corrects_drain:
            parts.append(PourLabelMarker.FILLS_ONLY)
        if self.learned_head:
            parts.append(PourLabelMarker.WITH_LEARNED_HEAD)
        return LABEL_SEPARATOR.join(parts)
