"""
What a demo pour was asked to do, kept in its bag next to what it delivered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
class PourSettings:
    """
    The goal and controller settings a demo pour ran with.
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
