"""
The masses a pour ended with, measured by hand on a scale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import Any, Dict, Self

# %% the scale the measurements are read against

FULL_CUP_MASS_GRAMS = 2.65
"""
Mass of all particles together, which is the mass of a completely filled cup, in grams.
"""

WEIGHED_PARTICLES_FILE_NAME = "weighed_particles.json"
"""
Name of the file the measurements are written to inside the bag.
"""

RECEIVED_MASS_PROMPT = "Weigh the particles in the receiving cup, mass in grams: "
"""
What the experimenter is asked about the pour's result.
"""

SPILLED_MASS_PROMPT = "Weigh the particles that missed the cup, mass in grams: "
"""
What the experimenter is asked about what landed beside the cup.
"""

DRAINED_MASS_PROMPT = "Weigh the particles that came out of the cup, mass in grams: "
"""
What the experimenter is asked after a ramp has turned the cup back upright.
"""

MASS_REJECTION = "That is not a mass in grams."
"""
What the experimenter is told when an answer does not read as a number.
"""

_MASS_PATTERN = re.compile(r"^\d+(\.\d+)?$")
"""
A non-negative decimal number, the only answer accepted as a mass.
"""


class WeighedParticlesJSONKey(StrEnum):
    """
    The keys of a weighed pour in its JSON file.
    """

    RECEIVED_MASS_GRAMS = "received_mass_grams"
    """
    The mass that reached the receiving cup.
    """

    SPILLED_MASS_GRAMS = "spilled_mass_grams"
    """
    The mass that missed it.
    """

    RECEIVED_FILL_LEVEL = "received_fill_level"
    """
    The fill level the received mass amounts to.
    """

    FULL_CUP_MASS_GRAMS = "full_cup_mass_grams"
    """
    The mass of a full cup the fill level is measured against.
    """


# %% the measurements


@dataclass
class WeighedParticles(SubclassJSONSerializer):
    """
    Where a pour's particles ended up, as read off a scale after the run.

    Together with the fill the source started at, these two masses say how much is left
    in the source cup.
    """

    received_mass_grams: float
    """
    Mass of the particles in the receiving cup after the pour, in grams.
    """

    spilled_mass_grams: float
    """
    Mass of the particles that landed outside the receiving cup, in grams.
    """

    full_cup_mass_grams: float = FULL_CUP_MASS_GRAMS
    """
    Mass of the material poured when it fills the cup, in grams.
    """

    @property
    def received_fill_level(self) -> float:
        """
        :return: The received mass as a fraction of a full cup, comparable to the fill
            level the pour aimed at.
        """
        return self.received_mass_grams / self.full_cup_mass_grams

    @classmethod
    def ask(
        cls,
        answer_prompt: Callable[[str], str] = input,
        full_cup_mass_grams: float = FULL_CUP_MASS_GRAMS,
    ) -> Self:
        """
        Ask for both masses on the scale.

        :param answer_prompt: Puts a question to the experimenter and returns the
            answer.
        :param full_cup_mass_grams: Mass of the material poured when it fills the cup.
        :return: The measurements given.
        """
        return cls(
            received_mass_grams=ask_for_mass(answer_prompt, RECEIVED_MASS_PROMPT),
            spilled_mass_grams=ask_for_mass(answer_prompt, SPILLED_MASS_PROMPT),
            full_cup_mass_grams=full_cup_mass_grams,
        )

    def as_json_text(self) -> str:
        """
        :return: The measurements as the text written into the bag.
        """
        return json.dumps(self.to_json(), indent=2)

    def to_json(self, **kwargs) -> Dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            WeighedParticlesJSONKey.RECEIVED_MASS_GRAMS: self.received_mass_grams,
            WeighedParticlesJSONKey.SPILLED_MASS_GRAMS: self.spilled_mass_grams,
            WeighedParticlesJSONKey.RECEIVED_FILL_LEVEL: self.received_fill_level,
            WeighedParticlesJSONKey.FULL_CUP_MASS_GRAMS: self.full_cup_mass_grams,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            received_mass_grams=data[WeighedParticlesJSONKey.RECEIVED_MASS_GRAMS],
            spilled_mass_grams=data[WeighedParticlesJSONKey.SPILLED_MASS_GRAMS],
            full_cup_mass_grams=data.get(
                WeighedParticlesJSONKey.FULL_CUP_MASS_GRAMS, FULL_CUP_MASS_GRAMS
            ),
        )


@dataclass
class WeighedRamp:
    """
    How much a calibration ramp poured out, collected and read off a scale afterwards.

    The cup stays in the gripper across a session's ramps, so only what came out can be
    weighed; what is left in the cup follows from the fill it was loaded with and the
    masses the session's earlier ramps drained.
    """

    drained_mass_grams: float
    """
    Mass of the particles the ramp poured out, in grams.
    """

    @property
    def drained_fill_level(self) -> float:
        """
        :return: The drained mass as a fraction of a full cup, which is how much the
            source's fill level dropped over the ramp.
        """
        return self.drained_mass_grams / FULL_CUP_MASS_GRAMS


def ask_for_mass(answer_prompt: Callable[[str], str], question: str) -> float:
    """
    Put a question about a mass, repeating it until the answer reads as one.

    :param answer_prompt: Puts the question to the experimenter and returns the answer.
    :param question: The mass being asked for.
    :return: The mass given, in grams.
    """
    answer = answer_prompt(question).strip()
    while not _MASS_PATTERN.match(answer):
        answer = answer_prompt(f"{MASS_REJECTION} {question}").strip()
    return float(answer)
