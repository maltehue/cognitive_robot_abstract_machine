"""
The parameters of the pouring effect model an experiment varies, and the reference values
they are varied around.
"""

# %% imports

from __future__ import annotations

import math
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from semantic_digital_twin.physics.equations.pouring_equations import (
    DEFAULT_DISCHARGE_COEFFICIENT,
    ArticulatedPouringEquation,
)

# %% which parameter is varied


class ModelParameter(StrEnum):
    """
    A parameter of the pouring effect model an experiment can set wrong on purpose.
    """

    REPOSE_ANGLE = "repose_angle"
    """
    The angle the contents hold before they flow; it moves the tilt a pour starts at.
    """

    OUTFLOW_RATE_CONSTANT = "outflow_rate_constant"
    """
    How fast a head above the lip drains the container.
    """

    DISCHARGE_COEFFICIENT = "discharge_coefficient"
    """
    How fast the stream leaves the lip, and so where it lands.
    """

    LIP_OFFSET = "lip_offset"
    """
    How far the pouring lip sits from the tilt axis.
    """


# %% one set of parameter values


class ParametersKey(StrEnum):
    """
    The keys a parameter set stores beside its parameters.
    """

    DYNAMIC_REPOSE_ANGLE = "dynamic_repose_angle"
    """
    The repose angle held while the cup is turning.
    """

    DYNAMIC_REPOSE_SPREAD = "dynamic_repose_spread"
    """
    How widely that angle scatters from pour to pour.
    """


@dataclass(frozen=True)
class ModelParameters:
    """
    The values the pouring effect model is evaluated with.

    The geometry the model reads off the container is not among them: it is measured, and
    every run reads the same one.
    """

    repose_angle: float
    """
    Angle the contents hold before they flow, in radians.
    """

    outflow_rate_constant: float
    """
    Outflow rate constant of the drain, in 1/s.
    """

    discharge_coefficient: float = DEFAULT_DISCHARGE_COEFFICIENT
    """
    Scales the Torricelli exit speed at the lip.
    """

    lip_offset_scale: float = 1.0
    """
    Factor on the lip's distance from the tilt axis the container geometry gives: half
    the width for a cup, the outlet's distance for a spout.
    """

    dynamic_repose_angle: Optional[float] = None
    """
    Angle the contents hold before they flow while the cup is turning, in radians, or
    ``None`` where it is not told apart from :attr:`repose_angle`, the angle they hold
    once they have come to rest.
    """

    dynamic_repose_spread: Optional[float] = None
    """
    How widely the angle held while turning scatters from pour to pour, as a standard
    deviation in radians, or ``None`` for one angle every pour holds.
    """

    def value_of(self, parameter: ModelParameter) -> float:
        """
        :param parameter: The parameter to read.
        :return: Its value in this set; the lip offset as its factor on the geometry.
        """
        match parameter:
            case ModelParameter.REPOSE_ANGLE:
                return self.repose_angle
            case ModelParameter.OUTFLOW_RATE_CONSTANT:
                return self.outflow_rate_constant
            case ModelParameter.DISCHARGE_COEFFICIENT:
                return self.discharge_coefficient
            case ModelParameter.LIP_OFFSET:
                return self.lip_offset_scale

    def with_value(self, parameter: ModelParameter, value: float) -> ModelParameters:
        """
        :param parameter: The parameter to change.
        :param value: Its new value; the lip offset as its factor on the geometry.
        :return: A copy of this set with that one parameter changed.
        """
        match parameter:
            case ModelParameter.REPOSE_ANGLE:
                return replace(self, repose_angle=value)
            case ModelParameter.OUTFLOW_RATE_CONSTANT:
                return replace(self, outflow_rate_constant=value)
            case ModelParameter.DISCHARGE_COEFFICIENT:
                return replace(self, discharge_coefficient=value)
            case ModelParameter.LIP_OFFSET:
                return replace(self, lip_offset_scale=value)

    def scaled(self, parameter: ModelParameter, factor: float) -> ModelParameters:
        """
        :param parameter: The parameter to scale.
        :param factor: What to multiply it by.
        :return: A copy of this set with that one parameter scaled; scaling the repose
            angle scales the angle held while turning with it.
        """
        scaled = self.with_value(parameter, self.value_of(parameter) * factor)
        if parameter is not ModelParameter.REPOSE_ANGLE or (
            self.dynamic_repose_angle is None
        ):
            return scaled
        return scaled.with_dynamic_repose_angle(self.dynamic_repose_angle * factor)

    def with_dynamic_repose_angle(self, angle: Optional[float]) -> ModelParameters:
        """
        :param angle: The angle the contents hold while the cup is turning, in radians.
        :return: A copy of this set holding that angle while turning.
        """
        return replace(self, dynamic_repose_angle=angle)

    def with_dynamic_repose_spread(self, spread: Optional[float]) -> ModelParameters:
        """
        :param spread: How widely the angle held while turning scatters from pour to
            pour, in radians.
        :return: A copy of this set with that spread.
        """
        return replace(self, dynamic_repose_spread=spread)

    def while_turning(self) -> ModelParameters:
        """
        :return: This set as it holds while the cup is turning: the dynamic repose angle
            in place of the one at rest, where one is known.
        """
        if self.dynamic_repose_angle is None:
            return self
        return replace(self, repose_angle=self.dynamic_repose_angle)

    def applied_to(
        self, equation: ArticulatedPouringEquation
    ) -> ArticulatedPouringEquation:
        """
        :param equation: A drain built from a container's geometry.
        :return: The same drain evaluated with these parameters.
        """
        return replace(
            equation,
            repose_angle=self.repose_angle,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
            lip_offset=equation.lip_offset * self.lip_offset_scale,
        )

    def to_json(self) -> Dict[str, Any]:
        values = {
            parameter.value: self.value_of(parameter) for parameter in ModelParameter
        }
        optional = {
            ParametersKey.DYNAMIC_REPOSE_ANGLE: self.dynamic_repose_angle,
            ParametersKey.DYNAMIC_REPOSE_SPREAD: self.dynamic_repose_spread,
        }
        return {
            **values,
            **{key: value for key, value in optional.items() if value is not None},
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ModelParameters:
        return cls(
            repose_angle=data[ModelParameter.REPOSE_ANGLE],
            outflow_rate_constant=data[ModelParameter.OUTFLOW_RATE_CONSTANT],
            discharge_coefficient=data[ModelParameter.DISCHARGE_COEFFICIENT],
            lip_offset_scale=data[ModelParameter.LIP_OFFSET],
            dynamic_repose_angle=data.get(ParametersKey.DYNAMIC_REPOSE_ANGLE),
            dynamic_repose_spread=data.get(ParametersKey.DYNAMIC_REPOSE_SPREAD),
        )


DEMO_PARAMETERS = ModelParameters(
    repose_angle=math.radians(36.8), outflow_rate_constant=0.24
)
"""
The parameters ``demo_pouring_transfer_particles.py`` runs with, taken from an earlier fit
in ``semantic_digital_twin/doc/pouring_effect_model_evaluation.md`` (findings 2 and 5).
"""

NOMINAL_PARAMETERS = ModelParameters(repose_angle=0.0, outflow_rate_constant=1.0)
"""
The model as it stands before any parameter of the contents is estimated: a free liquid
surface draining at the default rate.
"""


# %% how far a parameter is set wrong


class PerturbationKey(StrEnum):
    """
    The keys a perturbation is stored under.
    """

    PARAMETER = "parameter"
    FACTOR = "factor"
    NOMINAL = "nominal"
    SECOND_PARAMETER = "second_parameter"
    SECOND_FACTOR = "second_factor"


@dataclass(frozen=True)
class Perturbation:
    """
    One parameter of a baseline set scaled by a factor, or the nominal set.
    """

    parameter: ModelParameter | None
    """
    The parameter that is scaled, or ``None`` for an unscaled set.
    """

    factor: float = 1.0
    """
    What the parameter is multiplied by.
    """

    nominal: bool = False
    """
    Whether this is the nominal model rather than a scaled reference.
    """

    second_parameter: ModelParameter | None = None
    """
    A second parameter scaled at the same time, or ``None`` for one alone.
    """

    second_factor: float = 1.0
    """
    What the second parameter is multiplied by.
    """

    def around(self, baseline: ModelParameters) -> ModelParameters:
        """
        :param baseline: The parameters the perturbation is taken from.
        :return: The parameter values this perturbation evaluates the model with.
        """
        if self.nominal:
            return NOMINAL_PARAMETERS
        if self.parameter is None:
            return baseline
        scaled = baseline.scaled(self.parameter, self.factor)
        if self.second_parameter is None:
            return scaled
        return scaled.scaled(self.second_parameter, self.second_factor)

    @property
    def label(self) -> str:
        """
        :return: A short identifier of this perturbation, usable in a file name.
        """
        if self.nominal:
            return "nominal"
        if self.parameter is None:
            return "baseline"
        label = f"{self.parameter.value}_x{self.factor:g}"
        if self.second_parameter is None:
            return label
        return f"{label}_{self.second_parameter.value}_x{self.second_factor:g}"

    def to_json(self) -> Dict[str, Any]:
        result = {
            PerturbationKey.PARAMETER: (
                None if self.parameter is None else self.parameter.value
            ),
            PerturbationKey.FACTOR: self.factor,
            PerturbationKey.NOMINAL: self.nominal,
        }
        if self.second_parameter is not None:
            result[PerturbationKey.SECOND_PARAMETER] = self.second_parameter.value
            result[PerturbationKey.SECOND_FACTOR] = self.second_factor
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Perturbation:
        parameter = data[PerturbationKey.PARAMETER]
        second = data.get(PerturbationKey.SECOND_PARAMETER)
        return cls(
            parameter=None if parameter is None else ModelParameter(parameter),
            factor=data[PerturbationKey.FACTOR],
            nominal=data[PerturbationKey.NOMINAL],
            second_parameter=None if second is None else ModelParameter(second),
            second_factor=data.get(PerturbationKey.SECOND_FACTOR, 1.0),
        )


BASELINE = Perturbation(parameter=None)
"""
The baseline parameters, unscaled.
"""

NOMINAL = Perturbation(parameter=None, nominal=True)
"""
The nominal model.
"""


def closed_loop_perturbations() -> List[Perturbation]:
    """
    The parameter sets the closed-loop study runs: the baseline, each controller-relevant
    parameter scaled one at a time on both sides of it, and the nominal model.

    :return: The perturbations, the baseline first.
    """
    scaled = {
        ModelParameter.REPOSE_ANGLE: (0.0, 0.5, 0.75, 1.25),
        ModelParameter.OUTFLOW_RATE_CONSTANT: (0.25, 0.5, 2.0, 4.0),
        ModelParameter.DISCHARGE_COEFFICIENT: (0.33, 2.0),
    }
    return (
        [BASELINE]
        + [
            Perturbation(parameter=parameter, factor=factor)
            for parameter, factors in scaled.items()
            for factor in factors
        ]
        + [NOMINAL]
    )


# %% the baseline of each source cup


@dataclass(frozen=True)
class Baselines:
    """
    The parameters each source cup's closed-loop perturbations are taken from: those
    fitted to that cup's own open-loop trials.
    """

    parameters: Dict[str, ModelParameters]
    """
    The fitted parameters, keyed by source cup.
    """

    def of(self, source: str) -> ModelParameters:
        """
        :param source: A source cup.
        :return: Its baseline parameters.
        :raises MissingBaselineError: If no fit was recorded for that cup.
        """
        if source not in self.parameters:
            raise MissingBaselineError(source=source, known=sorted(self.parameters))
        return self.parameters[source]

    def write(self, path: Path) -> None:
        """
        :param path: The file to write the baselines to.
        """
        path.write_text(
            json.dumps(
                {
                    source: parameters.to_json()
                    for source, parameters in self.parameters.items()
                },
                indent=2,
            )
        )

    @classmethod
    def read(cls, path: Path) -> Baselines:
        """
        :param path: A file written by :meth:`write`.
        :return: The baselines it holds.
        :raises BaselinesNotFittedError: If the file does not exist yet.
        """
        if not path.exists():
            raise BaselinesNotFittedError(path=path)
        return cls(
            parameters={
                source: ModelParameters.from_json(data)
                for source, data in json.loads(path.read_text()).items()
            }
        )


@dataclass
class MissingBaselineError(Exception):
    """
    A closed-loop trial was asked for a source cup whose open-loop trials have not been
    fitted yet; run the tilt-rig study and the fit first.
    """

    source: str
    """
    The source cup that has no baseline.
    """

    known: List[str]
    """
    The source cups that do.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"no fitted baseline for source cup {self.source!r} (fitted: {self.known}); "
            f"record the tilt-rig study and run the fit first"
        )


@dataclass
class BaselinesNotFittedError(Exception):
    """
    The closed-loop study was started before any source cup's open-loop trials were
    fitted; run the tilt-rig study and the fit first.
    """

    path: Path
    """
    Where the baselines were looked for.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"no baselines at {self.path}; record the tilt-rig study and run the fit first"
        )
