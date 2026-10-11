"""
The pouring model of a spouted cup with its slot edge as a second lip.

A spouted cup's grains leave over the outlet at the end of the trough, but they first
pass the edge of the slot in the cup's wall. The model with only the outlet as its lip
misses that edge; this model pours over whichever of the two stands lower at the tilt,
with the slot edge's height as something a controller can switch on and move while it
runs, so every hypothesis about the edge is compiled once in advance.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from krrood.patterns.field_metadata import JSONMetadata
from krrood.symbolic_math.symbolic_math import Scalar
from typing_extensions import Optional

from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    FillContext,
    GatedArticulatedPouringEquation,
)

# %% the drain


@dataclass
class SecondLipPouringEquation(ArticulatedPouringEquation):
    """
    The articulated drain of a cup that pours over its outlet or over the slot edge,
    whichever gives the higher head at the tilt.
    """

    slot_offset: float = field(default=0.0, kw_only=True)
    """
    Horizontal distance of the slot edge from the tilt axis, in metres.
    """

    slot_height: Optional[Scalar] = field(
        default=None, kw_only=True, metadata=JSONMetadata(serialize=False).as_dict()
    )
    """
    Height of the slot edge above the cup's floor, in metres, as a free variable.
    """

    slot_open: Optional[Scalar] = field(
        default=None, kw_only=True, metadata=JSONMetadata(serialize=False).as_dict()
    )
    """
    One while the model pours over the slot edge as well, zero while it pours over the
    outlet alone, as a free variable.
    """

    def _repose(self) -> Scalar:
        """
        :return: The repose angle the model currently pours with.
        """
        if self.repose_correction is None:
            return sm.Scalar(self.repose_angle)
        return self.repose_angle + self.repose_correction

    def head_above_lip(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context providing the tilt and fill symbols.
        :return: The head above the outlet, or above the slot edge where that stands
            lower and the slot is switched on.
        """
        outlet = super().head_above_lip(context)
        if self.slot_height is None or self.slot_open is None:
            return outlet
        surface = context.fill_position * self.container_height
        dry = self.slot_height - surface
        reach = sm.sqrt(dry**2 + self.slot_offset**2)
        onset = sm.atan2(dry, sm.Scalar(self.slot_offset)) + self._repose()
        over_slot = sm.max(
            sm.Scalar(0.0), reach * sm.sin(context.tilt_expression - onset)
        )
        return sm.max(outlet, self.slot_open * over_slot)

    def with_gate(self, gate: Scalar) -> GatedSecondLipPouringEquation:
        return GatedSecondLipPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
            lip_offset=self.lip_offset,
            repose_angle=self.repose_angle,
            capacity=self.capacity,
            outflow_scale=self.outflow_scale,
            repose_correction=self.repose_correction,
            discharge_scale=self.discharge_scale,
            slot_offset=self.slot_offset,
            slot_height=self.slot_height,
            slot_open=self.slot_open,
            gate=gate,
        )


@dataclass
class GatedSecondLipPouringEquation(
    SecondLipPouringEquation, GatedArticulatedPouringEquation
):
    """
    The two-lip drain, flowing only while the transfer gate is open.
    """
