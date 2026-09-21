"""
Holding a drain's scale at what makes a pouring model predict the pour that is observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from typing_extensions import ClassVar

# %% the scale a drain is corrected by


@dataclass
class CalibratedDrainScale:
    """
    The factor a drain is scaled by, held at what makes the model predict the inflow
    that is measured.

    A drain model says how fast a tilted container pours. Where what arrives disagrees,
    correcting the level a controller steers by does not help: it goes on predicting the
    same arrival from the same tilt. Correcting the factor does, since the prediction it
    acts on then follows what has been seen.

    The factor is a free variable, so the compiled expressions read its current value
    every cycle and a correction reaches a controller that is already running.

    ..note:: Registering the factor is what lets the physics between cycles integrate
        the same drain: the registration gives the variable the resolve function that
        :meth:`~krrood.symbolic_math.symbolic_math.Scalar.evaluate` looks for. It has to
        be registered before anything evaluates the drain, and every controller sharing
        the world has to share the registration.
    """

    MINIMUM_PREDICTED_INFLOW: ClassVar[float] = 1e-6
    """
    Below this the model predicts nothing at any factor, so what is measured says
    nothing about what the factor should be.
    """

    scale: FloatVariable
    """
    The drain's factor.
    """

    inflow: Scalar
    """
    The model's predicted inflow, as an expression of that factor.
    """

    variables: FloatVariableData
    """
    Where the factor's value is read from and written to.
    """

    smoothing: float = 0.5
    """
    How far towards each new observation the factor moves, in ``[0, 1]``.
    """

    maximum: float = 1.0
    """
    The largest factor the drain is scaled by, so a correction cannot make the model
    pour faster than it was calibrated to.
    """

    @property
    def value(self) -> float:
        """
        :return: The factor the drain is currently scaled by.
        """
        return float(self.variables.get_value(self.scale))

    def predicted(self) -> float:
        """
        :return: The inflow the model predicts at the current factor.
        """
        return float(self.inflow.evaluate()[0])

    def uncorrected(self) -> float:
        """
        :return: The inflow the model predicts with the factor at one, which is the
            model as it was calibrated.
        """
        return float(
            self.inflow.substitute([self.scale], [sm.Scalar(1.0)]).evaluate()[0]
        )

    def calibrate(self, measured_inflow: float) -> float:
        """
        Move the factor towards what would make the model predict the measured inflow.

        A model predicting nothing whatever the factor leaves it where it is: that the
        pour has not started says nothing about how fast it will run once it has.

        :param measured_inflow: The inflow that was measured, per second.
        :return: The factor the drain is now scaled by.
        """
        uncorrected = self.uncorrected()
        if abs(uncorrected) < self.MINIMUM_PREDICTED_INFLOW:
            return self.value
        target = min(self.maximum, max(0.0, measured_inflow / uncorrected))
        corrected = (1.0 - self.smoothing) * self.value + self.smoothing * target
        self.variables.set_value(self.scale, corrected)
        return corrected
