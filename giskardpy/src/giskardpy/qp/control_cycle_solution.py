"""
Result of solving the control quadratic program of a single control cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from giskardpy.qp.qp_data import QPData


@dataclass(eq=False)
class ControlCycleSolution:
    """
    The commands of one control cycle together with the quadratic program they came
    from.

    Carrying the problem alongside its solution lets diagnostics reconstruct why the
    optimizer chose these commands without solving or evaluating anything a second time.
    """

    control_commands: np.ndarray
    """
    The command of every degree of freedom of the world, in world state order.
    """

    qp_data: QPData
    """
    The numeric quadratic program that was solved, before its rows and columns were
    filtered.
    """

    decision_variables: np.ndarray
    """
    The raw solver solution, in the filtered decision-variable layout.
    """
