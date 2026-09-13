"""
Derived views on a :class:`~giskardpy.qp.control_cycle_recording.ControlCycleRecording`
that explain which constraint moved which degree of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from giskardpy.qp.control_cycle_recording import ControlCycleRecording


@dataclass(eq=False)
class ControlCycleAnalysis:
    """
    Answers what each constraint demanded of each degree of freedom, and where
    constraints demanded opposite things.

    All quantities are derived from the recording alone, so the same analysis runs on a
    file copied off the robot.
    """

    recording: ControlCycleRecording
    """
    The recorded motion being analysed.
    """

    @cached_property
    def achieved_changes(self) -> np.ndarray:
        """
        How much the commanded velocities change each constraint expression over the
        horizon, shaped ``(cycles, rows)``.
        """
        return self.recording.row_contributions.sum(axis=2)

    @cached_property
    def bound_violations(self) -> np.ndarray:
        """
        How far each constraint stays outside its bounds, shaped ``(cycles, rows)``.

        Zero while a row is satisfied, so a nonzero value marks a constraint the
        optimizer could not serve.
        """
        below = self.recording.row_lower_bounds - self.achieved_changes
        above = self.achieved_changes - self.recording.row_upper_bounds
        return np.maximum(np.maximum(below, above), 0.0)

    @cached_property
    def slack_costs(self) -> np.ndarray:
        """
        What each constraint contributes to the optimizer's cost, shaped ``(cycles,
        rows)``.

        This is the term the optimizer actually trades off, so the largest entries name
        the constraints that shaped the motion.  Rows the solver dropped are ``nan``.
        """
        return self.recording.row_weights * self.recording.row_slacks**2

    @cached_property
    def is_active(self) -> np.ndarray:
        """
        Whether each constraint could influence the motion, shaped ``(cycles, rows)``.
        """
        return self.recording.row_weights > 0

    def horizon_sensitivities(self, cycle_index: int) -> np.ndarray:
        """
        How much each constraint expression changes per unit of velocity held over the
        whole horizon, shaped ``(rows, degrees_of_freedom)``.

        This is the comparable view of the constraint gradients: strategies spread a
        constraint over the horizon differently, so a single block says something
        different for each of them while the sum does not.

        :param cycle_index: The control cycle to look at.
        """
        return self.recording.row_sensitivities[cycle_index]

    def degree_of_freedom_contributions(self, cycle_index: int) -> np.ndarray:
        """
        How much each degree of freedom contributes to each constraint's achieved
        change, shaped ``(rows, degrees_of_freedom)``.

        The row sums are :attr:`achieved_changes`, so a large entry names the joint a
        constraint is using to get what it wants.

        :param cycle_index: The control cycle to look at.
        """
        return self.recording.row_contributions[cycle_index]

    def conflict_matrix(self, cycle_index: int) -> np.ndarray:
        """
        Cosine similarity between the constraint gradients, shaped ``(rows, rows)``.

        A value near ``-1`` means two constraints pull the same degrees of freedom in
        opposite directions; near ``1`` means they ask for the same motion.  Rows with a
        vanishing gradient are reported as unrelated to everything.

        :param cycle_index: The control cycle to look at.
        """
        sensitivities = self.horizon_sensitivities(cycle_index)
        norms = np.linalg.norm(sensitivities, axis=1)
        directions = np.zeros_like(sensitivities)
        has_gradient = norms > 0
        directions[has_gradient] = (
            sensitivities[has_gradient] / norms[has_gradient, None]
        )
        return directions @ directions.T

    def velocity_saturation(self, cycle_index: int) -> np.ndarray:
        """
        How much of its velocity limit each degree of freedom uses, shaped
        ``(degrees_of_freedom,)``.

        A value near one means the optimizer ran out of joint speed, which disturbs
        every constraint at once and looks like a conflict without being one.

        :param cycle_index: The control cycle to look at.
        """
        velocities = self.recording.velocities[cycle_index]
        limits = np.maximum(
            np.abs(self.recording.velocity_lower_limits[cycle_index]),
            np.abs(self.recording.velocity_upper_limits[cycle_index]),
        )
        saturation = np.zeros_like(velocities)
        is_limited = limits > 0
        saturation[is_limited] = np.abs(velocities[is_limited]) / limits[is_limited]
        return saturation

    def most_conflicting_rows(self, cycle_index: int) -> tuple[int, int]:
        """
        The pair of constraints whose gradients point most directly against each other.

        :param cycle_index: The control cycle to look at.
        :return: The indices of the two rows, both zero when there are fewer than two
            rows.
        """
        conflicts = self.conflict_matrix(cycle_index)
        if conflicts.shape[0] < 2:
            return 0, 0
        upper_triangle = np.triu(np.ones_like(conflicts, dtype=bool), k=1)
        masked = np.where(upper_triangle, conflicts, np.inf)
        flat_index = int(np.argmin(masked))
        first_row, second_row = np.unravel_index(flat_index, conflicts.shape)
        return int(first_row), int(second_row)
