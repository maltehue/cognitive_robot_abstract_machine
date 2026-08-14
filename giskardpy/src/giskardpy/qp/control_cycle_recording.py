"""
Recording of what every control cycle asked of the quadratic program and what it got
back, so a finished motion can be inspected constraint by constraint afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from typing_extensions import Self

from giskardpy.qp.control_cycle_solution import ControlCycleSolution
from giskardpy.qp.exceptions import (
    EmptyControlCycleRecordingError,
    UnknownRecordingFormatVersionError,
    UnrecordableQPDataFormatError,
)
from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.qp_data_symbolic import QPDataSymbolic
from giskardpy.utils.utils import create_path

RECORDING_FORMAT_VERSION = 3
"""
Version of the stored recording layout, raised whenever stored fields change meaning.
"""

NODE_NAME_SEPARATOR = "/"
"""
Separates the parts of a constraint row name.
"""

NODE_INDEX_SEPARATOR = "#"
"""
Separates a motion statechart node's name from its index in its unique name.
"""

# %% problem layout


@dataclass
class ConstraintProblemStructure:
    """
    Static layout of a compiled control quadratic program.

    Names the task constraint rows and the degrees of freedom their coefficients refer
    to, so a recording can be read back without the world or the motion statechart that
    produced it.
    """

    row_names: list[str]
    """
    Name of every recorded task constraint row, equality rows first.
    """

    degree_of_freedom_names: list[str]
    """
    Name of every degree of freedom, in decision-variable column order.
    """

    number_of_equality_rows: int
    """
    How many of the rows are equality rows; the remaining ones are inequality rows.
    """

    control_horizon: int
    """
    Number of velocity blocks a constraint row spreads over.
    """

    model_predictive_control_time_step: float
    """
    Seconds between two steps of the prediction horizon.
    """

    @classmethod
    def from_symbolic_problem(cls, qp_data: QPDataSymbolic) -> Self:
        """
        Reads the layout of the task constraint rows off a compiled symbolic problem.

        The system dynamics rows are left out: they encode the integration scheme rather
        than anything a task asked for.
        """
        number_of_equality_rows = qp_data.number_equality_slack_variables
        equality_names = qp_data.equality_constraint_names[
            len(qp_data.equality_constraint_names) - number_of_equality_rows :
        ]
        configuration = qp_data.qp_controller_config
        return cls(
            row_names=list(equality_names) + list(qp_data.inequality_constraint_names),
            degree_of_freedom_names=[
                str(dof.name) for dof in qp_data.degrees_of_freedom
            ],
            number_of_equality_rows=number_of_equality_rows,
            control_horizon=configuration.control_horizon,
            model_predictive_control_time_step=configuration.model_predictive_control_time_step,
        )

    @property
    def node_names(self) -> list[str]:
        """
        The motion statechart node every row belongs to, one entry per row.

        Rows are named after the node that created them, but not always in first place:
        a strategy that spreads a constraint over the horizon prefixes the time step.
        The node is therefore found by its unique name rather than by position.
        """
        return [self._node_name_of(row_name) for row_name in self.row_names]

    @staticmethod
    def _node_name_of(row_name: str) -> str:
        """
        Picks the unique name of the node that owns a constraint row.
        """
        segments = row_name.split(NODE_NAME_SEPARATOR)
        for segment in segments:
            if NODE_INDEX_SEPARATOR in segment:
                return segment
        return segments[0]

    @property
    def row_is_equality(self) -> np.ndarray:
        """
        Whether each row is an equality row, one entry per row.
        """
        is_equality = np.zeros(self.number_of_rows, dtype=bool)
        is_equality[: self.number_of_equality_rows] = True
        return is_equality

    @property
    def number_of_rows(self) -> int:
        """
        How many task constraint rows the problem has.
        """
        return len(self.row_names)

    @property
    def number_of_degrees_of_freedom(self) -> int:
        """
        How many degrees of freedom the optimizer may move.
        """
        return len(self.degree_of_freedom_names)

    @property
    def number_of_velocity_columns(self) -> int:
        """
        How many decision-variable columns hold velocities.
        """
        return self.control_horizon * self.number_of_degrees_of_freedom

    def to_json(self) -> dict:
        """
        The layout as plain data, for storing next to the recorded arrays.
        """
        return {
            "row_names": self.row_names,
            "degree_of_freedom_names": self.degree_of_freedom_names,
            "number_of_equality_rows": self.number_of_equality_rows,
            "control_horizon": self.control_horizon,
            "model_predictive_control_time_step": self.model_predictive_control_time_step,
        }

    @classmethod
    def from_json(cls, data: dict) -> Self:
        """
        Rebuilds a layout from its stored plain data.
        """
        return cls(
            row_names=list(data["row_names"]),
            degree_of_freedom_names=list(data["degree_of_freedom_names"]),
            number_of_equality_rows=int(data["number_of_equality_rows"]),
            control_horizon=int(data["control_horizon"]),
            model_predictive_control_time_step=float(
                data["model_predictive_control_time_step"]
            ),
        )


# %% single cycle


@dataclass(eq=False)
class RecordedControlCycle:
    """
    What one solve of the control quadratic program asked for and achieved.
    """

    time: float
    """
    Simulation time of the cycle in seconds.
    """

    row_sensitivities: np.ndarray
    """
    How much each row's expression changes per unit of velocity held over the whole
    horizon, shaped ``(rows, degrees_of_freedom)``.
    """

    row_contributions: np.ndarray
    """
    How much each degree of freedom's commanded motion changes each row's expression,
    shaped ``(rows, degrees_of_freedom)``.
    """

    row_lower_bounds: np.ndarray
    """
    Lower bound of every row; equal to the upper bound for equality rows.
    """

    row_upper_bounds: np.ndarray
    """
    Upper bound of every row; equal to the lower bound for equality rows.
    """

    row_weights: np.ndarray
    """
    Normalized cost of violating each row; zero while the owning node is not running.
    """

    row_slacks: np.ndarray
    """
    How far each row was violated, ``nan`` for rows the solver dropped.
    """

    velocities: np.ndarray
    """
    Commanded velocity of every degree of freedom in the first velocity block, shaped
    ``(degrees_of_freedom,)``.
    """

    velocity_lower_limits: np.ndarray
    """
    Lower box limit of the first velocity block, shaped like :attr:`velocities`.
    """

    velocity_upper_limits: np.ndarray
    """
    Upper box limit of the first velocity block, shaped like :attr:`velocities`.
    """

    world_positions: np.ndarray
    """
    Position of every degree of freedom of the world, shaped ``(world degrees of
    freedom,)``.

    Covers the whole world rather than the degrees of freedom the optimizer may move, so
    a replay can put the objects the robot interacts with back where they were too.
    """


# %% recording


@dataclass(eq=False)
class ControlCycleRecording:
    """
    Time series of every control cycle of one motion, stacked along the first axis.

    Stores what the optimizer was asked for rather than plots of it, so the same
    recording can answer questions that were not asked while the robot was moving.
    """

    structure: ConstraintProblemStructure
    """
    Layout the recorded rows and columns refer to.
    """

    times: np.ndarray
    """
    Simulation time of every cycle in seconds.
    """

    row_sensitivities: np.ndarray
    """
    Per-cycle constraint gradients, shaped ``(cycles, rows, degrees_of_freedom)``.
    """

    row_contributions: np.ndarray
    """
    Per-cycle share of each degree of freedom in each row's change, shaped ``(cycles,
    rows, degrees_of_freedom)``.
    """

    row_lower_bounds: np.ndarray
    """
    Per-cycle lower bound of every row, shaped ``(cycles, rows)``.
    """

    row_upper_bounds: np.ndarray
    """
    Per-cycle upper bound of every row, shaped ``(cycles, rows)``.
    """

    row_weights: np.ndarray
    """
    Per-cycle normalized violation cost of every row, shaped ``(cycles, rows)``.
    """

    row_slacks: np.ndarray
    """
    Per-cycle violation of every row, shaped ``(cycles, rows)``.
    """

    velocities: np.ndarray
    """
    Per-cycle commanded velocities of the first velocity block, shaped ``(cycles,
    degrees_of_freedom)``.
    """

    velocity_lower_limits: np.ndarray
    """
    Per-cycle lower velocity box limits, shaped like :attr:`velocities`.
    """

    velocity_upper_limits: np.ndarray
    """
    Per-cycle upper velocity box limits, shaped like :attr:`velocities`.
    """

    world_degree_of_freedom_ids: list[str]
    """
    Identifier of every degree of freedom of the world, in the column order of
    :attr:`world_positions`.
    """

    world_positions: np.ndarray
    """
    Per-cycle position of every degree of freedom of the world, shaped ``(cycles, world
    degrees of freedom)``.
    """

    @classmethod
    def from_cycles(
        cls,
        structure: ConstraintProblemStructure,
        world_degree_of_freedom_ids: list[str],
        cycles: list[RecordedControlCycle],
    ) -> Self:
        """
        Stacks the recorded cycles into one time series.

        :raises EmptyControlCycleRecordingError: if no cycle was recorded.
        """
        if not cycles:
            raise EmptyControlCycleRecordingError()
        return cls(
            structure=structure,
            times=np.array([cycle.time for cycle in cycles], dtype=float),
            row_sensitivities=np.stack([cycle.row_sensitivities for cycle in cycles]),
            row_contributions=np.stack([cycle.row_contributions for cycle in cycles]),
            row_lower_bounds=np.stack([cycle.row_lower_bounds for cycle in cycles]),
            row_upper_bounds=np.stack([cycle.row_upper_bounds for cycle in cycles]),
            row_weights=np.stack([cycle.row_weights for cycle in cycles]),
            row_slacks=np.stack([cycle.row_slacks for cycle in cycles]),
            velocities=np.stack([cycle.velocities for cycle in cycles]),
            velocity_lower_limits=np.stack(
                [cycle.velocity_lower_limits for cycle in cycles]
            ),
            velocity_upper_limits=np.stack(
                [cycle.velocity_upper_limits for cycle in cycles]
            ),
            world_degree_of_freedom_ids=list(world_degree_of_freedom_ids),
            world_positions=np.stack([cycle.world_positions for cycle in cycles]),
        )

    @property
    def number_of_cycles(self) -> int:
        """
        How many control cycles were recorded.
        """
        return len(self.times)

    def save(self, file_path: str) -> None:
        """
        Writes the recording to a compressed numpy archive.
        """
        create_path(file_path)
        metadata = json.dumps(
            {
                "format_version": RECORDING_FORMAT_VERSION,
                "structure": self.structure.to_json(),
                "world_degree_of_freedom_ids": self.world_degree_of_freedom_ids,
            }
        )
        np.savez_compressed(
            file_path,
            metadata=np.array(metadata),
            times=self.times,
            row_sensitivities=self.row_sensitivities,
            row_contributions=self.row_contributions,
            row_lower_bounds=self.row_lower_bounds,
            row_upper_bounds=self.row_upper_bounds,
            row_weights=self.row_weights,
            row_slacks=self.row_slacks,
            velocities=self.velocities,
            velocity_lower_limits=self.velocity_lower_limits,
            velocity_upper_limits=self.velocity_upper_limits,
            world_positions=self.world_positions,
        )

    @classmethod
    def load(cls, file_path: str) -> Self:
        """
        Reads a recording written by :meth:`save`.

        :raises UnknownRecordingFormatVersionError: if the file was written by an
            incompatible version.
        """
        with np.load(file_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            found_version = int(metadata["format_version"])
            if found_version != RECORDING_FORMAT_VERSION:
                raise UnknownRecordingFormatVersionError(
                    file_path=file_path,
                    found_version=found_version,
                    expected_version=RECORDING_FORMAT_VERSION,
                )
            return cls(
                structure=ConstraintProblemStructure.from_json(metadata["structure"]),
                times=archive["times"],
                row_sensitivities=archive["row_sensitivities"],
                row_contributions=archive["row_contributions"],
                row_lower_bounds=archive["row_lower_bounds"],
                row_upper_bounds=archive["row_upper_bounds"],
                row_weights=archive["row_weights"],
                row_slacks=archive["row_slacks"],
                velocities=archive["velocities"],
                velocity_lower_limits=archive["velocity_lower_limits"],
                velocity_upper_limits=archive["velocity_upper_limits"],
                world_degree_of_freedom_ids=list(
                    metadata["world_degree_of_freedom_ids"]
                ),
                world_positions=archive["world_positions"],
            )


# %% recorder


@dataclass(eq=False)
class ControlCycleRecorder:
    """
    Collects one :class:`RecordedControlCycle` per solve of the control quadratic
    program.

    Reading the numbers off the solved problem keeps the recording free of a second
    evaluation of the symbolic expressions, which would not fit into a control cycle of
    a real robot.
    """

    structure: ConstraintProblemStructure = field(init=False)
    """
    Layout of the problem being recorded, set by :meth:`reset`.
    """

    _cycles: list[RecordedControlCycle] = field(init=False, default_factory=list)
    """
    The cycles recorded since the last reset.
    """

    _degree_of_freedom_aggregator: sp.csc_matrix = field(init=False)
    """
    Sums the velocity columns of every horizon block onto their degree of freedom.
    """

    world_degree_of_freedom_ids: list[str] = field(init=False, default_factory=list)
    """
    Identifier of every degree of freedom of the world, in world state column order.
    """

    def reset(
        self, qp_data: QPDataSymbolic, world_degree_of_freedom_ids: list[str]
    ) -> None:
        """
        Prepares to record the given problem, discarding any previous cycles.

        :param qp_data: The compiled problem whose rows are recorded.
        :param world_degree_of_freedom_ids: Identifier of every degree of freedom of the
            world, in the column order of the positions handed to :meth:`record`.
        """
        self.structure = ConstraintProblemStructure.from_symbolic_problem(qp_data)
        self.world_degree_of_freedom_ids = list(world_degree_of_freedom_ids)
        self._cycles = []
        self._degree_of_freedom_aggregator = self._create_aggregator()

    def _create_aggregator(self) -> sp.csc_matrix:
        """
        Builds the matrix that folds the horizon out of the velocity columns.

        The columns run block by block and, within a block, degree of freedom by degree
        of freedom, so every column maps onto the degree of freedom of its position.
        """
        number_of_degrees_of_freedom = self.structure.number_of_degrees_of_freedom
        number_of_velocity_columns = self.structure.number_of_velocity_columns
        return sp.csc_matrix(
            (
                np.ones(number_of_velocity_columns),
                (
                    np.arange(number_of_velocity_columns),
                    np.tile(
                        np.arange(number_of_degrees_of_freedom),
                        self.structure.control_horizon,
                    ),
                ),
            ),
            shape=(number_of_velocity_columns, number_of_degrees_of_freedom),
        )

    @property
    def has_recorded_cycles(self) -> bool:
        """
        Whether at least one cycle was recorded since the last reset.
        """
        return bool(self._cycles)

    def record(
        self,
        time: float,
        solution: ControlCycleSolution,
        world_positions: np.ndarray,
    ) -> None:
        """
        Reads the rows and commands of one solved control cycle.

        :param time: Simulation time of the cycle in seconds.
        :param solution: The solved quadratic program of the cycle.
        :param world_positions: Position of every degree of freedom of the world.
        :raises UnrecordableQPDataFormatError: if the solver format keeps no separate
            equality and inequality blocks.
        """
        qp_data = solution.qp_data
        if not isinstance(qp_data, QPDataExplicit):
            raise UnrecordableQPDataFormatError(qp_data_type=type(qp_data))
        number_of_non_slack_variables = self._number_of_non_slack_variables(qp_data)
        padded_solution = self._pad_solution(qp_data, solution.decision_variables)
        commanded_velocities = padded_solution[
            : self.structure.number_of_velocity_columns
        ]
        number_of_degrees_of_freedom = self.structure.number_of_degrees_of_freedom
        equality_columns, inequality_columns = self._velocity_columns(qp_data)
        self._cycles.append(
            RecordedControlCycle(
                time=time,
                row_sensitivities=self._fold_horizon(
                    equality_columns, inequality_columns
                ),
                row_contributions=self._fold_horizon(
                    equality_columns.multiply(commanded_velocities[None, :]),
                    inequality_columns.multiply(commanded_velocities[None, :]),
                ),
                row_lower_bounds=self._row_bounds(
                    qp_data, qp_data.inequality_lower_bounds
                ),
                row_upper_bounds=self._row_bounds(
                    qp_data, qp_data.inequality_upper_bounds
                ),
                row_weights=np.array(
                    qp_data.quadratic_weights[number_of_non_slack_variables:]
                ),
                row_slacks=padded_solution[number_of_non_slack_variables:],
                velocities=commanded_velocities[:number_of_degrees_of_freedom].copy(),
                velocity_lower_limits=np.array(
                    qp_data.box_lower_constraints[:number_of_degrees_of_freedom]
                ),
                velocity_upper_limits=np.array(
                    qp_data.box_upper_constraints[:number_of_degrees_of_freedom]
                ),
                world_positions=np.array(world_positions),
            )
        )

    def build_recording(self) -> ControlCycleRecording:
        """
        Stacks everything recorded since the last reset into one recording.

        :raises EmptyControlCycleRecordingError: if no cycle was recorded.
        """
        if not self.has_recorded_cycles:
            raise EmptyControlCycleRecordingError()
        return ControlCycleRecording.from_cycles(
            self.structure, self.world_degree_of_freedom_ids, self._cycles
        )

    def _number_of_non_slack_variables(self, qp_data: QPDataExplicit) -> int:
        """
        How many decision-variable columns belong to the degrees of freedom.
        """
        return qp_data.quadratic_weights.shape[0] - qp_data.num_slack_variables

    def _pad_solution(
        self, qp_data: QPDataExplicit, decision_variables: np.ndarray
    ) -> np.ndarray:
        """
        Expands the filtered solver solution back into the full column layout.

        Columns the solver dropped because their slack was weightless become ``nan``,
        which marks the rows of nodes that were not running.
        """
        weight_filter = qp_data.quadratic_weights != 0
        weight_filter[: self._number_of_non_slack_variables(qp_data)] = True
        padded_solution = np.full(qp_data.quadratic_weights.shape[0], np.nan)
        padded_solution[weight_filter] = decision_variables
        return padded_solution

    def _velocity_columns(
        self, qp_data: QPDataExplicit
    ) -> tuple[sp.spmatrix, sp.spmatrix]:
        """
        Cuts the velocity columns of the task rows out of the constraint matrices.

        The columns come off first because the matrices are stored by column, which
        makes the row cut that follows run over a fraction of the entries.
        """
        number_of_velocity_columns = self.structure.number_of_velocity_columns
        return (
            self._task_row_block(
                qp_data.equality_matrix,
                number_of_task_rows=self.structure.number_of_equality_rows,
                number_of_columns=number_of_velocity_columns,
            ),
            self._task_row_block(
                qp_data.inequality_matrix,
                number_of_task_rows=qp_data.inequality_matrix.shape[0],
                number_of_columns=number_of_velocity_columns,
            ),
        )

    def _fold_horizon(
        self, equality_columns: sp.spmatrix, inequality_columns: sp.spmatrix
    ) -> np.ndarray:
        """
        Sums the horizon blocks of every task row onto their degrees of freedom.
        """
        return np.vstack(
            [
                self._fold_matrix(equality_columns),
                self._fold_matrix(inequality_columns),
            ]
        )

    def _fold_matrix(self, velocity_columns: sp.spmatrix) -> np.ndarray:
        """
        Sums the horizon blocks of every row of one matrix onto their degrees of
        freedom.
        """
        return np.asarray(
            (velocity_columns @ self._degree_of_freedom_aggregator).todense()
        )

    def _row_bounds(
        self, qp_data: QPDataExplicit, inequality_bounds: np.ndarray
    ) -> np.ndarray:
        """
        Puts the equality bounds in front of the given inequality bounds.

        Equality rows share one bound, so the same value serves as their lower and upper
        bound and both sides of a row can be read the same way.
        """
        equality_bounds = qp_data.equality_bounds[
            qp_data.equality_bounds.shape[0] - self.structure.number_of_equality_rows :
        ]
        return np.concatenate([equality_bounds, inequality_bounds])

    @staticmethod
    def _task_row_block(
        matrix: sp.csc_matrix | np.ndarray,
        number_of_task_rows: int,
        number_of_columns: int,
    ) -> sp.spmatrix:
        """
        Cuts the trailing task rows and leading columns out of a constraint matrix.

        Returns an empty block for a matrix without rows or columns, which is what an
        empty constraint block compiles to.
        """
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return sp.csr_matrix((0, number_of_columns))
        columns = sp.csr_matrix(matrix[:, :number_of_columns])
        return columns[columns.shape[0] - number_of_task_rows :]
