"""
Exceptions raised while building and solving the quadratic program.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Type

from giskardpy.data_types.exceptions import GiskardException, DontPrintStackTrace
from giskardpy.qp.qp_data import QPDataExplicit

if TYPE_CHECKING:
    from giskardpy.qp.constraint import GiskardConstraint
    from giskardpy.qp.qp_data import QPData


@dataclass
class QPSolverException(GiskardException):
    """
    Base class for errors raised by the QP solvers.
    """


@dataclass
class SolverReturnedFailureError(QPSolverException):
    """
    Raised when a QP solver returns a non-optimal status.
    """

    solver_status: str
    """
    The solver-specific status describing the failure.
    """

    def error_message(self) -> str:
        return f"QP solver failed with status: {self.solver_status}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InfeasibleException(QPSolverException):
    """
    Raised when the QP has no feasible solution.
    """

    solver_status: str = ""
    """
    The solver-specific status describing the infeasibility.
    """

    def error_message(self) -> str:
        return f"QP is infeasible. Solver status: {self.solver_status}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class VelocityLimitUnreachableException(QPSolverException):
    """
    Raised when a degree of freedom cannot reach its velocity limit within the
    prediction horizon.
    """

    degree_of_freedom_name: str
    """
    The name of the degree of freedom that cannot reach its velocity limit.
    """

    velocity_limit: float
    """
    The velocity limit that cannot be reached.
    """

    prediction_horizon: int
    """
    The prediction horizon used by the QP controller.
    """

    jerk_limit: float
    """
    The jerk limit of the degree of freedom.
    """

    model_predictive_control_time_step: float
    """
    The time step of the model predictive controller.
    """

    max_reachable_velocity: float
    """
    The maximum velocity that is reachable given the limits and prediction horizon.
    """

    def error_message(self) -> str:
        return (
            f'Free variable "{self.degree_of_freedom_name}" can\'t reach velocity limit of '
            f'"{self.velocity_limit}". Maximum reachable with prediction horizon = '
            f'"{self.prediction_horizon}", jerk limit = "{self.jerk_limit}" and dt = '
            f'"{self.model_predictive_control_time_step}" is "{self.max_reachable_velocity}".'
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class OutOfJointLimitsException(InfeasibleException):
    """
    Raised when a degree of freedom is outside its position limits and cannot recover.
    """

    def error_message(self) -> str:
        return "A degree of freedom is outside its position limits and cannot recover."


@dataclass
class HardConstraintsViolatedException(InfeasibleException):
    """
    Raised when hard constraints cannot be satisfied.
    """

    def error_message(self) -> str:
        return "Hard constraints cannot be satisfied."


@dataclass
class EmptyProblemException(InfeasibleException, DontPrintStackTrace):
    """
    Raised when the QP problem has no free variables.
    """

    def error_message(self) -> str:
        return "Empty QP problem."


@dataclass
class MismatchedLimitLengthsError(GiskardException):
    """
    Raised when the bounds, weights, and names of a DirectLimits do not all share the
    same length.
    """

    field_lengths: dict[str, int]
    """
    The length of each DirectLimits field, keyed by field name.
    """

    def error_message(self) -> str:
        return f"All DirectLimits fields must have the same length, got {self.field_lengths}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ConstraintTypeMismatchError(QPSolverException):
    """
    Raised when an enforcement strategy receives a constraint of the wrong type for the
    requested bounds.
    """

    strategy_name: str
    """
    The name of the enforcement strategy that received the constraint.
    """

    expected_type: Type[GiskardConstraint]
    """
    The constraint type the strategy expected.
    """

    actual_type: Type[GiskardConstraint]
    """
    The constraint type that was actually received.
    """

    constraint_name: str
    """
    The name of the offending constraint.
    """

    def error_message(self) -> str:
        return (
            f"{self.strategy_name} expected constraints of type {self.expected_type.__name__}, "
            f"but got {self.actual_type.__name__} for constraint {self.constraint_name!r}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MultipleTerminalStateConstraintsError(QPSolverException):
    """
    Raised when more than one terminal-state prediction constraint is compiled into a
    single strategy block, which the single-row terminal prediction cannot represent.
    """

    constraint_names: list[str]
    """
    The names of the terminal-state constraints that were grouped together.
    """

    def error_message(self) -> str:
        return (
            f"TerminalStatePredictionStrategy supports exactly one constraint per block, "
            f"but received {len(self.constraint_names)}: {self.constraint_names}."
        )

    def suggest_correction(self) -> str:
        return "use at most one terminal-state task (e.g. one fill goal) per motion statechart."


@dataclass
class NoFactoryForQPDataTypeError(QPSolverException):
    """
    Raised when no registered factory handles the requested QPData type.
    """

    qp_data_type: Type[QPData]
    """
    The QPData type for which no factory is registered.
    """

    def error_message(self) -> str:
        return (
            f"No QPDataFactory registered for QPData type {self.qp_data_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return ""


# %% control cycle recording


@dataclass
class UnrecordableQPDataFormatError(QPSolverException):
    """
    Raised when control cycles are recorded from a quadratic program whose format keeps
    no separate equality and inequality blocks.
    """

    qp_data_type: Type[QPData]
    """
    The QPData type that was handed to the recorder.
    """

    def error_message(self) -> str:
        return (
            f"Control cycles can only be recorded from {QPDataExplicit.__name__}, "
            f"but the solver produced {self.qp_data_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            "configure a solver that uses the explicit format, for example "
            "QPSolverPIQP, or turn the control cycle recording off."
        )


@dataclass
class EmptyControlCycleRecordingError(QPSolverException):
    """
    Raised when a recording is built from a recorder that has not seen a control cycle.
    """

    def error_message(self) -> str:
        return "No control cycle was recorded."

    def suggest_correction(self) -> str:
        return (
            "attach the recorder before compiling the motion statechart and make sure "
            "the motion ticks at least once."
        )


@dataclass
class UnknownRecordingFormatVersionError(QPSolverException):
    """
    Raised when a stored recording was written by an incompatible format version.
    """

    file_path: str
    """
    The recording that was read.
    """

    found_version: int
    """
    The format version the file declares.
    """

    expected_version: int
    """
    The format version this code writes and reads.
    """

    def error_message(self) -> str:
        return (
            f"{self.file_path} declares control cycle recording format version "
            f"{self.found_version}, but version {self.expected_version} was expected."
        )

    def suggest_correction(self) -> str:
        return "record the motion again with the current version."
