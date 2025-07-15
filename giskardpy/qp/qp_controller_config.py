from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Type

import numpy as np

from giskardpy.data_types.data_types import Derivatives
from giskardpy.data_types.exceptions import QPSolverException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_formulation import QPFormulation
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver
from giskardpy.utils.utils import get_all_classes_in_module
from semantic_world.prefixed_name import PrefixedName


available_solvers: Dict[SupportedQPSolver, Type[QPSolver]] = {}


def detect_solvers():
    global available_solvers
    qp_solver_class: Type[QPSolver]
    for qp_solver_name in SupportedQPSolver:
        module_name = f'giskardpy.qp.solvers.qp_solver_{qp_solver_name.name}'
        try:
            qp_solver_class = list(get_all_classes_in_module(module_name, QPSolver).items())[0][1]
            available_solvers[qp_solver_name] = qp_solver_class
        except Exception:
            continue
    solver_names = [solver_name.name for solver_name in available_solvers.keys()]
    print(f'Found these qp solvers: {solver_names}')


detect_solvers()

@dataclass
class QPControllerConfig:
    dof_weights: Dict[PrefixedName, Dict[Derivatives, float]] = field(
        default_factory=lambda: defaultdict(lambda: {Derivatives.velocity: 0.01,
                                                     Derivatives.acceleration: np.inf,
                                                     Derivatives.jerk: None}))
    max_derivative: Derivatives = field(default=Derivatives.jerk)
    qp_solver_id: Optional[SupportedQPSolver] = field(default=None)
    qp_solver_class: Type[QPSolver] = field(init=False)
    prediction_horizon: int = field(default=7)
    mpc_dt: float = field(default=0.0125)
    control_dt: Optional[float] = field(default=None)
    max_trajectory_length: Optional[float] = field(default=30)
    horizon_weight_gain_scalar: float = 0.1
    qp_formulation: Optional[QPFormulation] = field(default_factory=QPFormulation)
    retries_with_relaxed_constraints: int = field(default=5)
    added_slack: float = field(default=100)
    weight_factor: float = field(default=100)
    verbose: bool = field(default=True)

    def __post_init__(self):
        """
        :param qp_solver: if not set, Giskard will search for the fasted installed solver.
        :param prediction_horizon: Giskard uses MPC and this is the length of the horizon. You usually don't need to change this.
        :param mpc_dt: time (s) difference between commands in the MPC horizon.
        :param max_trajectory_length: Giskard will stop planning/controlling the robot until this amount of s has passed.
                                      This is disabled if set to None.
        :param retries_with_relaxed_constraints: don't change, only for the pros.
        :param added_slack: don't change, only for the pros.
        :param weight_factor: don't change, only for the pros.
        """
        if self.control_dt is None:
            self.control_dt = self.mpc_dt
        if not self.qp_formulation.is_mpc:
            self.prediction_horizon = 1
            self.max_derivative = Derivatives.velocity

        if self.prediction_horizon < 4:
            raise ValueError('prediction horizon must be >= 4.')
        self.__endless_mode = self.max_trajectory_length is None
        self.set_qp_solver(self.qp_solver_id)
        self.init_qp_controller()

    def set_qp_solver(self, solver_id: SupportedQPSolver) -> None:
        if solver_id is not None:
            self.qp_solver_class = available_solvers[solver_id]
        else:
            for solver_id in SupportedQPSolver:
                if solver_id in available_solvers:
                    self.qp_solver_class = available_solvers[solver_id]
                    break
            else:
                raise QPSolverException(f'No qp solver found')
            self.qp_solver_id = self.qp_solver_class.solver_id
        get_middleware().loginfo(f'QP Solver set to "{self.qp_solver_class.solver_id.name}"')

    def set_dof_weight(self, dof_name: PrefixedName, derivative: Derivatives, weight: float):
        """Set weight for a specific DOF derivative."""
        if dof_name not in self.dof_weights:
            self.dof_weights[dof_name] = {
                Derivatives.velocity: 0.01,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        self.dof_weights[dof_name][derivative] = weight

    def set_dof_weights(self, dof_name: PrefixedName, weight_map: Dict[Derivatives, float]):
        """Set multiple weights for a DOF."""
        if dof_name not in self.dof_weights:
            self.dof_weights[dof_name] = {
                Derivatives.velocity: 0.01,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None
            }
        self.dof_weights[dof_name].update(weight_map)

    def get_dof_weight(self, dof_name: PrefixedName, derivative: Derivatives) -> float:
        """Get weight for a specific DOF derivative."""
        return self.dof_weights[dof_name][derivative]

    def init_qp_controller(self):
        god_map.qp_controller = QPController(config=self)
