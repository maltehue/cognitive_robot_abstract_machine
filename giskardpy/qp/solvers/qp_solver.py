from __future__ import annotations

from typing import TYPE_CHECKING, Type, List

import numpy as np

from giskardpy.qp.qp_data import QPData
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver

if TYPE_CHECKING:
    from giskardpy.qp.adapters.qp_adapter import GiskardToQPAdapter


class QPSolver:
    solver_id: SupportedQPSolver
    required_adapter_type: Type[GiskardToQPAdapter]

    def solver_call(self, qp_data: QPData) -> np.ndarray:
        raise NotImplementedError()

    def solver_call_batch(self, qps: List[QPData]) -> np.ndarray:
        raise NotImplementedError()

    def solver_call_explicit_interface(self, qp_data: QPData) -> np.ndarray:
        """
        min_x 0.5 x^T H x + g^T x
        s.t.  lb <= x <= ub     (box constraints)
                   Ex <= bE     (equality constraints)
            lbA <= Ax <= ubA    (lower/upper inequality constraints)
        """
        raise NotImplementedError()
