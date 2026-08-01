from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Dict, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription

# %% result


@dataclass
class ValidationResult:
    """
    The outcome of performing a capability in the simulated robot.
    """

    status: TaskStatus
    """
    The terminal status of the performed capability.
    """

    reason: Optional[str]
    """
    The failure description when the capability did not succeed, otherwise ``None``.
    """

    @property
    def succeeded(self) -> bool:
        """
        :return: Whether the capability reached the succeeded status.
        """
        return self.status == TaskStatus.SUCCEEDED

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: A JSON-serializable view of the outcome.
        """
        return {
            "status": self.status.name,
            "succeeded": self.succeeded,
            "reason": self.reason,
        }


# %% validator


@dataclass
class SimulationValidator:
    """
    Performs a capability in the simulated (belief-state) robot and reports its outcome,
    so an agent can design a capability and see whether it works before using it.
    """

    def validate(self, action: ActionDescription, context: Context) -> ValidationResult:
        """
        Perform ``action`` once in the simulated robot and report whether it succeeded.

        :param action: The action to perform.
        :param context: The world and robot to perform it against.
        :return: The outcome of the performance.
        """
        return self.validate_plan(execute_single(action, context=context))

    def validate_plan(self, node: PlanNode) -> ValidationResult:
        """
        Perform a plan once in the simulated robot and report whether it succeeded.

        A plan failure is captured and reported rather than propagated, so a failed plan
        yields a result the agent can act on.

        :param node: The root node of the plan to perform.
        :return: The outcome of the performance.
        """
        with simulated_robot:
            try:
                node.perform()
            except PlanFailure as failure:
                return ValidationResult(TaskStatus.FAILED, str(failure))
        return ValidationResult(node.status, _reason_of(node))


def _reason_of(node: Any) -> Optional[str]:
    """
    :param node: The performed plan node.
    :return: The node's failure description, or ``None`` when it holds none.
    """
    if node.reason is None:
        return None
    return str(node.reason)
