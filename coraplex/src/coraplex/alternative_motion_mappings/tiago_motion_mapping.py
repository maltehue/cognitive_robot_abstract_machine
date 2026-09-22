from dataclasses import dataclass
from typing_extensions import ClassVar
from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans import MoveMotion
from coraplex.robot_plans.motions.base import AlternativeMotion
from semantic_digital_twin.robots.tiago import Tiago


@dataclass
class TiagoMoveSim(MoveMotion, AlternativeMotion[Tiago]):
    """
    Navigate Tiago through free space using its differential drive controller.
    """

    execution_type: ClassVar[ExecutionType] = ExecutionType.SIMULATED
    """
    Execution environment selecting this navigation mapping.
    """
