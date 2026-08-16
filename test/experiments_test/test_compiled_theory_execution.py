"""
A compiled specification driving the real transfer, end to end.

The compiler's golden test shows a specification concludes what the hand-written theory
concludes; this one shows the compiled artifact *executes*: assembled into a chart by
its own declarations, run in process against giskard, reaching the goal its parameter
rule supplies and terminating through its own conclusions. This is the synthesis target
proven executable — what remains for a synthesizer is producing the specification, not
making it runnable.
"""

from __future__ import annotations

import pytest

from experiments.knowledge_servoing.constraint_factories import build_transfer_catalog
from experiments.knowledge_servoing.demonstration import (
    FILL_LEVEL_TOLERANCE,
    REQUESTED_FILL_LEVEL,
)
from experiments.knowledge_servoing.scenario import build_transfer_scenario
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.knowledge_servoing.chart_assembler import (
    PluggedTheory,
    TheoryChartAssembler,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    GENERIC_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_compiler import (
    TheoryCompiler,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_specification import (
    TheorySpecification,
)
from semantic_digital_twin.reasoning.substance_transfer import (
    TransferSituationGrounding,
)
from semantic_digital_twin.reasoning.substance_transfer.declarations import (
    TRANSFER_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.substance_transfer.situation import (
    TransferSituation,
)
from semantic_digital_twin.utils import tracy_installed

from ..semantic_digital_twin_test.test_reasoning.test_theory_compilation import (
    TRANSFER_SPECIFICATION,
)


@pytest.fixture(scope="module")
def completed_compiled_run():
    """
    Compiles the reference specification, assembles its chart, and runs it to the end.
    """
    if not tracy_installed():
        pytest.skip("Tracy not installed")
    theory = TheoryCompiler(
        declaration_kinds={**GENERIC_DECLARATION_KINDS, **TRANSFER_DECLARATION_KINDS},
        situation_type=TransferSituation,
    ).compile(TheorySpecification.from_json(TRANSFER_SPECIFICATION))

    scenario = build_transfer_scenario()
    statechart = MotionStatechart()
    assembler = TheoryChartAssembler(
        catalog=build_transfer_catalog(), world=scenario.world
    )
    [assembled] = assembler.assemble(
        [
            PluggedTheory(
                name="compiled_transfer",
                theory=theory,
                grounding=TransferSituationGrounding(
                    source=scenario.source_cup,
                    receiver=scenario.receiving_cup,
                    requested_fill_level=REQUESTED_FILL_LEVEL,
                    fill_level_tolerance=FILL_LEVEL_TOLERANCE,
                ),
            )
        ],
        statechart,
    )
    statechart.add_node(EndMotion.when_true(assembled.monitors["return_upright"]))

    executor = Executor(
        MotionStatechartContext(world=scenario.world),
        pacer=SimulationPacer(real_time_factor=1),
    )
    executor.compile(motion_statechart=statechart)
    ended = False
    for _ in range(2000):
        executor.tick()
        if executor.motion_statechart.is_end_motion():
            ended = True
            break
    return scenario, ended


class TestCompiledTheoryExecution:
    """
    Whether a theory compiled from data is executable, not merely inference-equivalent.
    """

    def test_the_run_terminates_through_the_theory_conclusions(
        self, completed_compiled_run
    ):
        _scenario, ended = completed_compiled_run
        assert ended

    def test_the_receiver_reaches_the_specification_supplied_goal(
        self, completed_compiled_run
    ):
        scenario, _ended = completed_compiled_run
        assert scenario.receiving_cup.fill_level == pytest.approx(
            REQUESTED_FILL_LEVEL, abs=FILL_LEVEL_TOLERANCE
        )
