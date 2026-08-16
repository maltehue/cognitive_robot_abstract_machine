"""
Runs the two-theory knowledge-servoing transfer and writes its figures.

Produces the Gantt chart of which constraint regimes were active over the motion, and
the decision transcripts explaining, in each theory's own vocabulary, why they turned
over when they did.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.knowledge_servoing.demonstration import (
    REQUESTED_FILL_LEVEL,
    build_transfer_demonstration,
)
from experiments.knowledge_servoing.scenario import build_transfer_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("."),
        help="Where to write the Gantt chart and transcripts.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Publish the run's world to RViz while it executes.",
    )
    parser.add_argument(
        "--requested-fill-level",
        type=float,
        default=REQUESTED_FILL_LEVEL,
        help="Fill level the transfer theory is asked to reach.",
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    scenario = build_transfer_scenario()
    visualization = None
    if arguments.visualize:
        from experiments.knowledge_servoing.visualization import WorldVisualization

        visualization = WorldVisualization.attach(scenario.world)
    demonstration = build_transfer_demonstration(
        scenario, requested_fill_level=arguments.requested_fill_level
    )
    demonstration.run()
    if visualization is not None:
        visualization.close()

    gantt_path = arguments.output_directory / "knowledge_servoing_gantt.pdf"
    demonstration.plot_gantt_chart(str(gantt_path))

    transcript_path = arguments.output_directory / "knowledge_servoing_transcript.txt"
    transcript_path.write_text(
        "substance transfer\n"
        f"{demonstration.transfer_transcript}\n\n"
        "contextual safety\n"
        f"{demonstration.safety_transcript}\n"
    )

    print(f"receiving cup fill level: {scenario.receiving_cup.fill_level:.4f}")
    print(f"requested: {arguments.requested_fill_level}")
    print(f"control cycles: {demonstration.executor.control_cycles}")
    print(f"gantt chart: {gantt_path}")
    print(f"transcript: {transcript_path}")
    print()
    print("substance transfer")
    print(demonstration.transfer_transcript)
    print("contextual safety")
    print(demonstration.safety_transcript)


if __name__ == "__main__":
    main()
