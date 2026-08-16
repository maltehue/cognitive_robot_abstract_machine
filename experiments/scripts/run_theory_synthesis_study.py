"""
Runs the theory-synthesis sentence study live and archives every artifact.

One fresh conversation and one fresh analytic world per sentence. Nothing here judges
whether the run did what the sentence meant — the archived transcripts, specifications
and outcomes are for the person who wrote the sentences.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from experiments.theory_synthesis.generator import ClaudeCommandLineGenerator
from experiments.theory_synthesis.study import SynthesisStudy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("theory_synthesis_study"),
        help="Where each sentence's artifacts are written.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Publish each run's world to RViz while it executes.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier to pin, or the command line's default when omitted.",
    )
    arguments = parser.parse_args()

    study = SynthesisStudy(
        generator_factory=lambda: ClaudeCommandLineGenerator(model=arguments.model),
        visualize=arguments.visualize,
    )
    for outcome in study.run():
        sentence_directory = arguments.output_directory / outcome.sentence.identifier
        sentence_directory.mkdir(parents=True, exist_ok=True)
        (sentence_directory / "instruction.txt").write_text(
            outcome.sentence.instruction + "\n"
        )
        if outcome.response_text is not None:
            (sentence_directory / "response.txt").write_text(outcome.response_text)
        (sentence_directory / "outcome.json").write_text(
            json.dumps(dataclasses.asdict(outcome), indent=2, default=str)
        )
        print(f"\n=== {outcome.sentence.identifier}")
        print(f"    {outcome.sentence.instruction}")
        if outcome.rejection:
            print(f"    REJECTED: {outcome.rejection}")
        elif outcome.declined:
            print("    DECLINED: the model answered with an empty specification")
        else:
            print(f"    constraints: {', '.join(outcome.constraint_identifiers)}")
            print(
                f"    executed: ended_by_theory={outcome.ended_by_theory} "
                f"aborted={outcome.aborted_reason} "
                f"fill={outcome.final_fill_level:.4f} "
                f"cycles={outcome.control_cycles:.0f}"
            )
            print("    transcript:")
            for line in outcome.decision_transcript.splitlines():
                print(f"      {line}")


if __name__ == "__main__":
    main()
