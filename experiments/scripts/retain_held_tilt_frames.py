"""
Write compact wall caches, cut at the source's held tilt, from caches of whole pours.
"""

import argparse
from pathlib import Path

from experiments.real_pouring.wall_fill_validation import (
    HeldTiltFrames,
    WallFillArtifact,
)

# %% command-line entry point


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("whole", type=Path, help="Directory of whole-pour caches.")
    parser.add_argument(
        "compact", type=Path, help="Directory receiving the cut caches."
    )
    parser.add_argument("recordings", nargs="+", help="Recording names to cut.")
    arguments = parser.parse_args()
    for recording in arguments.recordings:
        HeldTiltFrames(
            arguments.whole / (recording + WallFillArtifact.CACHE_SUFFIX)
        ).write(arguments.compact / (recording + WallFillArtifact.CACHE_SUFFIX))
