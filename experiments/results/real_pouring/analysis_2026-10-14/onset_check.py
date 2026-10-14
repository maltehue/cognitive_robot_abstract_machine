"""
The ρ check on bank replays: how far the repose angle the controller runs on moves
within a pour once the stream has shown, with the onset only in the onset hypotheses
against every hypothesis tracking it.

Usage: python onset_check.py replay_<...>.json ... A repose angle that stays within a
few degrees per material after the stream starts means the model carries the pour; one
that swings by tens of degrees means the bank is fitting the pour with a moving offset.
"""

import json, sys
from pathlib import Path

for path in map(Path, sys.argv[1:]):
    steps = json.load(open(path))["steps"]
    first = next((i for i, s in enumerate(steps) if s["stream_seen"]), None)
    if first is None:
        print(f"{path.stem}: no stream")
        continue
    before = (
        steps[first - 1]["controller_repose"]
        if first
        else steps[0]["controller_repose"]
    )
    after = [s["controller_repose"] for s in steps[first:]]
    means = [
        s["active_repose_mean"]
        for s in steps[first:]
        if s["active_repose_mean"] is not None
    ]
    actives = []
    for s in steps:
        if not actives or actives[-1][0] != s["active"]:
            actives.append((s["active"], s["time"]))
    print(
        f"{path.stem}: stream at {steps[first]['time']:.1f} s, {steps[first]['tilt']:.0f} deg; controller rho there {before:.1f} deg; "
        f"after it {min(after):.1f}..{max(after):.1f} (range {max(after) - min(after):.1f}), at the end {after[-1]:.1f}; "
        f"active hypothesis' mean rho {('%.1f..%.1f' % (min(means), max(means))) if means else 'not estimated'}; "
        f"active: " + " ".join(f"{h}@{t:.1f}" for h, t in actives)
    )
