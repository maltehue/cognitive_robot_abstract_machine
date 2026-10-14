"""
Score the onset-tracking pours: the onset in its own hypotheses against every hypothesis
tracking it, pole camera alone, seeds 20-27.

Usage: python onset_tracking_analysis.py [trials folder name, default
trials_onset_tracking]
"""

import glob, json, math, sys
from pathlib import Path
import pandas
from experiments.pouring_model_verification.controller_comparison import (
    misplaced_grains,
    succeeded,
)
from experiments.pouring_model_verification.records import TrialRecord

R = Path(
    "/home/malte/libs/cognitive_robot_abstract_machine/experiments/results/pouring_model_verification"
)
HERE = Path(__file__).parent
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "trials_onset_tracking"


def perturbation_of(d):
    p = d["perturbation"]
    return (
        "nominal"
        if not p["parameter"]
        else f'{ {"repose_angle": "rho", "discharge_coefficient": "C_d", "outflow_rate_constant": "k"}[p["parameter"]]} x{p["factor"]:g}'
    )


rows = []
for path in sorted(glob.glob(str(R / FOLDER / "*"))):
    if not Path(path, "trial.json").exists():
        continue
    d = TrialRecord.read(Path(path)).description
    o = d["outcome"]
    rows.append(
        dict(
            mode="tracked" if "_tracked_" in d["name"] else "concept",
            source=d["source"],
            perturbation=perturbation_of(d),
            seed=d["seed"],
            delivered=o["delivered"],
            spilled=o["spilled"],
            misplaced=misplaced_grains(o["delivered"], o["spilled"]),
            ok=succeeded(o["delivered"], o["spilled"]),
            ending=o["ending"],
            peak_tilt=math.degrees(o["peak_tilt"]),
            duration=o["transfer_duration"],
            commits=" ".join(c["hypothesis"] for c in d.get("commits") or []),
        )
    )
table = pandas.DataFrame(rows)
table.to_csv(HERE / f"{FOLDER.removeprefix('trials_')}_rows.csv", index=False)
summary = []
for key, group in table.groupby(["source", "perturbation", "mode"]):
    summary.append(
        dict(
            zip(["source", "perturbation", "mode"], key),
            n=len(group),
            within=int(group.ok.sum()),
            misplaced=f"{group.misplaced.mean():.1f} ({group.misplaced.min():.1f}-{group.misplaced.max():.1f})",
            spilled=round(group.spilled.mean(), 1),
            peak_tilt=round(group.peak_tilt.mean(), 1),
            duration=round(group.duration.mean(), 1),
            endings=", ".join(
                f"{e} {n}" for e, n in group.ending.value_counts().items()
            ),
        )
    )
summary = pandas.DataFrame(summary)
summary.to_csv(HERE / f"{FOLDER.removeprefix('trials_')}_summary.csv", index=False)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 20)
print(summary.to_string(index=False))
for mode, group in table.groupby("mode"):
    print(
        f"{mode}: {int(group.ok.sum())} of {len(group)} within tolerance, misplaced {group.misplaced.mean():.1f} grains on average"
    )
