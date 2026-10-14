"""
The onset-tracking comparison of onset_tracking_runs.py without the twin's prior on the
repose angle and the drain rate (adapts_onset and adapts_drain_rate off), as the live
bank runs on the robot: every hypothesis that estimates a parameter starts from its full
range.
"""

import sys
from dataclasses import replace

import onset_tracking_runs as o
from experiments.pouring_model_verification.campaign import Campaign
from experiments.pouring_model_verification.recovery_study import bank_trial
from experiments.pouring_model_verification.records import TrialRecord

c = o.c
OUT = c.R / "pouring_model_verification/trials_onset_tracking_no_twin"
WITHOUT_TWIN = dict(adapts_onset=False, adapts_drain_rate=False)
MODES = {
    "concept": replace(c.SETTINGS, label="bankN_notwin_concept", **WITHOUT_TWIN),
    "tracked": replace(
        c.SETTINGS, label="bankN_notwin_tracked", tracks_onset=True, **WITHOUT_TWIN
    ),
}


def trials():
    return [
        replace(
            bank_trial(
                c.condition(scene, perturbation), c.TWIN, seed, settings, o.SENSORS
            ),
            check_residuals=c.RESIDUALS[scene.source],
        )
        for settings in MODES.values()
        for scene, perturbation in o.CONDITIONS
        for seed in c.SEEDS
    ]


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    all_trials = trials()
    assert len({t.name for t in all_trials}) == len(
        all_trials
    ), "two trials share a name"
    recorded = (
        {r.description["name"] for r in TrialRecord.read_all(OUT)}
        if OUT.exists()
        else set()
    )
    pending = [t for t in all_trials if t.name not in recorded]
    print(len(all_trials), "trials,", len(pending), "pending", flush=True)
    print("failed", Campaign(output=OUT, workers=workers).run(pending), flush=True)
