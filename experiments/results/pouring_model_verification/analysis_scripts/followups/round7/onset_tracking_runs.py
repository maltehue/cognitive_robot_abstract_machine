"""
Section N's frozen configuration with the robot's sensors (the pole camera alone), once
as frozen and once with every hypothesis tracking the onset (BankSettings.tracks_onset),
on seeds 20-27: is tracking the onset worth it in simulation?
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "round6"))
import confirmation_runs as c
from experiments.pouring_model_verification.campaign import Campaign
from experiments.pouring_model_verification.recovery_study import bank_trial
from experiments.pouring_model_verification.records import TrialRecord
from experiments.pouring_model_verification.sensor import SensorSet

OUT = c.R / "pouring_model_verification/trials_onset_tracking"
SENSORS = SensorSet.POLE_CAMERA
CONDITIONS = [
    (c.PLAIN_WIDE, c.BASELINE),
    (c.PLAIN_WIDE, c.rho(0.5)),
    (c.PLAIN_WIDE, c.rho(1.25)),
    (c.SPOUT_WIDE, c.BASELINE),
    (c.SPOUT_WIDE, c.rho(0.5)),
]
MODES = {
    "concept": replace(c.SETTINGS, label="bankN_concept"),
    "tracked": replace(c.SETTINGS, label="bankN_tracked", tracks_onset=True),
}


def trials():
    return [
        replace(
            bank_trial(
                c.condition(scene, perturbation), c.TWIN, seed, settings, SENSORS
            ),
            check_residuals=c.RESIDUALS[scene.source],
        )
        for settings in MODES.values()
        for scene, perturbation in CONDITIONS
        for seed in c.SEEDS
    ]


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    campaign = Campaign(output=OUT, workers=workers)
    all_trials = trials()
    names = {t.name for t in all_trials}
    assert len(names) == len(all_trials), "two trials share a name"
    recorded = (
        {r.description["name"] for r in TrialRecord.read_all(OUT)}
        if OUT.exists()
        else set()
    )
    pending = [t for t in all_trials if t.name not in recorded]
    print(len(all_trials), "trials,", len(pending), "pending", flush=True)
    print("failed", campaign.run(pending), flush=True)
