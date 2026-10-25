"""
V-c, step 1: replay the live bank on a recorded corn pour with its filters exposed, so
every hypothesis's controller point is known at every reading (the trace only holds the
active hypothesis's corrections).

python v_c_replay_bank.py <bag folder with the .mcap and pour_settings.json> <out.json>

Replays the detector on the camera frames and the bank as PourReplay.of does
(experiments/real_pouring/pour_replay.py), with the recorded pour's own settings and
drain, the demo's default seed 0, and the translucent-cup corn perception. Writes one
row per reading: time, tilt, stream seen, count, probabilities, active, the corrections
written (blended, stall-weighted, as the controller used them), the active hypothesis's
unblended controller point, and each hypothesis's controller point and source estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.pouring_model_verification.sensor import CameraCondition
from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.pour_perception import PourPerception
from experiments.real_pouring.pour_replay import (
    BACKGROUND_FRAMES,
    FRAME_PERIOD,
    GRAINS_PER_FILL,
    READING_EVERY,
    PourRecording,
    _replay_world,
    _transform,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceiverFillReport, ReceivingCup
from experiments.real_pouring.stream_detection import CameraView

bag = Path(sys.argv[1])
out = Path(sys.argv[2])
settings = json.loads((bag / "pour_settings.json").read_text())
material = PouredMaterial(settings["material"]["name"].lower())
cup = ReceivingCup(settings["receiving_cup"]["name"].lower())


def plain(data: dict) -> dict:
    """
    The serializer's type tag removed.
    """
    return {k: v for k, v in data.items() if k != "__json_type__"}


drain = ModelPoint.from_json(plain(settings["drain"]))
bank_settings = BankSettings.from_json(plain(settings["bank"]))
perception = PourPerception.for_cup(cup, material)
grains_per_receiver = GRAINS_PER_FILL * cup.source_cups_held(material)
goal_grains = settings["goal_fill"] * grains_per_receiver

recording = PourRecording.read(bag)
world, source, source_joint = _replay_world(recording, drain, cup, material)
detector = cup.detector_for(
    material=material,
    view=CameraView(
        intrinsics=recording.intrinsics, world_T_camera=recording.world_T_camera
    ),
    opening_centre=recording.receiver_position + numpy.array([0.0, 0.0, cup.height]),
    frame_period=FRAME_PERIOD,
    wall_calibration=(
        perception.wall_calibration
        if perception.fill_report is ReceiverFillReport.MEASURED
        else None
    ),
)
first = recording.first_pour_frame
detector.learn_background(
    [f.image() for f in recording.frames[max(0, first - BACKGROUND_FRAMES) : first]]
)
receiver = world.get_semantic_annotation_by_name("receiving_cup")
bank = LiveBank.for_cups(
    source=source,
    receiver=receiver,
    world=world,
    built=drain,
    settings=bank_settings,
    camera=CameraCondition.POLE_COUNTING.model,
    grains_per_fill=GRAINS_PER_FILL,
    loaded_grains=settings["start_fill"] * GRAINS_PER_FILL,
    goal_grains=goal_grains,
    opening_radius=cup.inner_radius,
    prediction_window=settings["prediction_duration"],
    fill_report=perception.fill_report,
    stop_lead_time=settings["stop_lead_time"],
    in_flight_limit=settings["fill_tolerance"] * grains_per_receiver,
    receiver_evidence=perception.receiver_evidence,
)

rows = []
for index in range(first, len(recording.frames)):
    frame = recording.frames[index]
    reading = detector.observe(frame.image(), stamp=frame.stamp - recording.pour_start)
    if (index - first) % READING_EVERY:
        continue
    pose = recording.cup_poses[index]
    source_joint.origin = _transform(pose, world)
    step = bank.observe(reading)
    coupling = bank.coupling
    tilt = float(step.tilt)
    hypotheses = {}
    for hypothesis, probability in coupling.bank.probabilities.items():
        f = coupling.bank.filters[hypothesis]
        point = f.controller_point(tilt, coupling._trickle)
        hypotheses[hypothesis.label] = dict(
            probability=probability,
            repose=point.repose_angle,
            discharge=point.discharge_coefficient,
            drain=point.outflow_rate_constant,
            mean_repose=f.mean().repose_angle,
            source=f.source_grains,
        )
    active_point = coupling.point
    rows.append(
        dict(
            time=frame.stamp - recording.pour_start,
            stamp=frame.stamp,
            tilt=tilt,
            stream_seen=bool(reading.stream_seen),
            counted_grains=reading.counted_grains,
            heap_fill=reading.heap_fill,
            wall_calibrated=bool(reading.wall_calibrated),
            active=step.active,
            cautious=coupling.cautious,
            trickle=coupling._trickle,
            written=dict(vars(step.corrections)),
            active_point=dict(
                repose=active_point.repose_angle,
                discharge=active_point.discharge_coefficient,
                drain=active_point.outflow_rate_constant,
            ),
            source_grains=step.source_grains,
            received_grains=step.received_grains,
            told_grains=step.told_grains,
            hypotheses=hypotheses,
        )
    )

out.write_text(
    json.dumps(
        dict(
            bag=bag.name,
            pour_start=recording.pour_start,
            seed=0,
            stall_probability=bank_settings.stall_probability,
            grains_per_receiver=grains_per_receiver,
            rows=rows,
        ),
        indent=None,
    )
)
print(bag.name, len(rows), "readings replayed", flush=True)
