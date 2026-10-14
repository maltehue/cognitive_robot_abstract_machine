"""
Replay the real pours of 2026-10-12 through the camera-only hypothesis bank: what would
it have concluded from the pole camera alone?
"""

import glob, json, math, pickle, sys
import numpy, cv2

sys.path.insert(
    0,
    "/home/malte/libs/cognitive_robot_abstract_machine/experiments/results/pouring_model_verification/analysis_scripts/followups/round6",
)
from confirmation_runs import SETTINGS  # noqa: E402  (the frozen bank of section N)
import experiments.pouring_model_verification.hypothesis_bank as hb
from experiments.pouring_model_verification.hypothesis_bank import (
    DrainGeometry,
    HypothesisBank,
    ModelPoint,
    ModelSwitch,
    Reading,
    SensorKnowledge,
    FreeParameter,
    Hypothesis,
)
from experiments.pouring_model_verification.recovery import BankCoupling, hypotheses_for
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.sensor import (
    POLE_CAMERA_DROPOUT,
    POLE_CAMERA_COUNT_NOISE,
    POLE_CAMERA_DELAY,
)

hb.REPOSE_ANGLE_BOUNDS = (
    0.0,
    math.radians(85.0),
)  # this replay only: the real cup's bulk onset lies near the sim bound of 65 deg
FULL_G, UNITS = 2.65, 100.0  # one bank "grain" = 1 % of a full cup
GEOMETRY = DrainGeometry(
    container_height=0.16, lip_offset=0.035, grains_per_fill=UNITS, slot_offset=0.035
)
DEMO = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)
SENSORS = SensorKnowledge(
    receiver_noise=None,
    receiver_delay=0.0,
    counts_falling_grains=False,
    fall_time=0.1,
    source_noise=None,
    source_delay=0.0,
    crossing_noise=None,
    stream_dropout=POLE_CAMERA_DROPOUT,
    count_noise=POLE_CAMERA_COUNT_NOISE,
)
PERIOD, SEEN_PX = 0.1, 30


def tilt_of(kin):
    T = numpy.array([r["t"] for r in kin["rows"]])
    return T, numpy.arccos(numpy.clip([r["tool_R"][2, 1] for r in kin["rows"]], -1, 1))


def signal(d, kind):
    """
    Per-frame white pixels where the camera sees grains pass: the receiver's opening
    (demo) or new grains on the tray (ramp).
    """
    if kind == "demo":
        return numpy.load(f"{d}/receiver_signal_raw.npy")
    o = json.load(open("onset.json"))[d.split("/")[1]]
    c = numpy.maximum.accumulate(numpy.convolve(o["counts"], numpy.ones(5) / 5, "same"))
    return numpy.r_[0.0, numpy.diff(c)]  # newly landed white area per frame


def readings(d, kind, px_per_unit):
    o = pickle.load(open(f"{d}/extracted.pkl", "rb"))
    k = pickle.load(open(f"{d}/kin.pkl", "rb"))
    T, tilt = tilt_of(k)
    it = numpy.array([im[0] for im in o["images"]])
    px = signal(d, kind)
    start = T[numpy.argmax(tilt > math.radians(1.5))]
    out, counted = [], 0.0
    for at in numpy.arange(start, min(T[-1], it[-1]) - 0.2, PERIOD):
        window = (it > at - POLE_CAMERA_DELAY - PERIOD) & (it <= at - POLE_CAMERA_DELAY)
        seen = (
            bool((px[window] > SEEN_PX).any())
            if kind == "demo"
            else bool(px[window].sum() > SEEN_PX)
        )
        counted += px[window].sum() / px_per_unit
        out.append(
            Reading(
                time=float(at - start),
                tilt=float(numpy.interp(at, T, tilt)),
                receiver_grains=None,
                source_grains=UNITS if not out else None,
                stream_seen=seen,
                counted_grains=counted,
            )
        )
    return out


def replay(reads):
    bank = HypothesisBank(
        hypotheses=hypotheses_for(
            SourceCup.CUP, outflow_capacity=SETTINGS.outflow_capacity
        ),
        geometry=GEOMETRY,
        base=DEMO,
        sensors=SENSORS,
        settings=SETTINGS,
        seed=0,
    )
    coupling = BankCoupling(bank=bank)
    trace = []
    for r in reads:
        commit = coupling.observe(r, poured_share=0.0)
        trace.append(
            (
                r.time,
                math.degrees(r.tilt),
                coupling.active.label,
                {h.label: round(p, 3) for h, p in bank.probabilities.items()},
            )
        )
    onset = bank.filters[Hypothesis.ONSET].mean().repose_angle
    first_seen = next((round(r.time, 2) for r in reads if r.stream_seen), None)
    return dict(
        first_stream=first_seen,
        first_stream_tilt=(
            None
            if first_seen is None
            else round(math.degrees(next(r.tilt for r in reads if r.stream_seen)), 1)
        ),
        commits=[
            (
                c.hypothesis.label,
                round(c.time, 2),
                round(min(trace, key=lambda x: abs(x[0] - c.time))[1], 1),
            )
            for c in coupling.commits
        ],
        onset_estimate_deg=round(math.degrees(onset), 1),
        received_estimate=round(coupling.received_grains, 1),
        source_estimate=round(coupling.source_grains, 1),
        final_probabilities=trace[-1][3],
    )


if __name__ == "__main__":
    demos = sorted(glob.glob("demo/*"))
    ramps = sorted(glob.glob("ramps/*"))
    weighed = {
        d: (
            json.load(open(f"{d}/weighed_particles.json"))["received_mass_grams"]
            + json.load(open(f"{d}/weighed_particles.json"))["spilled_mass_grams"]
        )
        / FULL_G
        * UNITS
        for d in demos
    }
    weighed.update(
        {
            d: json.load(open(f"{d}/ramp.json"))["weighed"]["drained_mass_grams"]
            / FULL_G
            * UNITS
            for d in ramps
        }
    )
    totals = {
        d: signal(d, "demo" if d in demos else "ramp").sum() for d in demos + ramps
    }
    results = {}
    for group, kind in ((demos, "demo"), (ramps, "ramp")):
        for d in group:
            others = [x for x in group if x != d]
            px_per_unit = numpy.mean(
                [totals[x] / weighed[x] for x in others]
            )  # calibrated on the other pours of the same kind
            res = replay(readings(d, kind, px_per_unit))
            res["weighed"] = round(weighed[d], 1)
            res["camera_total"] = round(totals[d] / px_per_unit, 1)
            results[d] = res
            print(d, json.dumps(res))
    json.dump(results, open("camera_replay.json", "w"), indent=1)
