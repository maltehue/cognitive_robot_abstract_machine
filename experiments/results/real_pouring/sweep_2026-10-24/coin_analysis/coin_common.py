"""
Shared pieces of the COIN-style offline analyses on the 2026-10-24 goal sweeps.

Everything here is replay on the recorded live-bank traces: no robot, no fabricated
values. Frozen forecasts reuse ``FrozenForecast`` from the repository
(experiments.real_pouring.pouring_ablation) exactly as the 2026-10-19 ablation did:
freeze a parameter point and the source estimate at an origin, integrate depletion along
the recorded future tilt, compare the predicted receiver increment with the later
measurement. Errors are in receiver percentage points.

Targets:
  corn      -> wall readings admitted to the bank's likelihood (``receiver_observation``
               in the trace, every ~0.4 s; the noise field tells calibrated from
               geometric readings), interpolated as the ablation did
  styrofoam -> the camera's crossing count (``counted_grains``; the Jeroen receiver
               equals the source cup, so one grain is one receiver percentage point)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy

from experiments.pouring_model_verification.hypothesis_bank import (
    DrainGeometry,
    ModelPoint,
)
from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.pour_replay import CUP_SIZE, GRAINS_PER_FILL
from experiments.real_pouring.pouring_ablation import FrozenForecast, TiltTrajectory

SWEEP = Path(
    "/home/malte/libs/cognitive_robot_abstract_machine/experiments/results/real_pouring/sweep_2026-10-24"
)
OUT = SWEEP / "coin_analysis"

BASE = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)
"""The uncalibrated drain every sweep pour started from (pour_settings.json: drain)."""

GEOMETRY = DrainGeometry(CUP_SIZE.z, CUP_SIZE.x / 2, GRAINS_PER_FILL, CUP_SIZE.x / 2)
"""The source geometry the forecast evaluation of 2026-10-19 used."""

CORN_RECEIVER_GRAINS = GRAINS_PER_FILL * 722.0 / 361.83
"""Grains in a full translucent cup of corn (README: 722 g receiver, 361.83 g source)."""

STYROFOAM_RECEIVER_GRAINS = GRAINS_PER_FILL
"""The styrofoam receiver is a second Jeroen cup: full receiver = full source."""

CALIBRATED_NOISE_LIMIT_GRAINS = 10.0
"""Receiver noise below this marks a calibrated wall reading (5.68 vs 13.67 grains)."""

MAX_OBSERVATION_GAP = 0.55
"""Longest gap between admitted wall readings inside a scored window, in seconds."""

HORIZONS = (0.5, 1.5, 3.0)
DELAYS = (0.0, 0.33)
STRIDE = 10
"""Origins every ten readings, about one second, as in the 2026-10-19 ablation."""


@dataclass
class Pour:
    name: str
    substance: str
    goal: float
    folder: Path
    t: numpy.ndarray
    tilt: numpy.ndarray
    seen: numpy.ndarray
    count: numpy.ndarray
    source: numpy.ndarray
    active: list
    probabilities: dict
    repose: numpy.ndarray
    log_discharge: numpy.ndarray
    log_drain: numpy.ndarray
    obs_t: numpy.ndarray
    obs_fill: numpy.ndarray
    obs_calibrated: numpy.ndarray
    grains_per_receiver: float
    weighed: dict
    settings: dict

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def trajectory(self) -> TiltTrajectory:
        return TiltTrajectory(self.t, self.tilt)

    @property
    def leading(self) -> list:
        """The hypothesis with the highest probability at each reading."""
        labels = list(self.probabilities)
        matrix = numpy.array([self.probabilities[h] for h in labels])
        return [labels[i] for i in matrix.argmax(axis=0)]

    def first_time(self, labels: list, label: str):
        return next(
            (float(self.t[i]) for i, a in enumerate(labels) if a == label), None
        )

    @property
    def first_commit(self):
        """When the controller first ran on the aim hypothesis (H2), or None."""
        return self.first_time(self.active, "H2")

    @property
    def first_lead(self):
        """When the aim hypothesis first had the highest probability, or None."""
        return self.first_time(self.leading, "H2")

    def rising(self) -> numpy.ndarray:
        """Readings at which the count rose: the stream was flowing by the camera."""
        return numpy.r_[False, numpy.diff(self.count) > 0]

    def corrections_at(self, index: int) -> DrainCorrectionValues:
        return DrainCorrectionValues(
            repose_correction=float(self.repose[index]),
            log_drain_scale=float(self.log_drain[index]),
            log_discharge_scale=float(self.log_discharge[index]),
        )

    def forecast(self, index: int, point: ModelPoint) -> FrozenForecast:
        return FrozenForecast(
            GEOMETRY,
            point,
            min(GRAINS_PER_FILL, max(0.0, float(self.source[index]))),
            self.grains_per_receiver,
        )

    def target(self, start: float, end: float):
        """
        :return: (observed increment in receiver pp, calibrated) or None where the
            window is not covered by consistent measurements.
        """
        times, fills = self.obs_t, self.obs_fill
        if len(times) < 2 or start < times[0] or end > times[-1]:
            return None
        first = max(0, int(numpy.searchsorted(times, start, side="right")) - 1)
        last = min(len(times) - 1, int(numpy.searchsorted(times, end)))
        sel_t = times[first : last + 1]
        if numpy.any(numpy.diff(sel_t) > MAX_OBSERVATION_GAP):
            return None
        calibrated = self.obs_calibrated[first : last + 1]
        if not numpy.all(calibrated == calibrated[0]):
            return None
        sel_f = fills[first : last + 1]
        increment = float(
            numpy.interp(end, sel_t, sel_f) - numpy.interp(start, sel_t, sel_f)
        )
        return increment, bool(calibrated[0])


def point_from(repose: float, log_discharge: float, log_drain: float) -> ModelPoint:
    """The effective deterministic model for a set of drain corrections."""
    return FrozenForecast.corrected(
        BASE,
        DrainCorrectionValues(
            repose_correction=repose,
            log_drain_scale=log_drain,
            log_discharge_scale=log_discharge,
        ),
    )


def load_pour(folder: Path) -> Pour:
    settings = json.loads((folder / "pour_settings.json").read_text())
    substance = settings["material"]["name"].lower()
    trace = json.loads((folder / "live_bank_trace.json").read_text())
    t0 = trace[0]["time"]
    t = numpy.array([x["time"] - t0 for x in trace])
    keep = numpy.r_[True, numpy.diff(t) > 0]
    trace = [x for x, k in zip(trace, keep) if k]
    t = t[keep]
    grains_per_receiver = (
        CORN_RECEIVER_GRAINS if substance == "corn" else STYROFOAM_RECEIVER_GRAINS
    )
    labels = sorted(trace[0]["probabilities"])
    count = numpy.array([x["counted_grains"] for x in trace])
    if substance == "corn":
        obs = [
            (x["time"] - t0, x["receiver_observation"], x["receiver_noise"])
            for x in trace
            if x["receiver_observation"] is not None
        ]
        obs_t = numpy.array([o[0] for o in obs])
        obs_fill = numpy.array([o[1] for o in obs]) / grains_per_receiver * 100.0
        obs_cal = numpy.array([o[2] < CALIBRATED_NOISE_LIMIT_GRAINS for o in obs])
    else:
        obs_t = t.copy()
        obs_fill = count / grains_per_receiver * 100.0
        obs_cal = numpy.ones(len(t), dtype=bool)
    return Pour(
        name=folder.name[:19],
        substance=substance,
        goal=settings["goal_fill"],
        folder=folder,
        t=t,
        tilt=numpy.array([x["tilt"] for x in trace]),
        seen=numpy.array([bool(x["stream_seen"]) for x in trace]),
        count=count,
        source=numpy.array([x["source_grains"] for x in trace]),
        active=[x["active"] for x in trace],
        probabilities={
            h: numpy.array([x["probabilities"][h] for x in trace]) for h in labels
        },
        repose=numpy.array([x["corrections"]["repose_correction"] for x in trace]),
        log_discharge=numpy.array(
            [x["corrections"]["log_discharge_scale"] for x in trace]
        ),
        log_drain=numpy.array([x["corrections"]["log_drain_scale"] for x in trace]),
        obs_t=obs_t,
        obs_fill=obs_fill,
        obs_calibrated=obs_cal,
        grains_per_receiver=grains_per_receiver,
        weighed=json.loads((folder / "weighed_particles.json").read_text()),
        settings=settings,
    )


def load_pours() -> list:
    folders = sorted((SWEEP / "bags_corn").iterdir()) + sorted(
        (SWEEP / "bags_styrofoam").iterdir()
    )
    return [load_pour(f) for f in folders if (f / "live_bank_trace.json").exists()]


def score(
    pour: Pour,
    models: dict,
    horizons=HORIZONS,
    delays=DELAYS,
    stride: int = STRIDE,
    origin_filter=None,
) -> list:
    """
    :param models: name -> function(index) returning the ModelPoint frozen at that origin.
    :param origin_filter: function(pour, index) -> bool, or None for every origin.
    :return: One row per (origin, horizon, delay, model) that has a scorable target.
    """
    trajectory = pour.trajectory
    rows = []
    for index in range(0, len(pour.t), stride):
        if origin_filter is not None and not origin_filter(pour, index):
            continue
        origin = float(pour.t[index])
        for horizon in horizons:
            if origin + horizon > trajectory.times[-1]:
                continue
            predictions = {}
            for name, model in models.items():
                point = model(index)
                predictions[name] = (
                    pour.forecast(index, point).predict(trajectory, origin, horizon)
                    * 100.0
                )
            for delay in delays:
                target = pour.target(origin + delay, origin + horizon + delay)
                if target is None:
                    continue
                observed, calibrated = target
                for name, predicted in predictions.items():
                    rows.append(
                        dict(
                            pour=pour.name,
                            substance=pour.substance,
                            time=origin,
                            horizon=horizon,
                            delay=delay,
                            model=name,
                            observed=observed,
                            predicted=predicted,
                            error=predicted - observed,
                            flowing=bool(pour.seen[index]),
                            calibrated=calibrated,
                        )
                    )
    return rows


def balanced(rows: list, key=lambda r: abs(r["error"])) -> dict:
    """
    :return: {'mean': recording-balanced mean of key, 'per_pour': {pour: mean},
        'origins': number of rows, 'pours': number of recordings}.
    """
    per = {}
    for r in rows:
        per.setdefault(r["pour"], []).append(key(r))
    means = {p: float(numpy.mean(v)) for p, v in per.items()}
    return dict(
        mean=float(numpy.mean(list(means.values()))) if means else math.nan,
        per_pour=means,
        origins=len(rows),
        pours=len(means),
    )


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, allow_nan=True))


# %% end-of-pour concept values

QUIET_WINDOW = 1.0
"""A pour whose camera saw the stream within this many seconds of its end ended flowing."""

RESTORE_TOLERANCE = math.radians(5.0)
"""
Restoring the hysteresis is only plausible where the restored onset lies no further than
this above the pour's highest onset estimate (the bound tracked before the first grain).
"""


def concept_values(pour: Pour) -> dict:
    """
    The bank's end-of-pour concept values: the repose correction and the aim
    hypothesis's discharge factor as the controller last ran on them. The repose
    correction the controller holds is lowered by the material's hysteresis while the
    bank's flow state is "flowing", so a second variant restores the hysteresis where
    the pour ended with the stream still seen and the restored value is plausible.
    """
    from experiments.real_pouring.poured_material import PouredMaterial

    material = PouredMaterial(pour.substance)
    h2 = numpy.array([a == "H2" for a in pour.active])
    ended_flowing = bool(
        pour.seen.any() and pour.t[-1] - pour.t[pour.seen][-1] < QUIET_WINDOW
    )
    share_left = float(pour.source[-1]) / 100.0
    hysteresis = material.repose_hysteresis + material.repose_hysteresis_growth * (
        1 - share_left
    )
    rep_end = float(pour.repose[-1])
    restored = rep_end + hysteresis
    plausible = restored <= float(pour.repose.max()) + RESTORE_TOLERANCE
    rep_quiescent = restored if ended_flowing and plausible else rep_end
    first_lead = next(
        (
            float(pour.t[i])
            for i in range(len(pour.t))
            if pour.probabilities["H2"][i] > 0.5
        ),
        None,
    )
    first_commit = pour.first_commit
    return dict(
        pour=pour.name,
        substance=pour.substance,
        goal=pour.goal,
        duration=round(pour.duration, 1),
        weighed=round(pour.weighed["received_fill_level"], 4),
        end_active=pour.active[-1],
        ended_flowing=ended_flowing,
        hysteresis_restored=bool(ended_flowing and plausible),
        repose_end_deg=round(math.degrees(rep_end), 1),
        repose_quiescent_deg=round(math.degrees(rep_quiescent), 1),
        repose_end_rad=rep_end,
        repose_quiescent_rad=rep_quiescent,
        discharge_factor_end=round(math.exp(pour.log_discharge[-1]), 3),
        discharge_factor_last_aim=(
            round(math.exp(pour.log_discharge[h2][-1]), 3) if h2.any() else None
        ),
        log_discharge_last_aim=(
            float(pour.log_discharge[h2][-1]) if h2.any() else None
        ),
        drain_factor_end=round(math.exp(pour.log_drain[-1]), 3),
        first_lead_s=None if first_lead is None else round(first_lead, 1),
        first_commit_s=None if first_commit is None else round(first_commit, 1),
        share_before_commit=round(
            (pour.duration if first_commit is None else first_commit) / pour.duration, 3
        ),
        committed=first_commit is not None,
    )
