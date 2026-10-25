"""
One sample per 0.1 s of every weighed translucent-cup corn pour: features known at t,
and the wall-measured fill gain over the next 1.5 s.
"""

import json, sys
from pathlib import Path
import numpy
from experiments.real_pouring.wall_fill_validation import (
    WallFillRecording,
    WallFillPartition,
    SourceReturn,
)
from krrood.adapters.json_serializer import from_json

HORIZON = 1.5
STRIDE = 3
EXCLUDED = {"2026-10-19_16-13-33"}
cache_dir, calibration_path, out = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
)
calibration = from_json(json.loads(calibration_path.read_text()))
rows = []
for cache in sorted(cache_dir.glob("2026-*.npz")):
    name = cache.stem
    if name in EXCLUDED or name.endswith("prediction"):
        continue
    with numpy.load(cache) as c:
        tilts = c["tilts"].astype(float)
    rec = WallFillRecording.read(cache, WallFillPartition.DEVELOPMENT)
    raw = rec.replay(calibration)
    fill = numpy.array([numpy.nan if v is None else v for v in raw])
    first = numpy.flatnonzero(~numpy.isnan(fill))[0]
    fill[:first] = 0.0
    for i in range(1, len(fill)):
        if numpy.isnan(fill[i]):
            fill[i] = fill[i - 1]
    smooth = numpy.array(
        [numpy.median(fill[max(0, i - 4) : i + 1]) for i in range(len(fill))]
    )
    t = rec.times
    end = SourceReturn(tilts).held_tilt_end()
    start = int(numpy.flatnonzero(tilts > 45.0)[0])
    peak_so_far = numpy.maximum.accumulate(tilts)

    def at(time):
        return int(min(len(t) - 1, numpy.searchsorted(t, time)))

    last_burst = -1e9
    for i in range(start, end, STRIDE):
        j = at(t[i] + HORIZON)
        if j > end:
            break
        past_short = smooth[i] - smooth[at(t[i] - 0.5)]
        past_long = smooth[i] - smooth[at(t[i] - HORIZON)]
        recent = smooth[max(start, at(t[i] - 0.33)) : i + 1]
        if len(recent) and recent[-1] - recent.min() > 0.02:
            last_burst = t[i]
        earlier_peak = peak_so_far[at(t[i] - 0.5)]
        steps = [at(t[i] + k * 0.1) for k in range(16)]
        rows.append(
            dict(
                pour=name,
                t=float(t[i]),
                fill=float(smooth[i]),
                tilt=float(tilts[i]),
                tilt_change=float(tilts[j] - tilts[i]),
                tilt_rate=float((tilts[i] - tilts[at(t[i] - 0.33)]) / 0.33),
                above_earlier_peak=float(tilts[i] - earlier_peak),
                gain_past_short=float(past_short),
                gain_past_long=float(past_long),
                since_burst=float(min(10.0, t[i] - last_burst)),
                target=float(smooth[j] - smooth[i]),
                tilt_sequence=[float(tilts[k]) for k in steps],
                fill_sequence=[float(smooth[k]) for k in steps],
                weighed=float(rec.weighed),
            )
        )
out.write_text(json.dumps(rows))
pours = sorted({r["pour"] for r in rows})
print(f"{len(rows)} samples from {len(pours)} pours")
