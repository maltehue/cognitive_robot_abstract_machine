import json, sys, numpy
from pathlib import Path
from dataclasses import replace
from experiments.real_pouring.wall_fill import WallFillCalibration
from experiments.real_pouring.wall_fill_validation import WallFillPartition
from krrood.adapters.json_serializer import from_json

d = Path(sys.argv[1])
ds = from_json(json.loads((d / "dataset.json").read_text()))
obs = [r.observe(ds.measurement) for r in ds.read(d)]
old = from_json(json.loads(Path(sys.argv[2]).read_text()))
new = ds.calibrate(obs)
for name, cal in (
    ("deployed (upright, proportional)", old),
    ("new (held tilt, offset)", new),
):
    res = [o.evaluate(cal) for o in obs]
    for grp, sel in (
        ("2026-10-23", lambda r: r.recording.startswith("2026-10-23")),
        (
            "older dev",
            lambda r: not r.recording.startswith("2026-10-23")
            and r.partition is WallFillPartition.DEVELOPMENT,
        ),
        ("validation", lambda r: r.partition is WallFillPartition.VALIDATION),
    ):
        e = numpy.array([r.error for r in res if sel(r)])
        print(
            f"{name:34s} {grp:10s} bias {e.mean()*100:+.2f} pp  mean|e| {abs(e).mean()*100:.2f} pp  max {abs(e).max()*100:.2f}"
        )
