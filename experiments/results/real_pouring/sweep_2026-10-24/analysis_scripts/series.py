import json, math, sys
from pathlib import Path
import numpy

bag = Path(sys.argv[1])
a, b, step = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
tr = json.loads((bag / "live_bank_trace.json").read_text())
t0 = tr[0]["time"]
d = numpy.load(bag / "transfer_control_cycles.npz", allow_pickle=True)
ct = d["times"]
dv = d["debug_values"]
nxt = a
for x in tr:
    tt = x["time"] - t0
    if tt < nxt or tt > b:
        continue
    nxt = tt + step
    i = min(numpy.searchsorted(ct, tt), len(ct) - 1)
    print(
        f"{tt:5.1f} tilt {math.degrees(x['tilt']):5.1f} seen {int(x['stream_seen'])} cnt {x['counted_grains']/100:.3f} told {x['told_grains']/100:.3f} src {x['source_grains']/100:.3f} pred {dv[i,0]:.3f} rate {dv[i,1]:+.3f} {x['active']} dis {x['corrections']['log_discharge_scale']:+.2f} rep {math.degrees(x['corrections']['repose_correction']):3.0f}"
    )
