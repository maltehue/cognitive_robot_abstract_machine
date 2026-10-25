"""
Per bag: motion quality during the transfer and return, reversal episodes and fill told
vs weighed.
"""

import json, math, sys
from pathlib import Path
import numpy, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from experiments.real_pouring.pour_replay import PourRecording

bag = Path(sys.argv[1])
motion = json.load(open(sys.argv[2]))
rec = PourRecording.read(bag)
start = rec.pour_start
G = 722.0 / 361.83 * 100
trace = json.loads((bag / "live_bank_trace.json").read_text())
tt = numpy.array([s["time"] - start for s in trace])
told = numpy.array([s["told_grains"] / G for s in trace])
tilt_tr = numpy.degrees([s["tilt"] for s in trace])
seen = numpy.array([s["stream_seen"] for s in trace])
end = tt[-1]
t = numpy.array([r["t"] for r in motion])
tilt = numpy.array([r["tilt"] for r in motion])
rate = numpy.array([r["rate"] for r in motion])
align = numpy.array([r["align"] for r in motion])
speed = numpy.array([r["speed"] for r in motion]) * 100
w = json.loads((bag / "weighed_particles.json").read_text())["received_fill_level"]
transfer = (t > 1.5) & (t < end)
ret = (t >= end) & (tilt > 3)
print(
    f"== {bag.name}: weighed {w:.3f}; transfer {end:.1f}s; final told {numpy.median(told[tt > end - 1.5]):.3f} (told - weighed {numpy.median(told[tt > end - 1.5]) - w:+.3f})"
)
print(
    f"   transfer: max align {align[transfer].max():.2f} deg, max cup speed {speed[transfer].max():.1f} cm/s, peak tilt {tilt[transfer].max():.1f}"
)
print(
    f"   return while tilted >3 deg: max cup speed {speed[ret].max() if ret.any() else 0:.1f} cm/s, max align {align[ret].max() if ret.any() else 0:.2f}"
)
# reversals: rate crossing below -2 deg/s during the transfer
rev = [
    i for i in range(1, len(t)) if transfer[i] and rate[i] < -2 and rate[i - 1] >= -2
]
r = rosbag2_py.SequentialReader()
r.open(
    rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
    rosbag2_py.ConverterOptions("cdr", "cdr"),
)
types = {x.name: get_message(x.type) for x in r.get_all_topics_and_types()}
JT, JP = [], []
while r.has_next():
    topic, data, _ = r.read_next()
    if topic != "/left_arm/joint_states":
        continue
    m = deserialize_message(data, types[topic])
    names = [n.replace("left_", "") for n in m.name]
    JT.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 - start)
    JP.append(list(m.position))
JT = numpy.array(JT)
JV = numpy.degrees(numpy.gradient(numpy.array(JP), JT, axis=0))
w3 = names.index("wrist_3_joint")
for i in rev:
    a = t[i]
    sel = (JT >= a - 0.3) & (JT <= a + 1.5)
    others = numpy.abs(numpy.delete(JV[sel], w3, axis=1)).max()
    k = int(numpy.argmin(abs(tt - a)))
    win = (t >= a - 0.3) & (t <= a + 1.5)
    print(
        f"   reversal {a:5.1f}s at {tilt[i]:5.1f} deg, told {told[k]:.3f}: other joints <= {others:.1f} deg/s, wrist_3 {numpy.abs(JV[sel, w3]).max():.1f}, align <= {align[win].max():.2f}, speed <= {speed[win].max():.1f}"
    )
# stream onsets after a dry spell
onsets = [i for i in range(1, len(tt)) if seen[i] and not seen[max(0, i - 5) : i].any()]
print(
    "   stream (re)starts: "
    + ", ".join(f"{tt[i]:.1f}s@{tilt_tr[i]:.0f}deg told {told[i]:.3f}" for i in onsets)
)
