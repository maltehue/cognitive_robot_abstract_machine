import sys, json, math
from pathlib import Path
from experiments.real_pouring.pour_replay import PourRecording

bag = Path(sys.argv[1])
a, b = float(sys.argv[2]), float(sys.argv[3])
start = PourRecording.read(bag).pour_start
G = 722.0 / 361.83 * 100
for s in json.loads((bag / "live_bank_trace.json").read_text()):
    t = s["time"] - start
    if a <= t <= b:
        c = s["corrections"]
        print(
            f"{t:5.2f} tilt {math.degrees(s['tilt']):5.1f} seen {int(s['stream_seen'])} told {s['told_grains']/G:.3f} {s['active']} repose {math.degrees(c['repose_correction']):5.1f} logdis {c['log_discharge_scale']:.2f} logdrain {c['log_drain_scale']:.2f}"
        )
