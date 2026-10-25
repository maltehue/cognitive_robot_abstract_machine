import json, sys, math

G = 722.0 / 361.83 * 100
t = json.load(open(sys.argv[1]))
t0 = t[0]["time"]
a = float(sys.argv[2])
step = int(sys.argv[3])
prev = None
for i, s in enumerate(t):
    x = s["time"] - t0
    if x < a or i % step:
        continue
    ro = s["receiver_observation"]
    print(
        f"{x:5.1f} tilt {math.degrees(s['tilt']):5.1f} seen {int(s['stream_seen'])} cnt {s['counted_grains']:5.1f} told {s['told_grains']/G:.3f} fl {s['in_flight_grains']/G:.3f} wall {'' if ro is None else f'{ro/G:.3f}'} src {s['source_grains']:5.1f} {s['active']} rep {math.degrees(s['corrections']['repose_correction']):4.0f} dis {s['corrections']['log_discharge_scale']:.2f}"
    )
