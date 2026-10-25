import json, sys, math

t = json.load(open(sys.argv[1]))
print("keys", list(t[0].keys()))
t0 = t[0]["time"]
for s in t[:: int(sys.argv[2]) if len(sys.argv) > 2 else 1]:
    ro = s["receiver_observation"]
    rn = s["receiver_noise"]
    print(
        f"{s['time']-t0:6.2f} tilt {math.degrees(s['tilt']):6.1f} seen {int(s['stream_seen'])} cnt {s['counted_grains']:6.1f} {s['active']:3s} "
        f"rec {s['received_grains']:6.1f} fl {s['in_flight_grains']:5.1f} told {s['told_grains']:6.1f} obs {ro if ro is None else round(ro,1)} n {rn if rn is None else round(rn,1)} src {s['source_grains']:6.1f} "
        f"corr {s['corrections']['repose_correction']:.2f},{s['corrections']['log_drain_scale']:.2f},{s['corrections']['log_discharge_scale']:.2f} "
        + " ".join(f"{k}:{v:.2f}" for k, v in s["probabilities"].items() if v > 0.05)
    )
