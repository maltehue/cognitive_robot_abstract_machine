import json, sys, math, glob, os

G = 722.0 / 361.83 * 100  # source grains per receiver fill
goals = [0.3, 0.3, 0.2, 0.2, 0.4, 0.4]
for goal, d in zip(goals, sorted(glob.glob(sys.argv[1] + "/2026*"))):
    t = json.load(open(d + "/live_bank_trace.json"))
    w = json.load(open(d + "/weighed_particles.json"))
    t0 = t[0]["time"]
    tt = [s["time"] - t0 for s in t]
    tilt = [math.degrees(s["tilt"]) for s in t]
    told = [s["told_grains"] / G for s in t]
    obs = [
        (s["time"] - t0, s["receiver_observation"] / G)
        for s in t
        if s["receiver_observation"] is not None
    ]
    seen = [i for i, s in enumerate(t) if s["stream_seen"]]
    first = seen[0] if seen else None
    ip = max(range(len(t)), key=lambda i: tilt[i])
    cross = next((i for i in range(len(t)) if told[i] >= goal - 0.01), None)
    print(
        f"== {os.path.basename(d)} goal {goal} weighed {w['received_fill_level']:.3f} (err {w['received_fill_level']-goal:+.3f}) spilled {w['spilled_mass_grams']} g; trace {tt[-1]:.1f}s n={len(t)}"
    )
    print(
        f"   stream first {tt[first]:.1f}s at {tilt[first]:.1f} deg; seen frames {len(seen)}; peak tilt {tilt[ip]:.1f} at {tt[ip]:.1f}s"
    )
    if cross is not None:
        print(
            f"   told>=goal-0.01 at {tt[cross]:.1f}s tilt {tilt[cross]:.1f}; told {told[cross]:.3f}; last wall obs before {[round(o,3) for x,o in obs if x<=tt[cross]][-3:]}"
        )
    print(
        f"   final told {told[-1]:.3f} max told {max(told):.3f}; final count(inside) {t[-1]['counted_grains']:.1f}cnt; last wall obs {[round(o,3) for x,o in obs[-5:]]}"
    )
    print(
        f"   final active {t[-1]['active']} probs {t[-1]['probabilities']} corr {t[-1]['corrections']}"
    )
