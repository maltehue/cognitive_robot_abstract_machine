import json, sys, math, glob, os, statistics as st

G = 722.0 / 361.83 * 100
goals = [0.3, 0.3, 0.2, 0.2, 0.4, 0.4]
print(
    "bag       goal weighed  settled_wall  wall_bias  own_err  scale_err  wall_sd_plateau  stop_trigger"
)
for goal, d in zip(goals, sorted(glob.glob(sys.argv[1] + "/2026*"))):
    t = json.load(open(d + "/live_bank_trace.json"))
    w = json.load(open(d + "/weighed_particles.json"))["received_fill_level"]
    t0 = t[0]["time"]
    end = t[-1]["time"] - t0
    last_seen = max(s["time"] - t0 for s in t if s["stream_seen"])
    tail = [
        s["receiver_observation"] / G
        for s in t
        if s["receiver_observation"] is not None
        and s["time"] - t0 > max(last_seen + 0.7, end - 2.0)
    ]
    if len(tail) < 3:
        tail = [
            s["receiver_observation"] / G
            for s in t
            if s["receiver_observation"] is not None and s["time"] - t0 > end - 2.0
        ]
    settled = st.median(tail)
    # plateau noise: wall obs during windows where count is flat for >= 2 s and fill > 0.1
    obs = [
        (s["time"] - t0, s["counted_grains"], s["receiver_observation"] / G)
        for s in t
        if s["receiver_observation"] is not None
    ]
    res = []
    for i in range(len(obs)):
        win = [o for o in obs if abs(o[0] - obs[i][0]) <= 1.0]
        if (
            obs[i][2] > 0.1
            and max(o[1] for o in win) - min(o[1] for o in win) < 0.3
            and len(win) >= 4
        ):
            res.append(obs[i][2] - st.mean(o[2] for o in win))
    sd = st.pstdev(res) if len(res) > 3 else float("nan")
    # last tick and what told was: the task ends on told >= goal-0.01 with settled inflow
    told_end = t[-1]["told_grains"] / G
    print(
        f"{os.path.basename(d)[11:]}  {goal:.1f}  {w:.3f}    {settled:.3f} (n={len(tail)})  {settled-w:+.3f}    {settled-goal:+.3f}   {w-goal:+.3f}     {sd:.3f}            max told last 2s {max(s['told_grains']/G for s in t if s['time']-t0>end-2):.3f}"
    )
