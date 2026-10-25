import json, math, sys
from pathlib import Path
import numpy

C = 722.0 / 361.83
for bag in sorted(Path(p) for p in sys.argv[1:]):
    s = json.loads((bag / "pour_settings.json").read_text())
    mat = s["material"]["name"].lower()
    cd0 = s["drain"]["discharge_coefficient"]
    g = 100 * (C if mat == "corn" else 1)
    w = json.loads((bag / "weighed_particles.json").read_text())
    tr = json.loads((bag / "live_bank_trace.json").read_text())
    t0 = tr[0]["time"]
    t = numpy.array([x["time"] - t0 for x in tr])
    dis = numpy.array([x["corrections"]["log_discharge_scale"] for x in tr])
    drn = numpy.array([x["corrections"]["log_drain_scale"] for x in tr])
    act = [x["active"] for x in tr]
    flowing = numpy.array([x["counted_grains"] for x in tr])
    fl = numpy.r_[0, numpy.diff(flowing)] > 0
    share = {h: sum(1 for a in act if a == h) / len(act) for h in sorted(set(act))}
    first_h2 = next((t[i] for i, a in enumerate(act) if a == "H2"), None)
    rec_belief = tr[-1]["received_grains"] / g
    src_belief = tr[-1]["source_grains"] / 100
    delivered = (
        (w["received_mass_grams"] + w["spilled_mass_grams"])
        / w["full_cup_mass_grams"]
        * (C if mat == "corn" else 1)
    )
    print(
        f"{mat[:5]} goal {s['goal_fill']:.1f} {bag.name[11:19]} | c_D end {cd0*math.exp(dis[-1]):.3f} (x{math.exp(dis[-1]):.2f}), min {cd0*math.exp(dis.min()):.3f}, while flowing median x{math.exp(numpy.median(dis[fl])) if fl.any() else 1:.2f} | drain x{math.exp(drn[-1]):.2f} | active "
        + " ".join(f"{h} {v:.0%}" for h, v in share.items())
        + f" | first H2 {first_h2 if first_h2 is None else round(first_h2,1)} | bank received {rec_belief:.3f} vs weighed {w['received_fill_level']:.3f}; source left {src_belief:.3f} vs {1-delivered:.3f}"
    )
