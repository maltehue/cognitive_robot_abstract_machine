import numpy, sys, json, math

d = numpy.load(sys.argv[1] + "/transfer_control_cycles.npz", allow_pickle=True)
a, b = float(sys.argv[2]), float(sys.argv[3])
t = d["times"]
v = d["velocities"]
S = d["row_sensitivities"]
lo = d["row_lower_bounds"]
up = d["row_upper_bounds"]
sl = d["row_slacks"]
dv = d["debug_values"]
for i in range(len(t)):
    if a <= t[i] <= b and i % 8 == 0:
        print(
            f"{t[i]:5.2f} wrist3 {math.degrees(v[i,5]):+6.1f} deg/s  pred {dv[i,0]:.3f} rate {dv[i,1]:+.3f}  fill-row dF/dwrist3 {S[i,0,5]:+.4f} bounds [{lo[i,0]:+.4f},{up[i,0]:+.4f}] slack {sl[i,0]:+.4f}"
        )
