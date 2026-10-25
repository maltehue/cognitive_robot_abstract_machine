"""
Consecutive offline solves through a recorded tilt reversal, with chosen tasks paused.
"""

import json, math, sys
from pathlib import Path
import numpy
from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.command_evaluation import ArmSample
from experiments.real_pouring.controller_ablation import ControllerProbe
from experiments.real_pouring.pour_replay import PourRecording, GRAINS_PER_FILL
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.pouring import (
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
    TiltPacing,
    FillByTransferTask,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

bag = Path(sys.argv[1])
t_start, duration = float(sys.argv[2]), float(sys.argv[3])
paused_names = [a for a in sys.argv[4:] if not a.startswith("--")]
PAUSABLE = {
    "aim": KeepProjectileInReceiver,
    "clearance": KeepSourceRimAboveReceiverRim,
    "align": AlignPlanes,
    "pacing": TiltPacing,
    "fill": FillByTransferTask,
}
import functools
from dataclasses import replace
import experiments.real_pouring.controller_ablation as ablation
from giskardpy.motion_statechart.data_types import DefaultWeights

options = dict(
    a[2:].split("=") for a in sys.argv[4:] if a.startswith("--") and "=" in a
)
base = ablation.ProbeSettings()
overrides = {}
if "return" in options:
    overrides["return_rate"] = math.radians(float(options["return"]))
if "stop" in options:
    overrides["stopping_velocity"] = float(options["stop"])
ablation.ProbeSettings = functools.partial(replace, base, **overrides)
if "align_weight" in options:
    ablation.AlignPlanes = functools.partial(
        AlignPlanes, weight=getattr(DefaultWeights, options["align_weight"])
    )
if "accel" in options:
    ablation.TiltPacing = functools.partial(
        TiltPacing, maximum_acceleration=math.radians(float(options["accel"]))
    )
if "fill_weight" in options:
    ablation.FillByTransferTask = functools.partial(
        FillByTransferTask, weight=getattr(DefaultWeights, options["fill_weight"])
    )
world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
Tracy.from_world(world)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap

configuration = QPControllerConfig(
    target_frequency=80, prediction_horizon=30, verbose=False
)
if "heavy" in options:
    for dof in world.degrees_of_freedom:
        name = str(dof.name)
        if "left_" in name and "wrist_3" not in name and "knuckle" not in name:
            configuration.set_dof_weights(
                dof.name,
                DerivativeMap(None, 0.01 * float(options["heavy"]), None, None),
            )
probe = ControllerProbe.build(world, 0.3, configuration=configuration)
rec = PourRecording.read(bag)
arms = ArmSample.read(bag)
trace = rec.live_trace
times = numpy.array([a.time for a in arms])


def apply_arm(t):
    arms[
        max(0, int(numpy.searchsorted(times, t + rec.pour_start, side="right")) - 1)
    ].apply(world)


# receiver placement as in CommandEvaluation
apply_arm(t_start)
local_source = world.compute_forward_kinematics_np(world.root, probe.source.root)
i0 = int(numpy.argmin([abs(p.stamp - rec.pour_start - t_start) for p in rec.cup_poses]))
pose = rec.cup_poses[i0]
recorded = numpy.eye(4)
recorded[:3, :3] = pose.rotation
recorded[:3, 3] = pose.position
local_from_recording = local_source @ numpy.linalg.inv(recorded)
receiver = local_from_recording @ numpy.append(rec.receiver_position, 1.0)
probe.receiver_joint.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
    *receiver[:3], reference_frame=world.root
)
nodes = probe.executor.motion_statechart.nodes
lifecycle = probe.executor.motion_statechart.life_cycle_state
paused = [n for n in nodes if any(isinstance(n, PAUSABLE[p]) for p in paused_names)]
trace_times = numpy.array([s["time"] - rec.pour_start for s in trace])
conversion = GRAINS_PER_FILL * probe.receiver.capacity / probe.source.capacity
dt = 1 / 80
start_pos = world.compute_forward_kinematics_np(world.root, probe.source.root)[
    :3, 3
].copy()
names = [c.name for c in []]
out = []
for k in range(int(duration / dt)):
    t = t_start + k * dt
    s = trace[max(0, int(numpy.searchsorted(trace_times, t, side="right")) - 1)]
    probe.source.fill_connection.position = min(
        1.0, max(0.0, s["source_grains"] / GRAINS_PER_FILL)
    )
    probe.receiver.fill_connection.position = min(1.0, s["told_grains"] / conversion)
    for n in paused:
        lifecycle[n] = LifeCycleValues.PAUSED
    c = s["corrections"]
    r = probe.advance(
        DrainCorrectionValues(
            c["repose_correction"], c["log_drain_scale"], c["log_discharge_scale"]
        )
    )
    fk = world.compute_forward_kinematics_np(world.root, probe.source.root)
    tool = world.compute_forward_kinematics_np(
        world.root, world.get_body_by_name("l_gripper_tool_frame")
    )
    align = math.degrees(math.acos(numpy.clip(tool[0, 2], -1, 1)))
    out.append(
        (
            t,
            math.degrees(r.tilt_velocity),
            (fk[:3, 3] - start_pos) * 100,
            align,
            numpy.degrees(r.velocities),
        )
    )
names = (
    [str(v.name).split("/")[-1] for v in world.state.position_float_variables]
    if False
    else None
)
arm_idx = [
    i for i, dof in enumerate(world.degrees_of_freedom) if "left_" in str(dof.name)
]
dof_names = [
    str(world.degrees_of_freedom[i].name).replace("left_", "") for i in arm_idx
]
D = numpy.array([o[2] for o in out])
A = numpy.array([o[3] for o in out])
V = numpy.array([o[4] for o in out])
w3 = (
    arm_idx[dof_names.index("tracy/wrist_3_joint")]
    if "tracy/wrist_3_joint" in dof_names
    else arm_idx[-1]
)
others = [i for i in arm_idx if i != w3]
print(
    f"SUMMARY {options} paused={paused_names or 'none'}: max|dx| {abs(D[:,0]).max():.1f} max|dy| {abs(D[:,1]).max():.1f} max|dz| {abs(D[:,2]).max():.1f} cm, max align {A.max():.2f} deg, tilt at end {sum(o[1] for o in out)*dt:+.1f} deg, max other-joint speed {abs(V[:, others]).max():.1f} deg/s, wrist_3 peak {abs(V[:, w3]).max():.1f}"
)
if "--quiet" in sys.argv:
    raise SystemExit
print("paused:", paused_names or "none")
print(
    "   t   tiltrate   dx    dy    dz  align | "
    + " ".join(f"{n[:9]:>9}" for n in dof_names)
)
for t, tr, d, al, v in out[::4]:
    print(
        f"{t:5.2f} {tr:7.1f} {d[0]:5.1f} {d[1]:5.1f} {d[2]:5.1f} {al:5.2f} | "
        + " ".join(f"{v[i]:9.1f}" for i in arm_idx)
    )
