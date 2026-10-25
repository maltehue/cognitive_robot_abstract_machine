"""
Replay recorded states into the offline controller and read the fill row's prediction up
to the first turn-back.
"""

import json, math, sys
from pathlib import Path
import numpy
import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import VariableParameters
from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.command_evaluation import ArmSample
from experiments.real_pouring.controller_ablation import ControllerProbe
from experiments.real_pouring.pour_replay import PourRecording, GRAINS_PER_FILL
from giskardpy.qp.terminal_state_prediction_strategy import (
    TerminalStatePredictionConstraint,
    TerminalStatePredictionStrategy,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

bag = Path(sys.argv[1])
a, b = float(sys.argv[2]), float(sys.argv[3])
world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
Tracy.from_world(world)
probe = ControllerProbe.build(
    world, json.loads((bag / "pour_settings.json").read_text())["goal_fill"]
)
rec = PourRecording.read(bag)
arms = ArmSample.read(bag)
trace = rec.live_trace
times = numpy.array([s.time for s in arms])


def apply_arm(t):
    arms[
        max(0, int(numpy.searchsorted(times, t + rec.pour_start, side="right")) - 1)
    ].apply(world)


apply_arm(a)
local_source = world.compute_forward_kinematics_np(world.root, probe.source.root)
i0 = int(numpy.argmin([abs(p.stamp - rec.pour_start - a) for p in rec.cup_poses]))
pose = rec.cup_poses[i0]
recorded = numpy.eye(4)
recorded[:3, :3] = pose.rotation
recorded[:3, 3] = pose.position
receiver = (
    local_source @ numpy.linalg.inv(recorded) @ numpy.append(rec.receiver_position, 1.0)
)
probe.receiver_joint.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
    *receiver[:3], reference_frame=world.root
)
controller = probe.executor.qp_controller
[row] = [
    c
    for c in controller.constraint_collection._constraints
    if isinstance(c, TerminalStatePredictionConstraint)
]
strategy = TerminalStatePredictionStrategy(
    degrees_of_freedom=(
        controller.active_dofs if hasattr(controller, "active_dofs") else []
    ),
    constraints=[row],
    qp_controller_config=controller.config,
)
context = probe.executor.context


def compiled(expression):
    f = sm.Vector([expression]).compile(
        parameters=VariableParameters.from_lists(
            world.state.position_float_variables, context.float_variable_data.variables
        ),
        sparse=False,
    )
    f.bind_args_to_memory_view(0, world.state.positions)
    context.float_variable_data.bind_argument(f, 1)
    return f


coupled = compiled(strategy._free_response())
held = compiled(strategy._state_model.free_response())
bound = compiled(strategy.create_equality_bounds()[0])
source_rate = compiled(row.coupled_states[0].velocity) if row.coupled_states else None
inflow = compiled(row.expression)
tt = numpy.array([s["time"] - rec.pour_start for s in trace])
conversion = GRAINS_PER_FILL * probe.receiver.capacity / probe.source.capacity
goal = row.goal_value
rows = []
print(f"{bag.name}: goal {goal}; window {row.prediction_duration} s")
print(
    "    t   tilt seen  told   inflow/s  pred_held pred_coupled  bound   told+1.5s  rep  srcfill"
)
for k, s in enumerate(trace):
    t = tt[k]
    if not a <= t <= b:
        continue
    apply_arm(t)
    probe.source.fill_connection.position = min(
        1.0, max(0.0, s["source_grains"] / GRAINS_PER_FILL)
    )
    probe.receiver.fill_connection.position = min(1.0, s["told_grains"] / conversion)
    c = s["corrections"]
    DrainCorrectionValues(
        c["repose_correction"], c["log_drain_scale"], c["log_discharge_scale"]
    ).write_to(world, probe.source.fill_connection.drain_correction)
    probe.fill_task.on_tick(context)
    later = s["told_grains"] / G if False else None
    j = int(numpy.searchsorted(tt, t + 1.5))
    future = trace[min(j, len(trace) - 1)]["told_grains"] / conversion
    rows.append(
        (
            t,
            s["told_grains"] / conversion,
            float(inflow.evaluate()[0]),
            int(s["stream_seen"]),
            float(coupled.evaluate()[0]),
            future,
            math.degrees(s["tilt"]),
        )
    )
    if "--quiet" in sys.argv:
        continue
    print(
        f"{t:5.2f} {math.degrees(s['tilt']):5.1f}  {int(s['stream_seen'])}  {s['told_grains']/conversion:.3f}  {float(inflow.evaluate()[0]):7.4f}   {float(held.evaluate()[0]):.3f}     {float(coupled.evaluate()[0]):.3f}     {float(bound.evaluate()[0]):+.3f}    {future:.3f}   {math.degrees(c['repose_correction']):4.1f}  {s['source_grains']/GRAINS_PER_FILL:.3f}"
    )

R = numpy.array(rows)
modeled = float(numpy.sum(R[1:, 2] * numpy.diff(R[:, 0])))
measured = R[-1, 1] - R[0, 1]
flowing = R[:, 2] > 0
print(
    f"SUMMARY {bag.name} {a:.1f}-{b:.1f}s: modeled inflow integrated {modeled:.3f}, measured fill gain {measured:.3f}, ratio {modeled / max(measured, 1e-6):.2f}; model rate while it predicts flow: mean {R[flowing, 2].mean():.3f}/s; seconds model flowing {numpy.sum(numpy.diff(R[:, 0])[flowing[1:]]):.1f}"
)

if "--forecast" in sys.argv:
    R = numpy.array(rows)
    usable = R[:, 0] <= b - 1.5
    out = Path(sys.argv[-1])
    out.write_text(json.dumps(R[usable].tolist()))
