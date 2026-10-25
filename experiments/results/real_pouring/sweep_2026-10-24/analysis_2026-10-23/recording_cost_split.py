"""
Per-cycle time of the demo's transfer tasks with and without control-cycle recording
(debug values included).
"""

import functools, statistics, sys, time
import experiments.real_pouring.controller_ablation as ablation
from experiments.real_pouring.controller_ablation import ControllerProbe
from giskardpy.executor import Executor
from giskardpy.qp.control_cycle_recording import ControlCycleRecorder
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.tracy import Tracy

recording = sys.argv[1] in ("on", "nodebug")
if sys.argv[1] == "nodebug":
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart

    MotionStatechart.collect_debug_expressions = lambda self: []
if recording:
    ablation.Executor = functools.partial(
        Executor, control_cycle_recorder=ControlCycleRecorder()
    )
world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
Tracy.from_world(world)
start = time.perf_counter()
probe = ControllerProbe.build(world, 0.3)
compile_seconds = time.perf_counter() - start
executor = probe.executor
durations = []
for _ in range(400):
    t0 = time.perf_counter()
    executor.tick()
    durations.append(time.perf_counter() - t0)
durations = durations[20:]
build = ""
if recording:
    t0 = time.perf_counter()
    rec = executor.control_cycle_recorder.build_recording()
    rec.save(
        "/tmp/claude-1000/-home-malte-libs-cognitive-robot-abstract-machine/c06b6632-9071-439f-a55d-5344755d80e9/scratchpad/an1023/cost_recording.npz"
    )
    build = f"; build+save {time.perf_counter() - t0:.2f}s for {rec.number_of_cycles} cycles, debug columns {rec.debug_columns}"
print(
    f"recording {sys.argv[1]}: compile {compile_seconds:.1f}s, tick mean {statistics.mean(durations)*1000:.2f} ms, median {statistics.median(durations)*1000:.2f}, p95 {sorted(durations)[int(0.95*len(durations))]*1000:.2f}, max {max(durations)*1000:.2f}{build}"
)
