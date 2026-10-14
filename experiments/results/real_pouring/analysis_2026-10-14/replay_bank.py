"""
Replay a 2026-10-14 live-bank bag through the bank offline, as the robot ran it: the
stream detector on the recorded frames, the cup posed by the recorded transforms, a
reading every third frame (10 Hz).

The controller is not in the loop, so the cup follows the recorded tilt whatever the
bank decides; what the replay shows is which hypothesis the bank runs on and what it
tells the controller the receiver holds, reading by reading.

Usage: python replay_bank.py <bag dir> <uncalibrated|fitted> The bag dir needs
extracted.pkl and kin.pkl (see README.md). Prints one line per reading and writes
replay_<bag>_<drain>.json next to this script.
"""

import json, math, pickle, sys
from pathlib import Path
import cv2, numpy
from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.pouring_model_verification.sensor import CameraCondition
from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.stream_detection import (
    CameraView,
    OpeningRegion,
    StreamDetector,
    WhiteSegmentation,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from scipy.spatial.transform import Rotation

DRAINS = {
    "uncalibrated": ModelPoint(
        repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
    ),
    "fitted": ModelPoint(
        repose_angle=math.radians(27.0),
        discharge_coefficient=0.2,
        outflow_rate_constant=2.03,
    ),
}
PIXEL_SECONDS_PER_GRAIN = 29.15  # the node's styrofoam default
GRAINS, GOAL_GRAINS, TOLERANCE_GRAINS = 100.0, 70.0, 5.0
bag, drain_name = Path(sys.argv[1]), sys.argv[2]
drain = DRAINS[drain_name]


def cup(world, name):
    body = Body(
        name=PrefixedName(name),
        collision=ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.08),
                    scale=Scale(0.07, 0.07, 0.16),
                )
            ]
        ),
    )
    with world.modify_world():
        world.add_body(body)
        connection = Connection6DoF.create_with_dofs(world, world.root, body)
        world.add_connection(connection)
    annotation = HasFillLevel(name=PrefixedName(name), root=body)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    return annotation, connection


def place(world, connection, p, R):
    q = Rotation.from_matrix(R).as_quat()
    connection.origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=p[0],
        pos_y=p[1],
        pos_z=p[2],
        quat_x=q[0],
        quat_y=q[1],
        quat_z=q[2],
        quat_w=q[3],
        reference_frame=world.root,
    )


decode = lambda b: cv2.imdecode(numpy.frombuffer(b, numpy.uint8), cv2.IMREAD_COLOR)
o = pickle.load(open(bag / "extracted.pkl", "rb"))
k = pickle.load(open(bag / "kin.pkl", "rb"))
rows = [r for r in k["rows"] if "cup_p" in r and "receiver_p" in r]
T = numpy.array([r["t"] for r in rows])
trace = json.load(open(bag / "live_bank_trace.json"))
world = World()
with world.modify_world():
    world.add_body(Body(name=PrefixedName("map")))
source, source_joint = cup(world, "source_cup")
receiver, receiver_joint = cup(world, "receiving_cup")
place(world, receiver_joint, rows[0]["receiver_p"], numpy.eye(3))
place(world, source_joint, rows[0]["cup_p"], rows[0]["cup_R"])
source.initialize_fill_level(
    world=world,
    initial_fill=1.0,
    outflow_rate_constant=drain.outflow_rate_constant,
    discharge_coefficient=drain.discharge_coefficient,
    repose_angle=drain.repose_angle,
)
receiver.initialize_fill_level(world=world, initial_fill=0.0)
receiver.receive_outflow_from(source=source, world=world)
r0 = k["rows"][0]
cam_T_map = numpy.eye(4)
cam_T_map[:3, :3] = r0["map_to_cam_R"]
cam_T_map[:3, 3] = r0["map_to_cam_p"]
detector = StreamDetector(
    view=CameraView(intrinsics=k["K"], world_T_camera=numpy.linalg.inv(cam_T_map)),
    opening=OpeningRegion(centre=rows[0]["receiver_p"] + [0, 0, 0.16], radius=0.035),
    segmentation=WhiteSegmentation(),
    pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
    frame_period=1 / 30,
)
start = trace[0]["time"]
detector.learn_background([decode(im[2]) for im in o["images"] if im[0] < start][-15:])
bank = LiveBank.for_cups(
    source=source,
    receiver=receiver,
    world=world,
    built=drain,
    settings=BankSettings(),
    camera=CameraCondition.POLE_COUNTING.model,
    grains_per_fill=GRAINS,
    loaded_grains=GRAINS,
    goal_grains=GOAL_GRAINS,
    opening_radius=0.035,
    prediction_window=1.5,
)
stop = None
steps = []
for i, im in enumerate(o["images"]):
    if im[0] < start or im[0] > trace[-1]["time"] + 0.05:
        continue
    seen = detector.observe(decode(im[2]), stamp=im[0] - start)
    if i % 3:
        continue
    r = rows[int(numpy.argmin(abs(T - im[0])))]
    place(world, source_joint, r["cup_p"], r["cup_R"])
    step = bank.observe(seen)
    told = step.received_grains + bank.coupling.margin
    if stop is None and told >= GOAL_GRAINS - TOLERANCE_GRAINS:
        stop = dict(
            time=round(seen.stamp, 2),
            tilt=round(math.degrees(step.tilt), 1),
            counted=round(step.counted_grains, 1),
            told=round(told, 1),
        )
    steps.append(
        dict(
            time=round(seen.stamp, 2),
            tilt=round(math.degrees(step.tilt), 1),
            active=step.active,
            counted=round(step.counted_grains, 1),
            received=round(step.received_grains, 1),
            told=round(told, 1),
            probabilities={h: round(p, 2) for h, p in step.probabilities.items()},
        )
    )
    print(
        f"{seen.stamp:5.1f} tilt {math.degrees(step.tilt):5.1f} count {step.counted_grains:5.1f} recv {step.received_grains:5.1f} told {told:5.1f} act {step.active} "
        + " ".join(f"{h}:{p:.2f}" for h, p in step.probabilities.items()),
        flush=True,
    )
print("would stop at", stop)
json.dump(
    dict(stop=stop, steps=steps),
    open(Path(__file__).parent / f"replay_{bag.name}_{drain_name}.json", "w"),
)
