"""
Replay the five real demo bags through the live bank exactly as the robot would run it.
"""

import glob, json, math, pickle, sys
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
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from scipy.spatial.transform import Rotation

WITHOUT_CROSSING = len(sys.argv) > 1 and sys.argv[1] == "without_crossing"
FULL_G, GRAINS = 2.65, 100.0  # one bank grain = 1 % of a full cup
PIXELS_PER_GRAM_30HZ = {
    "15-35-41": 32485,
    "15-38-01": 31985,
    "15-40-29": 35545,
    "15-42-23": 35005,
    "15-44-51": 37655,
}
DEMO = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)


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
    x, y, z, qx, qy, qz, qw = p[0], p[1], p[2], q[0], q[1], q[2], q[3]
    connection.origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=x,
        pos_y=y,
        pos_z=z,
        quat_x=qx,
        quat_y=qy,
        quat_z=qz,
        quat_w=qw,
        reference_frame=world.root,
    )


decode = lambda b: cv2.imdecode(numpy.frombuffer(b, numpy.uint8), cv2.IMREAD_COLOR)
results = {}
for d in sorted(glob.glob("demo/*")):
    run = d[-8:]
    o = pickle.load(open(d + "/extracted.pkl", "rb"))
    k = pickle.load(open(d + "/kin.pkl", "rb"))
    rows = [r for r in k["rows"] if "cup_p" in r and "receiver_p" in r]
    T = numpy.array([r["t"] for r in rows])
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
        outflow_rate_constant=DEMO.outflow_rate_constant,
        discharge_coefficient=DEMO.discharge_coefficient,
    )
    receiver.initialize_fill_level(world=world, initial_fill=0.0)
    receiver.receive_outflow_from(source=source, world=world)
    pixels_per_gram = (
        numpy.mean([v for r, v in PIXELS_PER_GRAM_30HZ.items() if r != run]) / 30.0
    )  # pixel-seconds per gram, from the other pours
    r0 = k["rows"][0]
    cam_T_map = numpy.eye(4)
    cam_T_map[:3, :3] = r0["map_to_cam_R"]
    cam_T_map[:3, 3] = r0["map_to_cam_p"]
    detector = StreamDetector(
        view=CameraView(intrinsics=k["K"], world_T_camera=numpy.linalg.inv(cam_T_map)),
        opening=OpeningRegion(
            centre=rows[0]["receiver_p"] + [0, 0, 0.16], radius=0.035
        ),
        segmentation=WhiteSegmentation(),
        pixel_seconds_per_grain=pixels_per_gram * FULL_G / GRAINS,
        frame_period=1 / 30,
    )
    detector.learn_background([decode(im[2]) for im in o["images"][:15]])
    bank = LiveBank.for_cups(
        source=source,
        receiver=receiver,
        world=world,
        built=DEMO,
        settings=BankSettings(),
        camera=CameraCondition.POLE_COUNTING.model,
        grains_per_fill=GRAINS,
        loaded_grains=GRAINS,
        goal_grains=70.0,
        opening_radius=0.035,
        prediction_window=1.5,
    )
    start = T[0]
    first_stream = None
    for i, im in enumerate(o["images"]):
        if im[0] < start:
            continue
        seen = detector.observe(decode(im[2]), stamp=im[0] - start)
        if i % 3:
            continue
        r = rows[int(numpy.argmin(abs(T - im[0])))]
        place(world, source_joint, r["cup_p"], r["cup_R"])
        if WITHOUT_CROSSING:
            seen = type(seen)(
                stamp=seen.stamp,
                stream_seen=seen.stream_seen,
                counted_grains=seen.counted_grains,
                crossing=None,
            )
        step = bank.observe(seen)
        if seen.stream_seen and first_stream is None:
            first_stream = (round(seen.stamp, 1), round(math.degrees(step.tilt)))
    w = json.load(open(d + "/weighed_particles.json"))
    commits = [
        (
            c.hypothesis.label,
            round(c.time, 1),
            round(math.degrees(next(s.tilt for s in bank.trace if s.time >= c.time))),
        )
        for c in bank.coupling.commits
    ]
    last = bank.trace[-1]
    results[run] = dict(
        first_stream=first_stream,
        commits=commits,
        active=last.active,
        repose_correction_deg=round(
            math.degrees(last.corrections.repose_correction), 1
        ),
        drain_scale=round(math.exp(last.corrections.log_drain_scale), 2),
        discharge_scale=round(math.exp(last.corrections.log_discharge_scale), 2),
        counted=round(last.counted_grains, 1),
        received_estimate=round(last.received_grains, 1),
        weighed=round(
            (w["received_mass_grams"] + w["spilled_mass_grams"]) / FULL_G * GRAINS, 1
        ),
    )
    print(run, json.dumps(results[run]), flush=True)
    del o
json.dump(
    results,
    open(
        (
            "live_bank_replay_without_crossing.json"
            if WITHOUT_CROSSING
            else "live_bank_replay.json"
        ),
        "w",
    ),
    indent=1,
)
