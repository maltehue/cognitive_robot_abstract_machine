"""
Source-cup tilt over time from a demo bag's /tf (needs a sourced ROS 2); writes JSON per
bag.
"""

import json, sys
from pathlib import Path
import numpy, rosbag2_py
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from tf2_ros import BufferCore

out = Path(sys.argv[1])
for bag in map(Path, sys.argv[2:]):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    buf = BufferCore(Duration(seconds=3600))
    stamps = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ("/tf", "/tf_static"):
            continue
        msg = deserialize_message(data, types[topic])
        for tr in msg.transforms:
            (buf.set_transform_static if topic == "/tf_static" else buf.set_transform)(
                tr, "bag"
            )
            if topic == "/tf":
                stamps.append(tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9)
    stamps = numpy.unique(numpy.round(stamps, 3))
    rows = []
    for s in stamps:
        t = Time(nanoseconds=int(s * 1e9))
        if not buf.can_transform_core("map", "source_cup", t)[0]:
            continue
        q = buf.lookup_transform_core("map", "source_cup", t).transform.rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        rows.append(
            (float(s), float(numpy.degrees(numpy.arccos(numpy.clip(R[2, 2], -1, 1)))))
        )
    (out / (bag.name[:19] + ".json")).write_text(json.dumps(rows))
    print(
        bag.name[:40],
        len(rows),
        "poses",
        "tilt %.1f..%.1f" % (min(r[1] for r in rows), max(r[1] for r in rows)),
    )
