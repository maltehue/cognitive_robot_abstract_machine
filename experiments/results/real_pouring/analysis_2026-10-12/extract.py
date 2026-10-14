import sys, pickle, numpy as np, rosbag2_py
from pathlib import Path
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def extract(bag_dir: Path):
    r = rosbag2_py.SequentialReader()
    r.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
    out = dict(js=[], js_names=None, tf={}, tf_static=[], images=[], camera_info=None)
    while r.has_next():
        topic, data, t_rec = r.read_next()
        m = deserialize_message(data, types[topic])
        if topic == "/left_arm/joint_states":
            out["js_names"] = list(m.name)
            out["js"].append(
                [stamp(m.header), t_rec * 1e-9] + list(m.position) + list(m.velocity)
            )
        elif topic == "/tf":
            for tr in m.transforms:
                q = tr.transform.rotation
                p = tr.transform.translation
                out["tf"].setdefault(
                    (tr.header.frame_id, tr.child_frame_id), []
                ).append((stamp(tr.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        elif topic == "/tf_static":
            for tr in m.transforms:
                q = tr.transform.rotation
                p = tr.transform.translation
                out["tf_static"].append(
                    (
                        tr.header.frame_id,
                        tr.child_frame_id,
                        p.x,
                        p.y,
                        p.z,
                        q.x,
                        q.y,
                        q.z,
                        q.w,
                    )
                )
        elif topic.endswith("compressed"):
            out["images"].append((stamp(m.header), t_rec * 1e-9, bytes(m.data)))
        elif topic.endswith("camera_info") and out["camera_info"] is None:
            out["camera_info"] = dict(
                K=list(m.k),
                D=list(m.d),
                w=m.width,
                h=m.height,
                frame=m.header.frame_id,
                model=m.distortion_model,
            )
    out["js"] = np.array(out["js"])
    out["tf"] = {k: np.array(v) for k, v in out["tf"].items()}
    return out


for d in sys.argv[1:]:
    d = Path(d)
    o = extract(d)
    pickle.dump(o, open(d / "extracted.pkl", "wb"))
    print(
        d.name,
        len(o["js"]),
        len(o["images"]),
        sorted({k[1] for k in o["tf"]})[:60],
        o["camera_info"] and o["camera_info"]["K"],
    )
