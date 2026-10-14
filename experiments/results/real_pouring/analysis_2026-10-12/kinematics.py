import sys, pickle, numpy as np, rclpy.time, rclpy.duration, tf2_ros
from pathlib import Path
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation


def to_msg(parent, child, t, p, q):
    m = TransformStamped()
    m.header.frame_id = parent
    m.child_frame_id = child
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1000000000
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = p
    (
        m.transform.rotation.x,
        m.transform.rotation.y,
        m.transform.rotation.z,
        m.transform.rotation.w,
    ) = q
    return m


def lookup(buf, target, source, t):
    tr = buf.lookup_transform_core(
        target, source, rclpy.time.Time(nanoseconds=int(t * 1e9))
    )
    p = tr.transform.translation
    q = tr.transform.rotation
    return np.array([p.x, p.y, p.z]), Rotation.from_quat([q.x, q.y, q.z, q.w])


for d in sys.argv[1:]:
    d = Path(d)
    o = pickle.load(open(d / "extracted.pkl", "rb"))
    buf = tf2_ros.BufferCore(rclpy.duration.Duration(seconds=600))
    for s in o["tf_static"]:
        buf.set_transform_static(to_msg(s[0], s[1], 0, s[2:5], s[5:9]), "bag")
    for (parent, child), a in o["tf"].items():
        for row in a:
            buf.set_transform(to_msg(parent, child, row[0], row[1:4], row[4:8]), "bag")
    image_t = np.array([i[0] for i in o["images"]])
    js = o["js"]
    w3 = o["js_names"].index("left_wrist_3_joint")
    has_cup = any(k[1] == "source_cup" for k in o["tf"])
    t0, t1 = max(image_t[0], js[0, 0]), min(image_t[-1], js[-1, 0])
    grid = np.arange(t0 + 0.05, t1 - 0.05, 0.02)
    rows = []
    for t in grid:
        try:
            p_tool, R_tool = lookup(buf, "map", "l_gripper_tool_frame", t)
            p_cam, R_cam = lookup(buf, "camera_color_optical_frame", "map", t)
        except Exception as e:
            continue
        r = dict(
            t=t,
            tool_p=p_tool,
            tool_R=R_tool.as_matrix(),
            map_to_cam_p=p_cam,
            map_to_cam_R=R_cam.as_matrix(),
        )
        if has_cup:
            try:
                p_cup, R_cup = lookup(buf, "map", "source_cup", t)
                r["cup_p"] = p_cup
                r["cup_R"] = R_cup.as_matrix()
                r["source_fill"] = lookup(
                    buf, "source_cup", "source_cup_fill_level_phantom", t
                )[0][2]
                r["receiver_fill"] = lookup(
                    buf, "receiving_cup", "receiving_cup_fill_level_phantom", t
                )[0][2]
                r["receiver_p"] = lookup(buf, "map", "receiving_cup", t)[0]
            except Exception:
                pass
        rows.append(r)
    pickle.dump(
        dict(
            rows=rows,
            js=js,
            js_names=o["js_names"],
            image_t=image_t,
            K=np.array(o["camera_info"]["K"]).reshape(3, 3),
        ),
        open(d / "kin.pkl", "wb"),
    )
    print(d.name, len(rows), has_cup)
