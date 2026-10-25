import sys, numpy, rosbag2_py
from pathlib import Path
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from experiments.real_pouring.pour_replay import PourRecording

bag = Path(sys.argv[1])
a, b = float(sys.argv[2]), float(sys.argv[3])
start = PourRecording.read(bag).pour_start
r = rosbag2_py.SequentialReader()
r.open(
    rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
    rosbag2_py.ConverterOptions("cdr", "cdr"),
)
types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
T, P, names = [], [], None
while r.has_next():
    topic, data, _ = r.read_next()
    if topic != "/left_arm/joint_states":
        continue
    m = deserialize_message(data, types[topic])
    names = [n.replace("left_", "") for n in m.name]
    T.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 - start)
    P.append(list(m.position))
T = numpy.array(T)
P = numpy.degrees(numpy.array(P))
V = numpy.gradient(P, T, axis=0)
print("t     " + " ".join(f"{n[:11]:>11}" for n in names))
for t in numpy.arange(a, b, 0.1):
    i = numpy.searchsorted(T, t)
    print(f"{t:5.1f} " + " ".join(f"{v:11.1f}" for v in V[i]))
