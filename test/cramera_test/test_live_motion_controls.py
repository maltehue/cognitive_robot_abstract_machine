from cramera.live.bridge import Bridge, JointMoveRequest
from cramera.live.visualization import BridgePlanCallback
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from .test_live_transforms import make_kitchen


# %% viewer controls during stable motion
def test_joint_move_is_applied_without_a_new_chart_history_entry() -> None:
    world, hinge = make_kitchen()
    bridge = Bridge()
    bridge.attach(world)
    callback = BridgePlanCallback(bridge=bridge)
    chart = MotionStatechart()
    request = JointMoveRequest(connection_name=str(hinge.name), position=0.7)
    bridge.queue_joint_move(request)

    callback.on_motion_tick(chart)

    assert hinge.position == request.position
    assert len(chart.history) == 0
