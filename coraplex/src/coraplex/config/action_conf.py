from datetime import timedelta


class ActionConfig:
    pick_up_prepose_distance = 0.03

    grasping_prepose_distance = 0.03

    navigate_keep_joint_states = True

    face_at_keep_joint_states = True

    transport_look_at_operation_site = False
    """
    Whether a transport looks at the object it picks up and at the target it places it
    on.
    """

    execution_delay: timedelta = timedelta(seconds=0.0)
    """
    The delay between the execution of actions/motions to imitate real world execution
    time.
    """
