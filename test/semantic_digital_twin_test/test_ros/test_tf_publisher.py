import pytest
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_py import LookupException

from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    HomogeneousTransformationMatrixToRos2Converter,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body


def test_tf_publisher(rclpy_node, pr2_world_state_reset):
    r_gripper_tool_frame = (
        pr2_world_state_reset.get_kinematic_structure_entities_by_name(
            "r_gripper_tool_frame"
        )
    )
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=pr2_world_state_reset,
        ignored_kinematic_structure_entities=r_gripper_tool_frame,
    )

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
    transform = tf_wrapper.lookup_transform("odom_combined", "pr2/base_footprint")
    odom_combined = pr2_world_state_reset.get_kinematic_structure_entities_by_name(
        "odom_combined"
    )[0]
    base_footprint = pr2_world_state_reset.get_kinematic_structure_entities_by_name(
        "base_footprint"
    )[0]
    fk = pr2_world_state_reset.compute_forward_kinematics(odom_combined, base_footprint)
    transform2 = HomogeneousTransformationMatrixToRos2Converter.convert(fk)
    assert transform.transform == transform2.transform

    with pytest.raises(LookupException):
        tf_wrapper.lookup_transform("odom_combined", "pr2/r_gripper_tool_frame")


def test_tf_publisher_ignore_robot(rclpy_node, pr2_world_setup):
    with pr2_world_setup.modify_world():
        box = Body(name=PrefixedName("box"))
        c = FixedConnection(parent=pr2_world_setup.root, child=box)
        pr2_world_setup.add_connection(c)
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher.create_with_ignore_robot(
        node=rclpy_node,
        robot=pr2_world_setup.get_semantic_annotations_by_type(AbstractRobot)[0],
    )

    assert not tf_wrapper.wait_for_transform(
        "odom_combined",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
    with pytest.raises(LookupException):
        transform = tf_wrapper.lookup_transform("odom_combined", "pr2/base_footprint")

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        "box",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
    transform = tf_wrapper.lookup_transform("odom_combined", "box")

    odom_combined = pr2_world_setup.get_kinematic_structure_entities_by_name(
        "odom_combined"
    )[0]
    base_footprint = pr2_world_setup.get_kinematic_structure_entities_by_name("box")[0]
    fk = pr2_world_setup.compute_forward_kinematics(odom_combined, base_footprint)
    transform2 = HomogeneousTransformationMatrixToRos2Converter.convert(fk)
    assert transform.transform == transform2.transform


def test_tf_publisher_kitchen(rclpy_node, pr2_apartment_world):
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(
        node=rclpy_node,
        world=pr2_apartment_world,
    )
    VizMarkerPublisher(world=pr2_apartment_world, node=rclpy_node)

    milk = pr2_apartment_world.get_kinematic_structure_entities_by_name("milk.stl")[0]

    assert tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # delete dofs
    with pr2_apartment_world.modify_world():
        new_connection = FixedConnection(
            parent=milk.parent_kinematic_structure_entity, child=milk
        )
        pr2_apartment_world.remove_connection(milk.parent_connection)
        pr2_apartment_world.add_connection(new_connection)

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # add dofs
    with pr2_apartment_world.modify_world():
        new_connection = Connection6DoF.create_with_dofs(
            world=pr2_apartment_world,
            parent=milk.parent_kinematic_structure_entity,
            child=milk,
        )
        pr2_apartment_world.remove_connection(milk.parent_connection)
        pr2_apartment_world.add_connection(new_connection)

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # delete body
    with pr2_apartment_world.modify_world():
        pr2_apartment_world.remove_connection(milk.parent_connection)
        pr2_apartment_world.remove_kinematic_structure_entity(milk)

    # make sure wrapper forgets about it too
    tf_wrapper.tf_buffer.clear()
    assert not tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )

    # add body back
    with pr2_apartment_world.modify_world():
        new_connection = Connection6DoF.create_with_dofs(
            world=pr2_apartment_world,
            parent=pr2_apartment_world.root,
            child=milk,
        )
        pr2_apartment_world.add_connection(new_connection)

    assert tf_wrapper.wait_for_transform(
        "odom_combined",
        str(milk.name),
        timeout=Duration(seconds=1.0),
        time=Time(),
    )
