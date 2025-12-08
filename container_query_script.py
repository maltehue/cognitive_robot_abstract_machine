import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from giskardpy.executor import Executor
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.entity_query_language.entity import entity, let
from krrood.entity_query_language.quantify_entity import an
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.procthor.procthor_semantic_annotations import Milk
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Container,
    Door,
    Drawer,
    Fridge,
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


@dataclass
class ContainerInfo:
    """
    Data class to hold information about a container-like object.
    """

    annotation: SemanticAnnotation
    handle: Handle
    joint_name: str


class ContainerDemo:
    """
    Demonstrates finding and opening container-like objects in a semantic world.
    """

    def __init__(self):
        self.resource_dir = os.path.join(
            os.path.dirname(__file__), "pycram", "resources"
        )
        self.world = self._setup_world()
        self._reason_about_world()
        self._setup_ros()
        self.executor = Executor(world=self.world)

    def _setup_world(self):
        """
        Sets up the world by loading URDFs, STLs, and merging them.
        """
        pr2_sem_world = URDFParser.from_file(
            os.path.join(self.resource_dir, "robots", "pr2_calibrated_with_ft.urdf")
        ).parse()

        apartment_world = URDFParser.from_file(
            os.path.join(self.resource_dir, "worlds", "apartment.urdf")
        ).parse()

        milk_world = STLParser(
            os.path.join(self.resource_dir, "objects", "milk.stl")
        ).parse()

        cereal_world = STLParser(
            os.path.join(self.resource_dir, "objects", "breakfast_cereal.stl")
        ).parse()

        # Merge worlds
        apartment_world.merge_world(milk_world)
        apartment_world.merge_world(cereal_world)

        with apartment_world.modify_world():
            pr2_root = pr2_sem_world.get_body_by_name("base_footprint")
            apartment_root = apartment_world.root
            c_root_bf = OmniDrive.create_with_dofs(
                parent=apartment_root, child=pr2_root, world=apartment_world
            )
            apartment_world.merge_world(pr2_sem_world, c_root_bf)
            c_root_bf.origin = TransformationMatrix.from_xyz_rpy(1.5, 2.5, 0)

        # Set positions for objects
        apartment_world.get_body_by_name("milk.stl").parent_connection.origin = (
            TransformationMatrix.from_xyz_rpy(
                2.37, 2, 1.05, reference_frame=apartment_world.root
            )
        )
        apartment_world.get_body_by_name(
            "breakfast_cereal.stl"
        ).parent_connection.origin = TransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=apartment_world.root
        )

        # Add semantic annotation for milk
        milk_view = Milk(body=apartment_world.get_body_by_name("milk.stl"))
        with apartment_world.modify_world():
            apartment_world.add_semantic_annotation(milk_view)

        return apartment_world

    def _reason_about_world(self):
        """
        Runs the WorldReasoner to infer semantic annotations.
        """
        with self.world.modify_world():
            world_reasoner = WorldReasoner(self.world)
            world_reasoner.reason()

    def _setup_ros(self):
        """
        Initializes ROS 2 node and publishers.
        """
        print("Initializing ROS 2 node for visualization...")
        rclpy.init()
        self.node = rclpy.create_node("container_query_node")
        self.viz_publisher = VizMarkerPublisher(node=self.node, world=self.world)
        self.path_publisher = self.node.create_publisher(
            Marker, "/semworld/handle_path", 10
        )

    def find_containers(self) -> List[ContainerInfo]:
        """
        Queries the world for Drawer, Fridge, and Door annotations and returns ContainerInfo objects.
        """
        drawers = list(
            an(
                entity(d := let(Drawer, domain=self.world.semantic_annotations))
            ).evaluate()
        )
        fridges = list(
            an(
                entity(f := let(Fridge, domain=self.world.semantic_annotations))
            ).evaluate()
        )
        doors = list(
            an(
                entity(d := let(Door, domain=self.world.semantic_annotations))
            ).evaluate()
        )

        results = drawers + fridges + doors
        container_infos = []

        for annotation in results:
            info = self._extract_container_info(annotation)
            if info:
                container_infos.append(info)

        return container_infos

    def _extract_container_info(
        self, annotation: SemanticAnnotation
    ) -> Optional[ContainerInfo]:
        """
        Extracts body, handle, and joint info from a semantic annotation.
        """
        if isinstance(annotation, Drawer):
            body = annotation.container.body
            handle = annotation.handle
        elif isinstance(annotation, Fridge):
            body = annotation.door.body
            handle = annotation.door.handle
        elif isinstance(annotation, Door):
            body = annotation.body
            handle = annotation.handle
        else:
            return None

        joint_name = "None"
        if body.parent_connection:
            connection = body.parent_connection
            if isinstance(connection, (PrismaticConnection, RevoluteConnection)):
                joint_name = connection.name.name

        return ContainerInfo(
            annotation=annotation, handle=handle, joint_name=joint_name
        )

    def run(self):
        """
        Main execution loop: finds containers and opens them.
        """
        containers = self.find_containers()
        print(f"Found {len(containers)} openable objects:")

        for container in containers:
            self._process_container(container)

    def _process_container(self, container: ContainerInfo):
        """
        Prints container info and triggers the open action.
        """
        print(f"Container: {container.annotation.name.name}")
        print(f"  Joint: {container.joint_name}")
        print(f"  Handle: {container.handle.name.name}")

        if container.handle and container.joint_name != "None":
            print(f"  -> Opening {container.annotation.name.name}...")
            self._open_container(container)
            print(f"  -> Finished opening {container.annotation.name.name}.")

    def _open_container(self, container: ContainerInfo):
        """
        Executes the Open goal for the given container.
        """
        initial_state_data = self.world.state.data.copy()

        open_goal = Open(
            tip_link=container.handle.body,
            environment_link=container.handle.body,
            goal_joint_state=1.0,
        )

        msc = MotionStatechart()
        msc.add_node(open_goal)
        msc.add_node(EndMotion.when_true(open_goal))

        self.executor.compile(motion_statechart=msc)

        handle_path_points = []
        timeout = 500
        for _ in range(timeout):
            self.executor.tick()
            self._visualize_handle_path(container.handle, handle_path_points)

            if msc.is_end_motion():
                break
        else:
            print("Timeout reached.")

        # Reset state
        self.world.state.data = initial_state_data
        self.world.notify_state_change()

    def _visualize_handle_path(self, handle: Handle, points: List[Point]):
        """
        Computes FK for handle and publishes path marker.
        """
        handle_transform = self.world.compute_forward_kinematics(
            self.world.root, handle.body
        )
        pos = handle_transform.to_np()[:3, 3]
        points.append(Point(x=pos[0], y=pos[1], z=pos[2]))

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "handle_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        marker.points = points
        self.path_publisher.publish(marker)

    def cleanup(self):
        """
        Shuts down ROS node.
        """
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    demo = ContainerDemo()
    try:
        demo.run()
    finally:
        demo.cleanup()
