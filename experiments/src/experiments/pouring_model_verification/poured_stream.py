"""
The stream a pouring model says leaves the source's lip, against the receiver's opening:
where it starts, how it is thrown, and how far along it a point on the opening plane
lies.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass

import numpy
from typing_extensions import List, Tuple

from experiments.pouring_model_verification.concept_revision import Launch
from experiments.pouring_model_verification.hypothesis_bank import StreamGeometry
from semantic_digital_twin.physics.equations.pouring_equations import STANDARD_GRAVITY
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.world import World

# %% the stream


MINIMUM_SPEED = 1e-9
"""
The least speed a direction is taken from, in metres per second, so a still stream has a
direction at all.
"""


@dataclass
class PouredStream:
    """
    The stream the source's drain model throws from its lip now, seen against the
    receiver's opening.
    """

    source: HasFillLevel
    """
    The container poured from.
    """

    receiver: HasFillLevel
    """
    The container poured into.
    """

    world: World
    """
    The world both stand in.
    """

    def exit_state(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        :return: Where the model says the stream leaves the lip now, and with which
            velocity, in the world frame.
        """
        speed = self.source.current_outflow_velocity(self.world)
        exit_point = self.source.liquid_exit_point(self.world)
        velocity = self.source.liquid_exit_velocity(self.world, speed)
        return (
            numpy.asarray(exit_point.evaluate(), dtype=float).ravel()[:3],
            numpy.asarray(velocity.evaluate(), dtype=float).ravel()[:3],
        )

    @property
    def opening_height(self) -> float:
        """
        :return: How high the receiver's opening stands in the world, in metres.
        """
        base = self.world.compute_forward_kinematics_np(
            self.world.root, self.receiver.root
        )[2, 3]
        return float(base) + self.receiver.root.collision.height

    def launch(self) -> Launch:
        """
        :return: The stream as the model predicts it leaves the lip now.
        """
        exit_point, velocity = self.exit_state()
        speed = float(numpy.linalg.norm(velocity))
        return Launch(
            speed=speed,
            horizontal_share=float(numpy.linalg.norm(velocity[:2]))
            / max(speed, MINIMUM_SPEED),
            vertical_share=float(velocity[2]) / max(speed, MINIMUM_SPEED),
            drop=max(0.0, float(exit_point[2]) - self.opening_height),
            gravity=STANDARD_GRAVITY,
        )

    def geometry(self, opening_radius: float) -> StreamGeometry:
        """
        :param opening_radius: The receiver opening's radius, in metres.
        :return: How the model says the stream leaves the lip now, and where the
            receiver's centre lies from there.
        """
        exit_point, velocity = self.exit_state()
        speed = max(float(numpy.linalg.norm(velocity)), MINIMUM_SPEED)
        horizontal = float(numpy.linalg.norm(velocity[:2]))
        direction = velocity[:2] / max(horizontal, MINIMUM_SPEED)
        centre = self.world.compute_forward_kinematics_np(
            self.world.root, self.receiver.root
        )[:2, 3]
        return StreamGeometry(
            horizontal_share=horizontal / speed,
            vertical_share=float(velocity[2]) / speed,
            drop=max(0.0, float(exit_point[2]) - self.opening_height),
            centre_distance=float((centre - exit_point[:2]) @ direction),
            opening_radius=opening_radius,
        )

    def distance_from_lip(self, points: List[numpy.ndarray]) -> float:
        """
        :param points: Horizontal points on the opening plane, in metres.
        :return: How far from the lip, along the stream's horizontal direction, they
            lie on average, in metres.
        """
        exit_point, velocity = self.exit_state()
        direction = velocity[:2] / max(numpy.linalg.norm(velocity[:2]), MINIMUM_SPEED)
        return float(
            numpy.mean([(point - exit_point[:2]) @ direction for point in points])
        )
