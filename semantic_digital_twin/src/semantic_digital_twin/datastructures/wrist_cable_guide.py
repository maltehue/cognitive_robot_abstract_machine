from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from krrood.exceptions import DataclassException
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.soft_connections import (
    CosseratRodConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


# %% exceptions


@dataclass
class InvalidCableRouteError(DataclassException):
    """
    Raised when a cable route is requested with physically meaningless parameters.
    """

    slack_length: float
    """The requested slack length that made the route invalid."""

    segment_count: int
    """
    The requested number of Cosserat segments that made the route invalid.
    """

    def error_message(self) -> str:
        return (
            f"A cable route needs a non-negative slack length and at least one "
            f"segment, got slack_length={self.slack_length} and "
            f"segment_count={self.segment_count}."
        )

    def suggest_correction(self) -> str:
        return "Pass slack_length >= 0 and segment_count >= 1."


# %% cable model


@dataclass
class WristCableGuide:
    """
    Model of a cable routed around a robot wrist by a soft continuum mechanism.

    The cable is anchored on a link proximal to the wrist and guided past a body
    mounted on the distal (wrist output) link. Its body is represented by a
    :class:`~semantic_digital_twin.world_description.soft_connections.CosseratRodConnection`
    chain, whose longitudinal extension degrees of freedom carry the cable's stretch
    state.

    The routed span between the anchor and the guide changes as the wrist moves. When
    it exceeds the cable's rest length the cable is under tension; the difference is the
    stretch this model exposes.
    """

    name: PrefixedName
    """Identifier of this cable guide within its world."""

    cable_anchor: Body
    """
    Body where the cable is fixed on the link proximal to the wrist.
    """

    wrist_guide: Body
    """Body the cable is guided past on the distal (wrist output) link."""

    cable_segments: list[Body]
    """
    Rod segment bodies representing the cable, ordered from anchor to tip.
    """

    extension_dofs: list[DegreeOfFreedom]
    """Longitudinal extension degrees of freedom carrying the cable's stretch state."""

    rest_length: float
    """
    Unstretched length of the cable in meters.
    """

    _world: World
    """World this cable guide belongs to."""

    # %% construction

    @classmethod
    def route_around_wrist(
        cls,
        world: World,
        proximal_body: KinematicStructureEntity,
        distal_body: KinematicStructureEntity,
        proximal_body_T_anchor: HomogeneousTransformationMatrix,
        distal_body_T_guide: HomogeneousTransformationMatrix,
        slack_length: float = 0.05,
        segment_count: int = 1,
        name: PrefixedName | None = None,
    ) -> WristCableGuide:
        """
        Route a soft cable from a proximal anchor past a distal wrist guide.

        The rest length is the taut anchor-to-guide span at the world's current
        configuration plus ``slack_length``, so the cable starts slack.

        :param world: World the cable is added to.
        :param proximal_body: Link carrying the cable anchor, proximal to the wrist.
        :param distal_body: Link carrying the wrist guide, distal to the wrist.
        :param proximal_body_T_anchor: Anchor placement relative to ``proximal_body``.
        :param distal_body_T_guide: Guide placement relative to ``distal_body``.
        :param slack_length: Cable length beyond the taut span, in meters.
        :param segment_count: Number of Cosserat segments representing the cable.
        :param name: Identifier of the cable guide, derived from ``proximal_body`` if
            omitted.
        :raises InvalidCableRouteError: If ``slack_length`` is negative or
            ``segment_count`` is below one.
        :return: The constructed cable guide.
        """
        if slack_length < 0 or segment_count < 1:
            raise InvalidCableRouteError(
                slack_length=slack_length, segment_count=segment_count
            )

        name = name or PrefixedName("wrist_cable_guide", str(proximal_body.name))
        cable_anchor = Body(name=PrefixedName("anchor", str(name)))
        wrist_guide = Body(name=PrefixedName("guide", str(name)))
        # Copy the placements into the parent frames so a caller may reuse the same
        # transform objects across several routes without their reference frame leaking.
        anchor_placement = HomogeneousTransformationMatrix(
            proximal_body_T_anchor, reference_frame=proximal_body
        )
        guide_placement = HomogeneousTransformationMatrix(
            distal_body_T_guide, reference_frame=distal_body
        )
        with world.modify_world():
            world.add_body(cable_anchor)
            world.add_connection(
                FixedConnection(
                    parent=proximal_body,
                    child=cable_anchor,
                    parent_T_connection_expression=anchor_placement,
                )
            )
            world.add_body(wrist_guide)
            world.add_connection(
                FixedConnection(
                    parent=distal_body,
                    child=wrist_guide,
                    parent_T_connection_expression=guide_placement,
                )
            )

        taut_span = cls._span_between(world, cable_anchor, wrist_guide)
        rest_length = taut_span + slack_length

        cable_segments: list[Body] = []
        extension_dofs: list[DegreeOfFreedom] = []
        segment_length = rest_length / segment_count
        with world.modify_world():
            previous_body = cable_anchor
            for segment_index in range(segment_count):
                extension_dof = cls._add_cable_segment(
                    world=world,
                    name=name,
                    segment_index=segment_index,
                    segment_length=segment_length,
                    parent_body=previous_body,
                    cable_segments=cable_segments,
                )
                extension_dofs.append(extension_dof)
                previous_body = cable_segments[-1]

        return cls(
            name=name,
            cable_anchor=cable_anchor,
            wrist_guide=wrist_guide,
            cable_segments=cable_segments,
            extension_dofs=extension_dofs,
            rest_length=rest_length,
            _world=world,
        )

    @staticmethod
    def _add_cable_segment(
        world: World,
        name: PrefixedName,
        segment_index: int,
        segment_length: float,
        parent_body: Body,
        cable_segments: list[Body],
    ) -> DegreeOfFreedom:
        """
        Add one Cosserat segment of the cable and return its extension degree of
        freedom.

        :param world: World the segment is added to.
        :param name: Identifier of the owning cable guide.
        :param segment_index: Position of the segment along the cable, from the anchor.
        :param segment_length: Rest length of the segment in meters.
        :param parent_body: Body the segment is attached to.
        :param cable_segments: Segment body list the new body is appended to.
        :return: The extension degree of freedom carrying the segment's stretch.
        """
        strain_limits = DegreeOfFreedomLimits(
            lower=DerivativeMap(position=-10.0, velocity=-10.0),
            upper=DerivativeMap(position=10.0, velocity=10.0),
        )
        extension_limits = DegreeOfFreedomLimits(
            lower=DerivativeMap(position=0.1, velocity=-10.0),
            upper=DerivativeMap(position=5.0, velocity=10.0),
        )
        bending_x = DegreeOfFreedom(
            name=PrefixedName(f"bending_x_{segment_index}", str(name)),
            limits=strain_limits,
        )
        bending_y = DegreeOfFreedom(
            name=PrefixedName(f"bending_y_{segment_index}", str(name)),
            limits=strain_limits,
        )
        torsion = DegreeOfFreedom(
            name=PrefixedName(f"torsion_{segment_index}", str(name)),
            limits=strain_limits,
        )
        extension = DegreeOfFreedom(
            name=PrefixedName(f"extension_{segment_index}", str(name)),
            limits=extension_limits,
        )
        for dof in [bending_x, bending_y, torsion, extension]:
            world.add_degree_of_freedom(dof)
        world.state[extension.id].position = 1.0

        segment_body = Body(name=PrefixedName(f"segment_{segment_index}", str(name)))
        world.add_body(segment_body)
        world.add_connection(
            CosseratRodConnection(
                parent=parent_body,
                child=segment_body,
                bending_x_dof_id=bending_x.id,
                bending_y_dof_id=bending_y.id,
                torsion_dof_id=torsion.id,
                extension_dof_id=extension.id,
                segment_length=segment_length,
            )
        )
        cable_segments.append(segment_body)
        return extension

    # %% stretch state

    @staticmethod
    def _span_between(world: World, start: Body, end: Body) -> float:
        """
        Euclidean distance between two bodies at the world's current configuration.

        :param world: World the bodies belong to.
        :param start: Body the span starts at.
        :param end: Body the span ends at.
        :return: The distance in meters.
        """
        start_T_end = world.compute_forward_kinematics_np(start, end)
        return float(np.linalg.norm(start_T_end[:3, 3]))

    def current_span(self) -> float:
        """
        Distance the cable currently spans from its anchor to the wrist guide.

        :return: The span in meters.
        """
        return self._span_between(self._world, self.cable_anchor, self.wrist_guide)

    def current_stretch(self) -> float:
        """
        Length the cable is currently stretched beyond its rest length.

        A slack cable reports zero stretch.

        :return: The stretch in meters, never negative.
        """
        return max(0.0, self.current_span() - self.rest_length)

    def synchronize_soft_model_with_geometry(self) -> None:
        """
        Set the cable's extension so the soft model's length matches the routed span.

        This makes the Cosserat rod reflect the current stretch, keeping the soft
        mechanism's state consistent with the geometry of the routing bodies.
        """
        extension_ratio = self.current_span() / self.rest_length
        for extension_dof in self.extension_dofs:
            self._world.state[extension_dof.id].position = extension_ratio
        self._world.notify_state_change()
