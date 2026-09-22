"""
Instance-aware robot parts read from existing scene metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing_extensions import Any

from cramera.robot_fields import RobotField
from cramera.robot_parts import RobotPartAnnotation


# %% recorded identities
@dataclass(frozen=True)
class RecordedRobot:
    """
    One recorded instance and the parts belonging to that instance.
    """

    identifier: str
    """
    Native instance identity, or the legacy model name.
    """

    name: str
    """
    Query identity, qualified by instance when a scene holds several robots.
    """

    prefix: str
    """
    Namespace of the robot's native joint and link names.
    """

    parts: dict[str, list[str]]
    """
    Uniquely named parts mapped to their local link names.
    """

    annotations: list[RobotPartAnnotation]
    """
    Native arm and gripper relationships with unambiguous query names.
    """

    @classmethod
    def of_scene(cls, scene: dict[str, Any]) -> list[RecordedRobot]:
        """
        Read all instances, retaining the single-robot bundle format.

        :param scene: Parsed scene metadata.
        :return: Recorded robot descriptions in their saved order.
        """
        entries = scene.get(RobotField.ROBOTS) or [scene.get(RobotField.ROBOT) or {}]
        return [cls.from_payload(entry, len(entries) > 1) for entry in entries]

    @classmethod
    def from_payload(cls, payload: dict[str, Any], qualified: bool) -> RecordedRobot:
        """
        Read one metadata entry and namespace its query identities as needed.

        :param payload: Existing robot model and part annotations.
        :param qualified: Whether several robots require distinct part identities.
        :return: Parsed robot and its parts.
        """
        model_name = payload.get(RobotField.NAME, RobotField.ROBOT)
        identifier = payload.get(RobotField.IDENTIFIER, model_name)
        namespace = identifier + "/" if qualified else ""
        annotations = [
            RobotPartAnnotation.from_payload(entry)
            for entry in payload.get(RobotField.PART_ANNOTATIONS) or []
        ]
        return cls(
            identifier=identifier,
            name=identifier if qualified else model_name,
            prefix=payload.get(RobotField.PREFIX, ""),
            parts={
                namespace + part: links
                for part, links in (payload.get(RobotField.PARTS) or {}).items()
            },
            annotations=[
                replace(
                    annotation,
                    name=namespace + annotation.name,
                    attached_to=(
                        namespace + annotation.attached_to
                        if annotation.attached_to
                        else None
                    ),
                )
                for annotation in annotations
            ],
        )
