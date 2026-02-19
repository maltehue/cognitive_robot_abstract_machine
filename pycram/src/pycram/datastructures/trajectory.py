from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterable, Tuple

from .pose import PoseStamped


@dataclass(frozen=True)
class PoseTrajectory:
    """
    Immutable wrapper for a sequence of waypoint poses.
    """

    poses: Tuple[PoseStamped, ...]
    """
    Ordered TCP waypoints.
    """

    def __init__(self, poses: Iterable[PoseStamped]):
        poses_tuple = tuple(poses)
        if not poses_tuple:
            raise ValueError("Provide at least one PoseStamped.")
        if not all(isinstance(pose, PoseStamped) for pose in poses_tuple):
            raise ValueError("All poses  must be of type PoseStamped.")
        object.__setattr__(self, "poses", poses_tuple)
