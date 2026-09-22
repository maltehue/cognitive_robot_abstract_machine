"""
Live teleoperation of the running robot's arm.

The live world only advances on a motion tick, and a motion tick only fires while a
scripted plan runs (see :mod:`cramera.live.visualization`). Teleoperation has to move
the arm while the scene is otherwise idle, so it brings its own driver: a daemon thread
that, at a fixed rate, servos each commanded arm's end effector toward the latest target
and writes the result into the world.

The control is a resolved-rate servo, not a one-shot solve. Each tick clamps the target
to a small step from where the end effector is now, runs one inverse-kinematics solve to
that near, reachable step, writes the joint positions, and notifies the world. Streaming
targets at ~30 Hz then traces the hand's path smoothly, and a target that is out of
reach or momentarily unsolvable just skips that tick rather than throwing the run.

A client sends normalised coordinates in ``[-1, 1]``; the workspace box each arm maps
them onto lives here, so the browser never has to know the robot's frames.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing_extensions import Any, Dict, List, Optional, Tuple

import numpy as np

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_computations.ik_solver import IKSolverException
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from cramera.logging_setup import get_logger

logger = get_logger(__name__)

RATE_HZ = 30.0
"""
How often the driver servos the arm toward its target.
"""

STOP_TIMEOUT_SECONDS = 2.0
"""
Maximum time a pending servo tick may take to release its robot.
"""

MAX_STEP_METRES = 0.03
"""
How far the end effector is allowed to move per tick.

Small enough that each stepped target sits next to the current, reachable pose, so the
solve converges every tick; at the rate above this still allows ~0.9 m/s of hand travel.
"""

IK_MAX_ITERATIONS = 60
"""
Iterations the per-tick solve is given -- a near step converges in far fewer.
"""

# Where each arm's normalised [-1, 1] cube lands, as (centre, half-extent) in the arm's
# root frame (the torso for a PR2), in metres. Sized to the comfortable reach measured on
# the robot, so the corners stay solvable.
WORKSPACE: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "left": ((0.65, 0.20, 0.00), (0.15, 0.20, 0.20)),
    "right": ((0.65, -0.20, 0.00), (0.15, 0.20, 0.20)),
}


class MalformedTeleopRequest(Exception):
    """
    Raised when a teleop payload cannot be read.
    """


class TeleopUnavailable(Exception):
    """
    Raised when teleop is requested before a live robot is bound.
    """


@dataclass(frozen=True)
class TeleopRequest:
    """
    One hand target: which arm, and where in its normalised workspace it points.
    """

    arm: str
    """
    ``"left"`` or ``"right"``.
    """

    position: List[float]
    """
    Target ``[x, y, z]`` in ``[-1, 1]``, mapped onto the arm's workspace box.
    """

    gripper: Optional[float] = None
    """
    How far the gripper is opened, ``0`` shut to ``1`` wide, or None to leave it be.
    """

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> TeleopRequest:
        """
        Build a request from a decoded ``POST /teleop`` body.

        :param payload: The decoded JSON body.
        :raises MalformedTeleopRequest: If the arm or position is unusable.
        """
        arm = payload.get("arm", "left")
        if arm not in WORKSPACE:
            raise MalformedTeleopRequest("'arm' must be one of %s" % list(WORKSPACE))
        value = payload.get("position")
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise MalformedTeleopRequest("'position' must be a list of 3 numbers")
        try:
            position = [max(-1.0, min(1.0, float(v))) for v in value]
        except (TypeError, ValueError):
            raise MalformedTeleopRequest("'position' must be three numbers")
        gripper = payload.get("gripper")
        if gripper is not None:
            try:
                gripper = max(0.0, min(1.0, float(gripper)))
            except (TypeError, ValueError):
                raise MalformedTeleopRequest("'gripper' must be a number")
        return cls(arm=arm, position=position, gripper=gripper)


@dataclass
class _ArmChain:
    """
    The cached handles a servo tick needs for one arm.
    """

    root: Any
    tip: Any
    centre: np.ndarray
    half_extent: np.ndarray
    orientation: List[float]
    """
    The end effector's start orientation ``[qx, qy, qz, qw]``, held fixed while
    teleoperating so a client only has to command a position.
    """

    gripper_connections: List[Any] = None
    gripper_closed: List[float] = None
    gripper_open: List[float] = None
    """
    The gripper's finger connections and their shut/wide target values, so a commanded
    opening amount interpolates between them.

    Empty when the arm has no known gripper.
    """


class TeleopController:
    """
    Drives the live robot's arm(s) from streamed hand targets on its own thread.

    :param world: The world the live demo is executing in.
    :param robot: The robot annotation in that world.
    :param is_busy: Returns True when a scripted plan is currently moving the robot, so
        the teleop driver can stand aside rather than fight it for the world.
    """

    def __init__(
        self, world: World, robot: AbstractRobot, is_busy=lambda: False
    ) -> None:
        self._world = world
        self._robot = robot
        self._is_busy = is_busy
        self._lock = threading.Lock()
        self._targets: Dict[str, TeleopRequest] = {}
        self._chains: Dict[str, Optional[_ArmChain]] = {}
        self._active = False
        self._thread: Optional[threading.Thread] = None

    def submit(self, request: TeleopRequest) -> None:
        """
        Take the newest target for an arm and make sure the driver is running.

        :param request: The hand target to servo toward.
        """
        with self._lock:
            self._targets[request.arm] = request
            self._active = True
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="teleop-driver", daemon=True
                )
                self._thread.start()

    def stop(self) -> None:
        """
        Finish the current servo tick and release the robot.

        :raises TeleopUnavailable: If a pending tick cannot stop within its time budget.
        """
        with self._lock:
            self._active = False
            self._targets.clear()
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=STOP_TIMEOUT_SECONDS)
            if worker.is_alive():
                raise TeleopUnavailable("The previous hand control has not stopped yet")

    def _run(self) -> None:
        period = 1.0 / RATE_HZ
        while True:
            start = time.monotonic()
            with self._lock:
                if not self._active:
                    return
                targets = dict(self._targets)
            if not self._is_busy():
                try:
                    self._servo(targets)
                except Exception:  # a live driver must never die on one bad tick
                    logger.exception("teleop tick failed")
            time.sleep(max(0.0, period - (time.monotonic() - start)))

    def _servo(self, targets: Dict[str, TeleopRequest]) -> None:
        """
        Step every commanded arm one bounded move toward its target, then notify once.

        :param targets: The latest target per arm.
        """
        wrote = False
        for name, request in targets.items():
            chain = self._chain(name)
            if chain is None:
                continue
            if request.gripper is not None and chain.gripper_connections:
                for i, connection in enumerate(chain.gripper_connections):
                    connection.position = chain.gripper_closed[i] + request.gripper * (
                        chain.gripper_open[i] - chain.gripper_closed[i]
                    )
                wrote = True
            goal = chain.centre + np.asarray(request.position) * chain.half_extent
            current = self._tip_in_root(chain)[:3]
            step = goal - current
            distance = float(np.linalg.norm(step))
            if distance > MAX_STEP_METRES:
                step = step / distance * MAX_STEP_METRES
            stepped = current + step
            target = HomogeneousTransformationMatrix.from_xyz_quaternion(
                stepped[0],
                stepped[1],
                stepped[2],
                *chain.orientation,
                reference_frame=chain.root,
            )
            try:
                dofs = self._world.compute_inverse_kinematics(
                    chain.root, chain.tip, target, max_iterations=IK_MAX_ITERATIONS
                )
            except IKSolverException:
                continue
            for dof, value in dofs.items():
                self._world.state[dof.id].position = value
            wrote = True
        if wrote:
            self._world.notify_state_change(publish_changes=True)

    def _tip_in_root(self, chain: _ArmChain) -> List[float]:
        """
        The end effector's pose in its root frame as ``[x, y, z, qx, qy, qz, qw]``.
        """
        root_T_world = chain.root.global_pose.to_homogeneous_matrix().inverse()
        world_T_tip = chain.tip.global_pose.to_homogeneous_matrix()
        return (root_T_world @ world_T_tip).to_position_quaternion_list()

    def _chain(self, name: str) -> Optional[_ArmChain]:
        """
        The cached chain for an arm, resolved and measured on first use.

        :param name:``"left"`` or ``"right"``.
        """
        if name in self._chains:
            return self._chains[name]
        arm = (
            self._robot.get_left_arm_if_specified()
            if name == "left"
            else self._robot.get_right_arm_if_specified()
        )
        chain: Optional[_ArmChain] = None
        if arm is not None:
            centre, half = WORKSPACE[name]
            root, tip = arm.root, arm.end_effector.tool_frame
            chain = _ArmChain(
                root=root,
                tip=tip,
                centre=np.asarray(centre),
                half_extent=np.asarray(half),
                orientation=[0, 0, 0, 1],
                gripper_connections=[],
                gripper_closed=[],
                gripper_open=[],
            )
            chain.orientation = self._tip_in_root(chain)[3:]
            self._resolve_gripper(arm, chain)
        else:
            logger.warning("teleop: robot has no %s arm", name)
        self._chains[name] = chain
        return chain

    def _resolve_gripper(self, arm, chain: _ArmChain) -> None:
        """
        Fill ``chain``'s gripper handles from the end effector's open/shut joint states,
        so a commanded opening interpolates between them. Leaves them empty (gripper
        control disabled) if the end effector defines no such states.

        :param arm: The arm whose gripper is being resolved.
        :param chain: The chain to fill in place.
        """
        try:
            end_effector = arm.end_effector
            wide = end_effector.get_joint_state_by_type(GripperState.OPEN)
            shut = end_effector.get_joint_state_by_type(GripperState.CLOSE)
            shut_by_connection = {
                id(connection): value
                for connection, value in zip(shut.connections, shut.target_values)
            }
            chain.gripper_connections = list(wide.connections)
            chain.gripper_open = list(wide.target_values)
            chain.gripper_closed = [
                shut_by_connection.get(id(connection), 0.0)
                for connection in wide.connections
            ]
        except Exception:
            logger.info("teleop: no gripper joint states for the %s arm", chain.tip)
