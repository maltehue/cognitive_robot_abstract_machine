"""
Abstract BMP predicate definitions.

The three predicates of the Law of Task-Achieving Body Motion:

  SatisfiesRequest(Π, G_final) — semantic correctness
  Causes(τ, G_final, Φ, I_Φ)  — causal sufficiency
  CanPerform(R, τ)              — embodiment feasibility

Together they form the axiom:
  ∀R, E, Π, G, τ:
    SatisfiesRequest(Π, G_final) ∧
    Causes(τ, G_final, Φ, I_Φ) ∧
    CanPerform(R, τ)
    ⟹ CanAchieve(R, E, Π, τ)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from krrood.entity_query_language.predicate import Predicate

from semantic_digital_twin.reasoning.body_motion_problem.types import (
    Effect,
    Motion,
    TaskRequest,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World


@dataclass
class Causes(Predicate):
    """
    Causal sufficiency predicate: Causes(τ, G_final, Φ, I_Φ).

    Checks whether trajectory τ is a physically valid explanation for
    transitioning from the current SDT state to G_final under the scoped
    physics model Φ.

    Supports three usage modes:
      Case 1 — motion unknown: generate τ from motion_model, verify against effect.
      Case 2 — effect unknown: execute τ and check which effects become true.
      Case 3 — both unknown: union of Case 1 and Case 2.
    """

    effect: Effect

    environment: World

    motion: Optional[Motion]

    def __call__(self, *args, **kwargs):
        if self.effect.is_achieved():
            return False

        # Case 1: no trajectory yet — generate one via the physics model
        if (
            self.motion
            and self.motion.motion_model
            and len(self.motion.trajectory) == 0
        ):
            trajectory, _ = self.motion.motion_model.run(self.effect, self.environment)
            if trajectory and len(trajectory) > 0:
                self.motion.trajectory = trajectory

        # Verify the trajectory causes the effect by replaying it on the world state
        return self._map_motion_to_effect()

    def _map_motion_to_effect(self):
        initial_state_data = self.environment.state._data.copy()
        trajectory = self.motion.trajectory
        actuator = self.motion.actuator

        is_achieved_pre = self.effect.is_achieved()

        for position in trajectory:
            self.environment.set_positions_1DOF_connection({actuator: float(position)})

        is_achieved_post = self.effect.is_achieved()

        self.environment.state._data[:] = initial_state_data
        self.environment.notify_state_change()

        return (not is_achieved_pre) and is_achieved_post


@dataclass
class SatisfiesRequest(Predicate):
    """
    Semantic correctness predicate: SatisfiesRequest(Π, G_final).

    Checks that a final SDT state (represented by an Effect) matches the
    intent of a task specification (TaskRequest).

    Subclass this predicate to implement domain-specific semantic correctness
    checks for a given TEE class.
    """

    task: TaskRequest
    effect: Effect

    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__.")


@dataclass
class CanPerform(Predicate):
    """
    Embodiment feasibility predicate: CanPerform(R, τ).

    Checks whether trajectory τ is executable by robot R with respect to
    kinematic and dynamic limits and absence of self-collision:
      CanPerform(R, τ) ⟺ ∀t∈τ: (q_t, q̇_t, q̈_t) ∈ K_R ∧ ¬SelfCollision(q_t)

    Subclass this predicate to implement embodiment feasibility checks for
    a specific robot morphology or TEE class.
    """

    motion: Motion
    robot: AbstractRobot

    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__.")
