"""
Temporal state management and semantic querying for the
Semantic Digital Twin (SDT).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional, Tuple

from .object_state import ObjectState, CutState, FillState

# ============================================================================
# Exceptions
# ============================================================================


class StateUnknownError(Exception):
    pass


class LowConfidenceError(Exception):
    pass


class StaleStateError(Exception):
    pass


# ============================================================================
# Dynamic State Manager
# ============================================================================


class DynamicStateManager:
    """
    Memory-bounded temporal SDT state manager.
    """

    def __init__(
        self,
        max_history_per_object: int = 200,
        max_history_seconds: Optional[float] = 300.0,
    ):
        self.max_history_size = max_history_per_object
        self.max_history_seconds = max_history_seconds

        self._state_history: Dict[str, Deque[ObjectState]] = defaultdict(
            lambda: deque(maxlen=self.max_history_size)
        )

    # ------------------------------------------------------------------------
    # Update State
    # ------------------------------------------------------------------------

    def update_state(self, entity_id: str, new_state: ObjectState) -> None:
        if new_state.timestamp is None:
            new_state.timestamp = time.time()

        history = self._state_history[entity_id]

        if history:
            last = history[-1]
            if self._is_duplicate(last, new_state):
                return  # Skip redundant update

        history.append(new_state)
        self._cleanup_old(entity_id)

    # ------------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------------

    def get_current_state(self, entity_id: str) -> ObjectState:
        history = self._state_history.get(entity_id)
        if not history:
            raise StateUnknownError(f"No state history found for '{entity_id}'")
        return history[-1]

    def get_state_history(self, entity_id: str) -> List[ObjectState]:
        return list(self._state_history.get(entity_id, []))

    def get_state_at(self, entity_id: str, timestamp: float) -> Optional[ObjectState]:
        history = self._state_history.get(entity_id, [])
        for state in reversed(history):
            if state.timestamp <= timestamp:
                return state
        return None

    # ------------------------------------------------------------------------
    # Query Interface (Robot API)
    # ------------------------------------------------------------------------

    def is_cut(
        self,
        entity_id: str,
        min_confidence: float = 0.8,
        max_age_sec: Optional[float] = None,
    ) -> bool:
        state = self.get_current_state(entity_id)
        self._validate(state, min_confidence, max_age_sec)

        if state.cut_state is None:
            raise StateUnknownError(f"Cut state for '{entity_id}' is unknown.")

        return state.cut_state == CutState.CUT

    def is_filled(
        self,
        entity_id: str,
        min_confidence: float = 0.8,
        max_age_sec: Optional[float] = None,
    ) -> bool:
        state = self.get_current_state(entity_id)
        self._validate(state, min_confidence, max_age_sec)

        if state.fill_state is None:
            raise StateUnknownError(f"Fill state for '{entity_id}' is unknown.")

        return state.fill_state in (FillState.FILLED, FillState.FULL)

    def is_empty(
        self,
        entity_id: str,
        min_confidence: float = 0.8,
        max_age_sec: Optional[float] = None,
    ) -> bool:
        state = self.get_current_state(entity_id)
        self._validate(state, min_confidence, max_age_sec)

        if state.fill_state is None:
            raise StateUnknownError(f"Fill state for '{entity_id}' is unknown.")

        return state.fill_state == FillState.EMPTY

    # ------------------------------------------------------------------------
    # Temporal Reasoning & Transitions
    # ------------------------------------------------------------------------

    def was_cut_at(
        self, entity_id: str, timestamp: float, min_confidence: float = 0.8
    ) -> bool:
        state = self.get_state_at(entity_id, timestamp)
        if not state:
            raise StateUnknownError(
                f"No historical state available for '{entity_id}' at time {timestamp}"
            )

        self._validate(state, min_confidence, max_age_sec=None)

        if state.cut_state is None:
            raise StateUnknownError(
                f"Cut state for '{entity_id}' at time {timestamp} is unknown."
            )

        return state.cut_state == CutState.CUT

    def get_transitions(self, entity_id: str) -> List[Tuple[ObjectState, ObjectState]]:
        history = self._state_history.get(entity_id, [])
        transitions = []

        for i in range(1, len(history)):
            prev_state = history[i - 1]
            curr_state = history[i]

            if (
                prev_state.cut_state != curr_state.cut_state
                or prev_state.fill_state != curr_state.fill_state
            ):
                transitions.append((prev_state, curr_state))

        return transitions

    # ------------------------------------------------------------------------
    # Internal Validation & Memory Safety Utilities
    # ------------------------------------------------------------------------

    def _validate(
        self, state: ObjectState, min_confidence: float, max_age_sec: Optional[float]
    ) -> None:
        if state.confidence < min_confidence:
            raise LowConfidenceError(
                f"State confidence ({state.confidence}) is below required threshold ({min_confidence})"
            )

        if max_age_sec is not None and state.timestamp:
            age = time.time() - state.timestamp
            if age > max_age_sec:
                raise StaleStateError(
                    f"State observation is {age:.2f}s old (exceeds {max_age_sec:.2f}s limit)"
                )

    def _cleanup_old(self, entity_id: str) -> None:
        if self.max_history_seconds is None:
            return

        now = time.time()
        history = self._state_history[entity_id]

        while history:
            oldest_state = history[0]
            if (
                oldest_state.timestamp
                and (now - oldest_state.timestamp) > self.max_history_seconds
            ):
                history.popleft()
            else:
                break

    def _is_duplicate(self, a: ObjectState, b: ObjectState) -> bool:
        return (
            a.cut_state == b.cut_state
            and a.fill_state == b.fill_state
            and a.source == b.source
            and abs((a.confidence or 0) - (b.confidence or 0)) < 1e-3
        )
