"""
Unit tests for DynamicStateManager and Entity Query Language (EQL) Predicates.

Validates:
- Agnostic temporal ring-buffer storage and type-safe retrieval tracks
- Inline deduplication logic and sliding-window timestamp extensions per state classification
- Memory-bounded chronological pruning and aging drop-offs
- Predicate logic assertion accuracy and validation safety guards
"""

import time
import pytest
from unittest.mock import MagicMock

from semantic_digital_twin.world_description.world_entity import WorldEntity
from semantic_digital_twin.semantic_annotations.object_state import (
    CutCondition,
    CutState,
    FillLevel,
    FillState,
    IsCut,
    IsEmpty,
    IsFilled,
)
from semantic_digital_twin.semantic_annotations.state_manager import (
    DynamicStateManager,
    LowConfidenceError,
    StaleStateError,
    StateUnknownError,
)


@pytest.fixture
def dummy_target():
    """Provides a mocked WorldEntity required by ObjectState instances."""
    return MagicMock(spec=WorldEntity)


@pytest.fixture
def manager():
    """Provides a fresh state manager matching historical limits."""
    return DynamicStateManager(max_history_per_object=3, max_history_seconds=10.0)


# ------------------------------------------------------------------------
# Structural Ring-Buffer and Insertion Tests
# ------------------------------------------------------------------------


def test_basic_update_and_retrieve(manager, dummy_target):
    state = CutState(target=dummy_target, state=CutCondition.UNCUT)
    manager.update_state("onion_1", state)

    current = manager.get_current_state_by_type("onion_1", CutState)
    assert current.state == CutCondition.UNCUT
    assert current.timestamp is not None


def test_duplicate_filtering_per_type(manager, dummy_target):
    """
    Ensure consecutive duplicate observations don't bloat history,
    but do update metadata inline to extend safety-guard lifecycles.
    """
    initial_timestamp = time.time() - 5.0
    state_1 = CutState(
        target=dummy_target,
        state=CutCondition.CUT,
        confidence=0.9,
        timestamp=initial_timestamp,
    )
    manager.update_state("onion_1", state_1)

    # Interleaving a FillState update shouldn't break duplicate checks for CutState
    manager.update_state(
        "onion_1", FillState(target=dummy_target, state=FillLevel.EMPTY)
    )

    # Create an identical CutState observation with a fresher timestamp and tweaked confidence
    fresh_timestamp = time.time()
    state_2 = CutState(
        target=dummy_target,
        state=CutCondition.CUT,
        confidence=0.95,
        timestamp=fresh_timestamp,
    )
    manager.update_state("onion_1", state_2)

    history = manager.get_state_history("onion_1")

    # Structural count must be 2 (1 CutState, 1 FillState) because state_2 updates state_1 inline
    assert len(history) == 2

    # Check that the inline sliding-window update kept the trace fresh
    current_cut = manager.get_current_state_by_type("onion_1", CutState)
    assert current_cut.timestamp == fresh_timestamp
    assert current_cut.confidence == pytest.approx(0.95)


def test_ring_buffer_pruning(manager, dummy_target):
    """Verify maxlen drops oldest elements when max_history_per_object is reached."""
    # Using incrementally increasing state enums/values to prevent inline deduplication overwrites
    manager.update_state(
        "cup_1", FillState(target=dummy_target, state=FillLevel.EMPTY, confidence=0.1)
    )
    manager.update_state(
        "cup_1", FillState(target=dummy_target, state=FillLevel.FILLED, confidence=0.2)
    )
    manager.update_state(
        "cup_1", FillState(target=dummy_target, state=FillLevel.FULL, confidence=0.3)
    )
    manager.update_state(
        "cup_1", FillState(target=dummy_target, state=FillLevel.EMPTY, confidence=0.4)
    )
    manager.update_state(
        "cup_1", FillState(target=dummy_target, state=FillLevel.FILLED, confidence=0.5)
    )

    history = manager.get_state_history("cup_1")
    assert len(history) == 3
    # The first two elements should be pushed out by the ring-buffer maxlen sequence limits
    assert history[0].confidence == pytest.approx(0.3)
    assert history[2].confidence == pytest.approx(0.5)


def test_time_based_pruning(dummy_target):
    """Ensure _cleanup_old drops states that age out of the buffer."""
    fast_manager = DynamicStateManager(
        max_history_per_object=3, max_history_seconds=0.2
    )

    fast_manager.update_state(
        "cup_1", FillState(target=dummy_target, state=FillLevel.EMPTY)
    )
    assert len(fast_manager.get_state_history("cup_1")) == 1

    time.sleep(0.25)

    # Next disparate update triggers cleanup loop sequence
    fast_manager.update_state(
        "cup_1", FillState(target=dummy_target, state=FillLevel.FILLED)
    )

    history = fast_manager.get_state_history("cup_1")
    assert len(history) == 1
    assert history[0].state == FillLevel.FILLED


# ------------------------------------------------------------------------
# EQL Predicate Validation Guard Tests
# ------------------------------------------------------------------------


def test_predicate_safety_confidence(manager, dummy_target):
    state = CutState(target=dummy_target, state=CutCondition.CUT, confidence=0.6)
    manager.update_state("onion_1", state)

    predicate_pass = IsCut(manager, "onion_1", min_confidence=0.5)
    assert predicate_pass() is True

    predicate_fail = IsCut(manager, "onion_1", min_confidence=0.8)
    with pytest.raises(LowConfidenceError):
        predicate_fail()


def test_predicate_safety_staleness(manager, dummy_target):
    now = time.time()
    state = FillState(target=dummy_target, state=FillLevel.FULL, timestamp=now - 6.0)
    manager.update_state("cup_1", state)

    predicate_pass = IsFilled(manager, "cup_1", max_age_sec=10.0)
    assert predicate_pass() is True

    predicate_fail = IsFilled(manager, "cup_1", max_age_sec=5.0)
    with pytest.raises(StaleStateError):
        predicate_fail()


def test_unknown_state_queries(manager):
    """Verify queries for objects or states without history throw StateUnknownError."""
    with pytest.raises(StateUnknownError):
        IsCut(manager, "ghost_object")()

    # onion_1 only has a FillState recorded, looking for CutState must throw
    manager.update_state(
        "onion_1", FillState(target=MagicMock(spec=WorldEntity), state=FillLevel.EMPTY)
    )

    with pytest.raises(StateUnknownError, match="No state segment matching type"):
        IsCut(manager, "onion_1")()


# ------------------------------------------------------------------------
# Temporal Querying & Transitions
# ------------------------------------------------------------------------


def test_temporal_reasoning_get_state_at(manager, dummy_target):
    """Ensure get_state_at correctly fetches the state subclass valid at a given time point."""
    now = time.time()

    manager.update_state(
        "onion_1",
        CutState(target=dummy_target, state=CutCondition.UNCUT, timestamp=now - 6.0),
    )
    manager.update_state(
        "onion_1",
        CutState(target=dummy_target, state=CutCondition.CUT, timestamp=now - 2.0),
    )

    # Query exact timestamp of newest state
    assert (
        manager.get_state_at("onion_1", CutState, now - 2.0).state == CutCondition.CUT
    )

    # Query in between updates (should yield UNCUT state)
    assert (
        manager.get_state_at("onion_1", CutState, now - 4.0).state == CutCondition.UNCUT
    )

    # Query before trace timeline history begins
    assert manager.get_state_at("onion_1", CutState, now - 8.0) is None


def test_transitions(manager, dummy_target):
    """Ensure dynamic step changes are tracked sequentially along separate type lanes."""
    now = time.time()
    manager.update_state(
        "cup_1",
        FillState(target=dummy_target, state=FillLevel.EMPTY, timestamp=now - 3.0),
    )
    manager.update_state(
        "cup_1",
        FillState(target=dummy_target, state=FillLevel.FILLED, timestamp=now - 2.0),
    )
    manager.update_state(
        "cup_1",
        FillState(target=dummy_target, state=FillLevel.FULL, timestamp=now - 1.0),
    )

    transitions = manager.get_transitions("cup_1")
    assert len(transitions) == 2
    assert transitions[0][0].state == FillLevel.EMPTY
    assert transitions[0][1].state == FillLevel.FILLED
    assert transitions[1][0].state == FillLevel.FILLED
    assert transitions[1][1].state == FillLevel.FULL
