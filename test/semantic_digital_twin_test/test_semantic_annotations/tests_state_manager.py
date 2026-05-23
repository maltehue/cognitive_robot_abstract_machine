import pytest
import time

from semantic_digital_twin.semantic_annotations.object_state import ObjectState, CutState, FillState
from semantic_digital_twin.semantic_annotations.state_manager import (
    DynamicStateManager,
    StateUnknownError,
    LowConfidenceError,
    StaleStateError
)


class DummyTarget:
    pass


def make_state(cut_state=None, fill_state=None, confidence=1.0, timestamp=None, source="Test"):
    return ObjectState(
        target=DummyTarget(),
        cut_state=cut_state,
        fill_state=fill_state,
        confidence=confidence,
        timestamp=timestamp,
        source=source
    )


@pytest.fixture
def manager():
    return DynamicStateManager(max_history_per_object=3, max_history_seconds=10.0)


def test_basic_update_and_retrieve(manager):
    state = make_state(cut_state=CutState.UNCUT)
    manager.update_state("onion_1", state)

    current = manager.get_current_state("onion_1")
    assert current.cut_state == CutState.UNCUT
    assert current.timestamp is not None


def test_duplicate_filtering(manager):
    manager.update_state("onion_1", make_state(cut_state=CutState.CUT))
    manager.update_state("onion_1", make_state(cut_state=CutState.CUT))

    history = manager.get_state_history("onion_1")
    assert len(history) == 1

    manager.update_state("onion_1", make_state(cut_state=CutState.UNCUT))
    history = manager.get_state_history("onion_1")
    assert len(history) == 2


def test_ring_buffer_pruning(manager):
    for i in range(5):
        manager.update_state("cup_1", make_state(fill_state=FillState.EMPTY, confidence=0.1 * i))

    history = manager.get_state_history("cup_1")
    assert len(history) == 3
    assert history[0].confidence == 0.2


def test_time_based_pruning():
    """Ensure _cleanup_old drops states that age out of the buffer."""
    # Create a local manager with a tiny 0.2-second memory limit
    fast_manager = DynamicStateManager(max_history_per_object=3, max_history_seconds=0.2)

    # 1. Insert a fresh state
    fast_manager.update_state("cup_1", make_state(fill_state=FillState.EMPTY))
    assert len(fast_manager.get_state_history("cup_1")) == 1

    # 2. Wait for 0.25 seconds so the first state becomes older than max_history_seconds
    time.sleep(0.25)

    # 3. Insert a new state. This triggers _cleanup_old, which will see the first state is expired.
    fast_manager.update_state("cup_1", make_state(fill_state=FillState.FILLED))

    history = fast_manager.get_state_history("cup_1")

    # The length should be 1 because the EMPTY state was pruned out
    assert len(history) == 1
    assert history[0].fill_state == FillState.FILLED

def test_robot_api_safety_confidence(manager):
    state = make_state(cut_state=CutState.CUT, confidence=0.6)
    manager.update_state("onion_1", state)

    assert manager.is_cut("onion_1", min_confidence=0.5) is True

    with pytest.raises(LowConfidenceError):
        manager.is_cut("onion_1", min_confidence=0.8)


def test_robot_api_safety_staleness(manager):
    now = time.time()
    state = make_state(fill_state=FillState.FULL, timestamp=now - 6.0)
    manager.update_state("cup_1", state)

    assert manager.is_filled("cup_1", max_age_sec=10.0) is True

    with pytest.raises(StaleStateError):
        manager.is_filled("cup_1", max_age_sec=5.0)


def test_unknown_state_queries(manager):
    with pytest.raises(StateUnknownError):
        manager.is_cut("ghost_object")

    manager.update_state("onion_1", make_state(fill_state=FillState.EMPTY))

    with pytest.raises(StateUnknownError, match="is unknown"):
        manager.is_cut("onion_1")

def test_temporal_reasoning_get_state_at(manager):
    """Ensure get_state_at correctly fetches the state valid at a given time."""
    now = time.time()

    # We must keep these timestamps within the 10.0 second max_history_seconds limit!
    manager.update_state("onion_1", make_state(cut_state=CutState.UNCUT, timestamp=now - 6.0))
    manager.update_state("onion_1", make_state(cut_state=CutState.CUT, timestamp=now - 2.0))

    # Query exact time of the newer state (2 seconds ago)
    assert manager.get_state_at("onion_1", now - 2.0).cut_state == CutState.CUT

    # Query in between (4 seconds ago - Should return the UNCUT state)
    assert manager.get_state_at("onion_1", now - 4.0).cut_state == CutState.UNCUT

    # Query before history begins (8 seconds ago)
    assert manager.get_state_at("onion_1", now - 8.0) is None

def test_transitions(manager):
    now = time.time()
    manager.update_state("cup_1", make_state(fill_state=FillState.EMPTY, timestamp=now - 3.0))
    manager.update_state("cup_1", make_state(fill_state=FillState.FILLED, timestamp=now - 2.0))
    manager.update_state("cup_1", make_state(fill_state=FillState.FULL, timestamp=now - 1.0))

    transitions = manager.get_transitions("cup_1")
    assert len(transitions) == 2
    assert transitions[0][0].fill_state == FillState.EMPTY
    assert transitions[0][1].fill_state == FillState.FILLED
    assert transitions[1][0].fill_state == FillState.FILLED
    assert transitions[1][1].fill_state == FillState.FULL