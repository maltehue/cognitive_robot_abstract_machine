"""
Unit tests for DynamicStateManager and Entity Query Language (EQL) Predicates.

Validates:
- Agnostic history logging and type-safe retrieval
- Timeline duplicate deduplication per state type
- Out-of-bounds confidence validation and stale-state aging guards
- IsCut, IsFilled, and IsEmpty Predicate boolean evaluation accuracy
"""

import time
import unittest
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


class TestDynamicStateManagement(unittest.TestCase):

    def setUp(self):
        self.manager = DynamicStateManager(
            max_history_per_object=5, max_history_seconds=60.0
        )
        self.entity_id = "bowl_123"
        self.dummy_target = MagicMock(spec=WorldEntity)

    # ------------------------------------------------------------------------
    # State Manager Storage and Interleaving Tests
    # ------------------------------------------------------------------------

    def test_update_and_type_safe_retrieval(self):
        """Verify the manager stores different subclasses independently on the timeline."""
        fill = FillState(target=self.dummy_target, state=FillLevel.FILLED)
        cut = CutState(target=self.dummy_target, state=CutCondition.UNCUT)

        self.manager.update_state(self.entity_id, fill)
        self.manager.update_state(self.entity_id, cut)

        # Confirm we can pinpoint the exact current slice by type
        current_fill = self.manager.get_current_state_by_type(self.entity_id, FillState)
        current_cut = self.manager.get_current_state_by_type(self.entity_id, CutState)

        self.assertEqual(current_fill.state, FillLevel.FILLED)
        self.assertEqual(current_cut.state, CutCondition.UNCUT)

    def test_duplicate_filtering_per_type(self):
        """Ensure back-to-back duplicate evaluations within the same stream type are skipped."""
        state_1 = CutState(
            target=self.dummy_target,
            state=CutCondition.CUT,
            source="VLM",
            confidence=0.9,
        )
        state_2 = CutState(
            target=self.dummy_target,
            state=CutCondition.CUT,
            source="VLM",
            confidence=0.9,
        )

        # Interleaving a different type should not interrupt duplicate checks for CutState
        other_state = FillState(target=self.dummy_target, state=FillLevel.EMPTY)

        self.manager.update_state(self.entity_id, state_1)
        self.manager.update_state(self.entity_id, other_state)
        self.manager.update_state(self.entity_id, state_2)

        # Total history length should be 2 instead of 3 because state_2 is dropped
        history = self.manager.get_state_history(self.entity_id)
        self.assertEqual(len(history), 2)

    def test_missing_state_type_raises_error(self):
        """Confirm that searching for an unrecorded state subclass triggers a StateUnknownError."""
        fill = FillState(target=self.dummy_target, state=FillLevel.EMPTY)
        self.manager.update_state(self.entity_id, fill)

        with self.assertRaises(StateUnknownError):
            self.manager.get_current_state_by_type(self.entity_id, CutState)

    # ------------------------------------------------------------------------
    # Validation Rules and Safety Guard Tests
    # ------------------------------------------------------------------------

    def test_low_confidence_guard(self):
        """Verify that state updates drop below confidence thresholds break safety bounds."""
        low_conf_state = CutState(
            target=self.dummy_target, state=CutCondition.CUT, confidence=0.4
        )
        self.manager.update_state(self.entity_id, low_conf_state)

        predicate = IsCut(self.manager, self.entity_id, min_confidence=0.8)
        with self.assertRaises(LowConfidenceError):
            predicate()

    def test_stale_state_age_guard(self):
        """Verify that stale observations past our max-age boundary raise a StaleStateError."""
        old_timestamp = time.time() - 10.0
        stale_state = FillState(
            target=self.dummy_target, state=FillLevel.FULL, timestamp=old_timestamp
        )
        self.manager.update_state(self.entity_id, stale_state)

        predicate = IsFilled(self.manager, self.entity_id, max_age_sec=5.0)
        with self.assertRaises(StaleStateError):
            predicate()

    # ------------------------------------------------------------------------
    # EQL Predicate Operational Verification
    # ------------------------------------------------------------------------

    def test_is_cut_predicate(self):
        """Validate IsCut predicate mapping outputs against the active dynamic timeline."""
        state = CutState(target=self.dummy_target, state=CutCondition.CUT)
        self.manager.update_state(self.entity_id, state)

        predicate = IsCut(self.manager, self.entity_id)
        self.assertTrue(predicate())

    def test_is_filled_and_is_empty_predicates(self):
        """Validate IsFilled and IsEmpty predicate variations output cleanly."""
        # Scenario A: Container is empty
        state_empty = FillState(target=self.dummy_target, state=FillLevel.EMPTY)
        self.manager.update_state(self.entity_id, state_empty)

        self.assertTrue(IsEmpty(self.manager, self.entity_id)())
        self.assertFalse(IsFilled(self.manager, self.entity_id)())

        # Scenario B: Container transitions to full
        state_full = FillState(target=self.dummy_target, state=FillLevel.FULL)
        self.manager.update_state(self.entity_id, state_full)

        self.assertFalse(IsEmpty(self.manager, self.entity_id)())
        self.assertTrue(IsFilled(self.manager, self.entity_id)())


if __name__ == "__main__":
    unittest.main()
