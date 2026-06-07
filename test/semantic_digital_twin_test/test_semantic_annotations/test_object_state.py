"""
Unit tests for ObjectState.

Validates:
- default initialization
- VLM-based state assignment
- SDT-based structural state tracking
"""

import unittest
from unittest.mock import MagicMock

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from semantic_digital_twin.semantic_annotations.object_state import (
    ObjectState,
    FillState,
    CutState,
)


class TestObjectState(unittest.TestCase):

    def setUp(self):
        # Create a dummy target before each test runs
        self.dummy_target = MagicMock(spec=SemanticAnnotation)

    def test_default_initialization(self):
        """Test that an empty state initializes correctly."""
        state = ObjectState(target=self.dummy_target)

        self.assertIsNone(state.fill_state, "Default fill_state should be None")
        self.assertIsNone(state.cut_state, "Default cut_state should be None")
        self.assertEqual(state.confidence, 1.0, "Default confidence should be 1.0")
        self.assertEqual(state.source, "SDT", "Default source should be 'SDT'")
        self.assertIsNone(state.timestamp, "Default timestamp should be None")

        # Test your properties
        self.assertFalse(state.is_fill_known)
        self.assertFalse(state.is_cut_known)
        self.assertFalse(state.has_state)

    def test_vlm_fill_prediction(self):
        """Test assigning a state from a VLM (like CLIP)."""
        clip_state = ObjectState(
            target=self.dummy_target,
            fill_state=FillState.FULL,
            confidence=0.85,
            source="CLIP",
            timestamp=1620000000.0,
        )

        self.assertEqual(clip_state.fill_state, FillState.FULL)
        self.assertTrue(clip_state.is_fill_known)
        self.assertFalse(clip_state.is_cut_known)
        self.assertTrue(clip_state.has_state)

    def test_sdt_cut_tracking(self):
        """Test tracking a physical structural change."""
        sdt_state = ObjectState(target=self.dummy_target, cut_state=CutState.CUT)

        self.assertEqual(sdt_state.cut_state, CutState.CUT)
        self.assertTrue(sdt_state.is_cut_known)
        self.assertFalse(sdt_state.is_fill_known)
        self.assertTrue(sdt_state.has_state)


if __name__ == "__main__":
    unittest.main()
