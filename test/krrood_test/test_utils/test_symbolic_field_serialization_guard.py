"""
Guard against silently serializing a symbolic-valued field to JSON.

A symbolic value carrying free variables (a runtime-retargetable parameter) has no
faithful JSON representation: it serializes to a bare type marker and deserializes back
to zero. A record that round-trips through JSON must therefore fail loudly rather than
quietly losing its value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.exceptions import SymbolicValueNotSerializableError
from krrood.adapters.json_serializer import REJECTS_SYMBOLIC_VALUES, to_json


@dataclass
class SymbolicallyParameterizedRecord:
    """
    A record whose value may be a runtime-retargetable symbolic parameter.
    """

    value: sm.ScalarData = field(metadata={REJECTS_SYMBOLIC_VALUES: True})


@dataclass
class SymbolicallyExpressedRecord:
    """
    A record whose value is an expression, for which a symbolic form is normal content.
    """

    expression: sm.ScalarData


class TestSymbolicFieldSerializationGuard:
    """
    Whether serializing a free-variable-bearing field raises instead of losing the
    value.
    """

    def test_serializing_a_free_variable_field_raises(self):
        record = SymbolicallyParameterizedRecord(value=sm.FloatVariable(name="goal"))
        with pytest.raises(SymbolicValueNotSerializableError):
            to_json(record)

    def test_serializing_a_plain_float_field_is_unaffected(self):
        record = SymbolicallyParameterizedRecord(value=0.7)
        assert to_json(record)["value"] == 0.7

    def test_a_field_that_did_not_opt_in_serializes_a_symbolic_value(self):
        """
        Only fields that declared the marker are guarded, so expression-valued fields
        such as a coupled inflow equation's term keep serializing as before.
        """
        record = SymbolicallyExpressedRecord(expression=sm.FloatVariable(name="tilt"))
        assert "expression" in to_json(record)
