"""Spike tests for authoring a defeasible theory as a ``MultiClassRDR``.

These pin the three engine behaviours the knowledge-servoing plan
(`doc/knowledge_servoing/implementation_plan.md` §4.1.1) relies on before any substance-transfer
theory is written: whether a frozen dataclass survives use as an RDR ``Case``, whether stop rules and
multiple conclusions behave as claimed, and whether the generated-code path agrees with the in-memory
tree. The theory under test is a domain-free toy (three booleans, three regime conclusions) so the
tests exercise the mechanism, not pouring.
"""

from __future__ import annotations

import os
import shutil

import pytest

from krrood.ripple_down_rules.helpers import update_case_with_conclusion_output
from krrood.ripple_down_rules.utils import copy_case, make_set

from ..dataset.knowledge_servoing_case import (
    CONCLUSION_ATTRIBUTE_NAME,
    EngageRegime,
    EscalateRegime,
    FrozenSituationWithAccumulator,
    FrozenSituationWithoutAccumulator,
    MutableSituation,
    RestrictRegime,
    build_regime_multi_class_rdr,
)


def _write_engage_conclusion(case: object) -> None:
    """Write a single :class:`EngageRegime` conclusion into ``case`` the way ``_classify`` does."""
    update_case_with_conclusion_output(
        case, {EngageRegime()}, CONCLUSION_ATTRIBUTE_NAME, (EngageRegime,), False
    )


# %% item 1 — a frozen dataclass as an RDR Case


class TestFrozenCaseAsMultiClassRdrCase:
    """Whether ``copy_case`` plus conclusion-writing survives a frozen dataclass ``Case``."""

    def test_copy_case_returns_a_distinct_equal_frozen_instance(self):
        case = FrozenSituationWithoutAccumulator(True, False, False)
        copy = copy_case(case)
        assert copy is not case
        assert copy == case
        assert type(copy) is FrozenSituationWithoutAccumulator

    def test_writing_a_conclusion_into_a_frozen_case_without_accumulator_raises(self):
        copy = copy_case(FrozenSituationWithoutAccumulator(True, False, False))
        with pytest.raises(AttributeError):
            _write_engage_conclusion(copy)

    def test_copy_of_frozen_case_shares_its_accumulator_and_leaks_the_conclusion(self):
        original = FrozenSituationWithAccumulator(True, False, False)
        copy = copy_case(original)
        assert copy.conclusions is original.conclusions
        _write_engage_conclusion(copy)
        assert original.conclusions == {EngageRegime()}

    def test_copy_of_mutable_case_isolates_its_accumulator(self):
        original = MutableSituation(True, False, False)
        copy = copy_case(original)
        assert copy.conclusions is not original.conclusions
        _write_engage_conclusion(copy)
        assert copy.conclusions == {EngageRegime()}
        assert original.conclusions == set()


# %% item 2 — multiple conclusions, stop rules, intra-pass chaining


class TestMultiClassConclusionAndStopBehaviour:
    """Whether the engine draws several conclusions, honours stop rules, and chains within a pass."""

    def test_independent_top_rules_each_contribute_in_one_pass(self):
        reasoner = build_regime_multi_class_rdr()
        conclusions = make_set(reasoner.classify(MutableSituation(True, True, False)))
        assert conclusions == {EngageRegime(), RestrictRegime(), EscalateRegime()}

    def test_stop_rule_blocks_the_parent_conclusion_and_adds_nothing(self):
        reasoner = build_regime_multi_class_rdr()
        conclusions = make_set(reasoner.classify(MutableSituation(True, False, True)))
        assert conclusions == set()

    def test_a_later_rule_chains_on_an_earlier_conclusion(self):
        reasoner = build_regime_multi_class_rdr()
        conclusions = make_set(reasoner.classify(MutableSituation(True, False, False)))
        assert conclusions == {EngageRegime(), EscalateRegime()}

    def test_chaining_does_not_fire_without_its_antecedent(self):
        reasoner = build_regime_multi_class_rdr()
        conclusions = make_set(reasoner.classify(MutableSituation(False, True, False)))
        assert conclusions == {RestrictRegime()}


# %% item 3 — generated code agrees with the in-memory tree


class TestGeneratedAndInMemoryClassificationAgree:
    """Whether ``_write_to_python`` output classifies identically to the in-memory tree."""

    def test_generated_code_matches_in_memory_for_every_input(self):
        reasoner = build_regime_multi_class_rdr()
        # The loader imports the generated module by its package path, so it must live under an
        # importable, git-ignored directory rather than a bare temporary one.
        generated_dir = os.path.join(
            os.path.dirname(__file__), "test_generated_rdrs", "knowledge_servoing_spike"
        )
        shutil.rmtree(generated_dir, ignore_errors=True)
        try:
            reasoner._write_to_python(generated_dir)
            generated_classify = reasoner.get_rdr_classifier_from_python_file(
                generated_dir
            )
            for engagement in (True, False):
                for restriction in (True, False):
                    for defeated in (True, False):
                        case = MutableSituation(engagement, restriction, defeated)
                        in_memory = make_set(reasoner.classify(case))
                        generated = make_set(
                            generated_classify(
                                MutableSituation(engagement, restriction, defeated)
                            )
                        )
                        assert generated == in_memory
        finally:
            shutil.rmtree(generated_dir, ignore_errors=True)
