"""Compiles a theory specification into a runnable symbolic theory.

The compiler does everything a synthesizer must not be trusted with: it validates every condition
against the whitelist grammar and the situation type's facts, creates the decision types the
specification names, builds the declarations with their gates and channels, and assembles the
ripple-down rule chains — including defeaters and cross-family chaining — in the order-sensitive
way the engine requires. What comes out is the same kind of object a hand-written theory builder
returns, so everything downstream is indifferent to whether a person or a model wrote the theory.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, make_dataclass
from dataclasses import fields as dataclass_fields

from typing_extensions import Dict, FrozenSet, List, Mapping, Optional, Tuple, Type

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.rdr import GeneralRDR, MultiClassRDR
from krrood.ripple_down_rules.rules import MultiClassStopRule, MultiClassTopRule

from semantic_digital_twin.reasoning.knowledge_servoing.condition_validation import (
    ConditionValidator,
)
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ConstraintDeclaration,
    ParameterChannel,
)
from semantic_digital_twin.reasoning.knowledge_servoing.exceptions import (
    DecisionRoleConflictError,
    InvalidDeclarationParametersError,
    InvalidRuleValueError,
    UnconcludableDecisionError,
    UnknownDecisionNameError,
    UnknownDeclarationKindError,
)
from semantic_digital_twin.reasoning.knowledge_servoing.general_rdr_theory import (
    GeneralRDRTheory,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ParameterDecision,
    RegimeDecision,
    Situation,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_specification import (
    ConstraintSpecification,
    RuleSpecification,
    TheorySpecification,
)

REGIME_FAMILY = "regime_decisions"
"""Attribute name of the compiled regime decision family."""

PARAMETER_FAMILY = "parameter_decisions"
"""Attribute name of the compiled parameter decision family."""

PARAMETER_VALUE_FIELD = "value"
"""Field name every compiled parameter decision carries its value in."""

_DECLARATION_OWN_FIELDS = ("identifier", "gating_decision_type", "parameter_channel")
"""Declaration fields the compiler fills itself rather than from specification parameters."""


@dataclass
class TheoryCompiler:
    """Turns validated specifications into theories over one situation type."""

    declaration_kinds: Mapping[str, Type[ConstraintDeclaration]]
    """The constraint vocabulary: kind name to declaration type."""

    situation_type: Type[Situation]
    """The situation type the compiled rules read their facts from."""

    _condition_validator: ConditionValidator = field(init=False, repr=False)
    """Validates every condition and value expression before it is compiled."""

    def __post_init__(self) -> None:
        self._condition_validator = ConditionValidator(
            situation_type=self.situation_type
        )

    def compile(self, specification: TheorySpecification) -> GeneralRDRTheory:
        """Compiles a specification into a runnable theory.

        :param specification: The theory as data.
        :return: The compiled theory, declarations attached, ready for the chart assembler.
        """
        regime_names, parameter_names = self._decision_roles(specification)
        decision_types = self._create_decision_types(regime_names, parameter_names)
        self._validate_rules(specification, regime_names, parameter_names)

        declarations = tuple(
            self._build_declaration(constraint, decision_types)
            for constraint in specification.constraints
        )
        rule_set = GeneralRDR()
        regime_rules = [
            rule for rule in specification.rules if rule.concludes in regime_names
        ]
        parameter_rules = [
            rule for rule in specification.rules if rule.concludes in parameter_names
        ]
        if regime_rules:
            rule_set.add_rdr(
                self._compile_family(regime_rules, REGIME_FAMILY, decision_types)
            )
        if parameter_rules:
            rule_set.add_rdr(
                self._compile_family(parameter_rules, PARAMETER_FAMILY, decision_types)
            )
        return GeneralRDRTheory(
            rule_set=rule_set,
            declared_decision_types=frozenset(decision_types.values()),
            constraint_declarations=declarations,
        )

    # %% decision roles and types

    def _decision_roles(
        self, specification: TheorySpecification
    ) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        """Splits the decision names into regime and parameter roles from constraint usage.

        :param specification: The theory as data.
        :return: The regime decision names and the parameter decision names.
        :raises DecisionRoleConflictError: if one name is used in both roles.
        :raises UnconcludableDecisionError: if a referenced decision is concluded by no rule.
        """
        regime_names = frozenset(
            constraint.gated_by
            for constraint in specification.constraints
            if constraint.gated_by is not None
        )
        parameter_names = frozenset(
            constraint.value_from
            for constraint in specification.constraints
            if constraint.value_from is not None
        )
        for name in regime_names & parameter_names:
            raise DecisionRoleConflictError(name=name)
        concluded = {rule.concludes for rule in specification.rules}
        for name in (regime_names | parameter_names) - concluded:
            raise UnconcludableDecisionError(name=name)
        return regime_names, parameter_names

    @staticmethod
    def _create_decision_types(
        regime_names: FrozenSet[str], parameter_names: FrozenSet[str]
    ) -> Dict[str, type]:
        """Creates one frozen decision type per name, on the channel its role implies.

        :param regime_names: Names used to gate constraints.
        :param parameter_names: Names used to supply constraint values.
        :return: The decision types by name.
        """
        decision_types: Dict[str, type] = {}
        for name in sorted(regime_names):
            decision_types[name] = make_dataclass(
                name,
                (),
                bases=(RegimeDecision,),
                frozen=True,
                namespace={"__doc__": f"Compiled regime decision '{name}'."},
            )
        for name in sorted(parameter_names):
            decision_types[name] = make_dataclass(
                name,
                ((PARAMETER_VALUE_FIELD, float),),
                bases=(ParameterDecision,),
                frozen=True,
                namespace={"__doc__": f"Compiled parameter decision '{name}'."},
            )
        return decision_types

    # %% rule validation

    def _validate_rules(
        self,
        specification: TheorySpecification,
        regime_names: FrozenSet[str],
        parameter_names: FrozenSet[str],
    ) -> None:
        """Validates every rule's target, conditions and value before compilation.

        :param specification: The theory as data.
        :param regime_names: Names used to gate constraints.
        :param parameter_names: Names used to supply constraint values.
        :raises UnknownDecisionNameError: if a rule concludes or requires an unknown name.
        :raises InvalidRuleValueError: if a rule's value does not fit its channel.
        """
        known_names = regime_names | parameter_names
        for rule in specification.rules:
            if rule.concludes not in known_names:
                raise UnknownDecisionNameError(
                    name=rule.concludes, known_names=known_names
                )
            for required in rule.requires_concluded:
                if required not in regime_names:
                    raise UnknownDecisionNameError(
                        name=required, known_names=regime_names
                    )
            self._condition_validator.validate(rule.condition)
            for defeater in rule.defeated_by:
                self._condition_validator.validate(defeater)
            if rule.concludes in regime_names and rule.value is not None:
                raise InvalidRuleValueError(
                    concludes=rule.concludes, reason="regime rules carry no value"
                )
            if rule.concludes in parameter_names:
                if rule.value is None:
                    raise InvalidRuleValueError(
                        concludes=rule.concludes,
                        reason="parameter rules must carry one",
                    )
                self._condition_validator.validate(rule.value)

    # %% declarations

    def _build_declaration(
        self,
        constraint: ConstraintSpecification,
        decision_types: Mapping[str, type],
    ) -> ConstraintDeclaration:
        """Builds one declaration from a constraint specification.

        :param constraint: The constraint as data.
        :param decision_types: The compiled decision types by name.
        :return: The declaration, gate and channel attached.
        :raises UnknownDeclarationKindError: if the kind is outside the registry.
        :raises InvalidDeclarationParametersError: if the parameters do not fit the kind.
        """
        declaration_type = self.declaration_kinds.get(constraint.kind)
        if declaration_type is None:
            raise UnknownDeclarationKindError(
                kind=constraint.kind, known_kinds=frozenset(self.declaration_kinds)
            )
        self._require_parameters_fit(constraint, declaration_type)
        parameter_channel: Optional[ParameterChannel] = None
        if constraint.value_from is not None:
            parameter_channel = ParameterChannel(
                decision_type=decision_types[constraint.value_from],
                attribute_name=PARAMETER_VALUE_FIELD,
            )
        gating_decision_type = (
            decision_types[constraint.gated_by]
            if constraint.gated_by is not None
            else None
        )
        return declaration_type(
            identifier=constraint.identifier,
            gating_decision_type=gating_decision_type,
            parameter_channel=parameter_channel,
            **constraint.parameters,
        )

    @staticmethod
    def _require_parameters_fit(
        constraint: ConstraintSpecification,
        declaration_type: Type[ConstraintDeclaration],
    ) -> None:
        """Rejects parameters that do not match the declaration kind's fields.

        :param constraint: The constraint as data.
        :param declaration_type: The kind's declaration type.
        :raises InvalidDeclarationParametersError: on unexpected or missing parameters.
        """
        declared = {
            declared_field.name: declared_field
            for declared_field in dataclass_fields(declaration_type)
            if declared_field.name not in _DECLARATION_OWN_FIELDS
        }
        unexpected = tuple(
            name for name in constraint.parameters if name not in declared
        )
        missing = tuple(
            name
            for name, declared_field in declared.items()
            if declared_field.default is MISSING
            and declared_field.default_factory is MISSING
            and name not in constraint.parameters
        )
        if unexpected or missing:
            raise InvalidDeclarationParametersError(
                kind=constraint.kind, unexpected=unexpected, missing=missing
            )

    # %% rule compilation

    def _compile_family(
        self,
        rules: List[RuleSpecification],
        family: str,
        decision_types: Mapping[str, type],
    ) -> MultiClassRDR:
        """Compiles one decision family's rules into a chained multi-class classifier.

        :param rules: The family's rules, in specification order.
        :param family: The attribute name the family's conclusions accumulate under.
        :param decision_types: The compiled decision types by name.
        :return: The family's classifier.
        """
        top_rules = [self._compile_rule(rule, family, decision_types) for rule in rules]
        for previous_rule, next_rule in zip(top_rules, top_rules[1:]):
            previous_rule.alternative = next_rule
        rule_set = MultiClassRDR(start_rule=top_rules[0])
        rule_set.name = family
        return rule_set

    def _compile_rule(
        self,
        rule: RuleSpecification,
        family: str,
        decision_types: Mapping[str, type],
    ) -> MultiClassTopRule:
        """Compiles one rule, with its chaining requirements and defeaters.

        :param rule: The rule as data.
        :param family: The attribute name the rule's conclusion accumulates under.
        :param decision_types: The compiled decision types by name.
        :return: The compiled top rule.
        """
        decision_type = decision_types[rule.concludes]
        condition_sources = [f"({rule.condition})"]
        scope: Dict[str, type] = {rule.concludes: decision_type}
        for required in rule.requires_concluded:
            condition_sources.append(
                f"any(isinstance(decision, {required}) "
                f"for decision in case.{REGIME_FAMILY})"
            )
            scope[required] = decision_types[required]
        conclusion_source = (
            f"{rule.concludes}()"
            if rule.value is None
            else f"{rule.concludes}({rule.value})"
        )
        top_rule = MultiClassTopRule(
            conditions=CallableExpression(
                user_input=" and ".join(condition_sources), scope=dict(scope)
            ),
            conclusion=CallableExpression(
                user_input=conclusion_source,
                scope=dict(scope),
                conclusion_type=(decision_type,),
            ),
            conclusion_name=family,
        )
        previous_stop_rule: Optional[MultiClassStopRule] = None
        for defeater in rule.defeated_by:
            stop_rule = MultiClassStopRule(
                conditions=CallableExpression(user_input=defeater, scope={})
            )
            stop_rule.top_rule = top_rule
            if previous_stop_rule is None:
                top_rule.refinement = stop_rule
            else:
                previous_stop_rule.alternative = stop_rule
            previous_stop_rule = stop_rule
        return top_rule
