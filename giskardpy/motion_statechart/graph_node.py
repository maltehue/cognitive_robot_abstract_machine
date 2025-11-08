from __future__ import annotations

import ast
import re
import threading
from abc import ABC
from dataclasses import field, dataclass, fields
from enum import Enum

from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import (
    Dict,
    Any,
    Self,
    Optional,
    TYPE_CHECKING,
    List,
    TypeVar,
)

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.utils.utils import string_shortener
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
    )


class TransitionKind(Enum):
    START = 1
    PAUSE = 2
    END = 3
    RESET = 4


@dataclass(eq=False, repr=False)
class TrinaryCondition(SubclassJSONSerializer):
    """
    Represents a trinary condition used to define transitions in a motion statechart model.

    This class serves as a representation of a logical trinary condition with three possible states: true, false, and
    unknown. It is used as part of a motion statechart system to define transitions between nodes. The condition is
    evaluated using a logical expression and connects nodes via parent-child relationships. It includes methods to
    create predefined trinary values, update the expression of the condition, and format the condition for display.
    """

    kind: TransitionKind
    """
    The type of transition associated with this condition.
    """
    expression: cas.Expression = cas.TrinaryUnknown
    """
    The logical trinary condition to be evaluated.
    """

    owner: Optional[MotionStatechartNode] = field(default=None)

    def __hash__(self) -> int:
        return hash((str(self), self.kind))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def create_true(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(expression=cas.TrinaryTrue, kind=kind, owner=owner)

    @classmethod
    def create_false(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(expression=cas.TrinaryFalse, kind=kind, owner=owner)

    @classmethod
    def create_unknown(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(
            expression=cas.TrinaryUnknown,
            kind=kind,
            owner=owner,
        )

    def update_expression(
        self, new_expression: cas.Expression, child: MotionStatechartNode
    ) -> None:
        self.expression = new_expression
        self._child = child

    @property
    def variables(self) -> List[MotionStatechartNode]:
        """
        List of parent nodes involved in the condition, derived from the free symbols in the expression.
        """
        return [
            x.motion_statechart_node
            for x in self.expression.free_variables()
            if isinstance(x, ObservationVariable)
        ]

    def __str__(self):
        """
        Replaces the state symbols with motion statechart node names and formats it nicely.
        """
        free_symbols = self.expression.free_variables()
        if not free_symbols:
            str_representation = str(cas.is_const_binary_true(self.expression))
        else:
            str_representation = cas.trinary_logic_to_str(self.expression)
        str_representation = re.sub(
            r'"([^"]*?)/observation"', r'"\1"', str_representation
        )
        return str_representation

    def __repr__(self):
        return str(self)

    def to_json(self) -> Dict[str, Any]:
        json_data = super().to_json()
        json_data["kind"] = self.kind.name
        json_data["expression"] = str(self)
        json_data["owner"] = self.owner.name.to_json() if self.owner else None
        return json_data

    @classmethod
    def create_from_trinary_logic_str(
        cls,
        kind: TransitionKind,
        trinary_logic_str: str,
        observation_variables: List[ObservationVariable],
        owner: Optional[MotionStatechartNode] = None,
    ):
        tree = ast.parse(trinary_logic_str, mode="eval")
        return cls(
            kind=kind,
            expression=cls._parse_ast_expression(tree.body, observation_variables),
            owner=owner,
        )

    @staticmethod
    def _parse_ast_expression(
        node: ast.expr, observation_variables: List[ObservationVariable]
    ) -> cas.Expression:
        match node:
            case ast.BoolOp(op=ast.And()):
                return TrinaryCondition._parse_ast_and(node, observation_variables)
            case ast.BoolOp(op=ast.Or()):
                return TrinaryCondition._parse_ast_or(node, observation_variables)
            case ast.UnaryOp():
                return TrinaryCondition._parse_ast_not(node, observation_variables)
            case ast.Constant(value=str(val)):
                variable_name = PrefixedName("observation", val)
                for v in observation_variables:
                    if variable_name == v.name:
                        return v
                raise KeyError(f"unknown observation variable: {val!r}")
            case ast.Constant(value=True):
                return cas.TrinaryTrue
            case ast.Constant(value=False):
                return cas.TrinaryFalse
            case _:
                raise TypeError(f"failed to parse {type(node).__name__}")

    @staticmethod
    def _parse_ast_and(node, observation_variables: List[ObservationVariable]):
        return cas.trinary_logic_and(
            *[
                TrinaryCondition._parse_ast_expression(x, observation_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_or(node, observation_variables: List[ObservationVariable]):
        return cas.trinary_logic_or(
            *[
                TrinaryCondition._parse_ast_expression(x, observation_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_not(node, observation_variables: List[ObservationVariable]):
        if isinstance(node.op, ast.Not):
            return cas.trinary_logic_not(
                TrinaryCondition._parse_ast_expression(
                    node.operand, observation_variables
                )
            )

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], motion_statechart: MotionStatechart, **kwargs
    ) -> Self:
        return cls.create_from_trinary_logic_str(
            kind=TransitionKind[data["kind"]],
            trinary_logic_str=data["expression"],
            observation_variables=motion_statechart.observation_state.observation_symbols(),
            owner=motion_statechart.get_node_by_name(
                PrefixedName.from_json(data["owner"])
            ),
        )


@dataclass(repr=False, eq=False)
class ObservationVariable(cas.FloatVariable):
    """
    A symbol representing the observation state of a node.
    """

    name: PrefixedName = field(kw_only=True)
    motion_statechart_node: MotionStatechartNode

    def resolve(self) -> float:
        return self.motion_statechart_node.observation_state


@dataclass(repr=False, eq=False)
class LifeCycleVariable(cas.FloatVariable):
    """
    A symbol representing the life cycle state of a node.
    """

    name: PrefixedName = field(kw_only=True)
    motion_statechart_node: MotionStatechartNode

    def resolve(self) -> LifeCycleValues:
        return self.motion_statechart_node.life_cycle_state


@dataclass
class BuildContext:
    world: World


@dataclass
class NodeArtifacts:
    constraints: ConstraintCollection = field(default_factory=ConstraintCollection)
    observation: cas.Expression = field(default_factory=lambda: cas.TrinaryUnknown)


@dataclass(repr=False, eq=False)
class MotionStatechartNode(SubclassJSONSerializer):
    name: PrefixedName = field(kw_only=True)
    """
    A unique name for the node within a motion statechart.
    """

    _motion_statechart: MotionStatechart = field(init=False)
    """
    Back reference to the motion statechart that owns this node.
    """
    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    parent_node: MotionStatechartNode = field(default=None, init=False)
    """
    The parent node of this node, if None, it is on the top layer of a motion statechart.
    """

    _life_cycle_variable: LifeCycleVariable = field(init=False)
    """
    A symbol referring to the life cycle state of this node.
    """
    _observation_variable: ObservationVariable = field(init=False)

    _constraint_collection: ConstraintCollection = field(init=False)
    _observation_expression: cas.Expression = field(init=False)

    _start_condition: TrinaryCondition = field(init=False)
    _pause_condition: TrinaryCondition = field(init=False)
    _end_condition: TrinaryCondition = field(init=False)
    _reset_condition: TrinaryCondition = field(init=False)

    _plot: bool = field(default=True, kw_only=True)
    _plot_style: str = field(default="filled, rounded", init=False)
    _plot_shape: str = field(default="rectangle", init=False)
    _plot_extra_boarder_styles: List[str] = field(default_factory=list, kw_only=True)

    def __post_init__(self):
        self._observation_variable = ObservationVariable(
            name=PrefixedName("observation", str(self.name)),
            motion_statechart_node=self,
        )
        self._life_cycle_variable = LifeCycleVariable(
            name=PrefixedName("life_cycle", str(self.name)),
            motion_statechart_node=self,
        )
        self._start_condition = TrinaryCondition.create_true(
            kind=TransitionKind.START, owner=self
        )
        self._pause_condition = TrinaryCondition.create_false(
            kind=TransitionKind.PAUSE, owner=self
        )
        self._end_condition = TrinaryCondition.create_false(
            kind=TransitionKind.END, owner=self
        )
        self._reset_condition = TrinaryCondition.create_false(
            kind=TransitionKind.RESET, owner=self
        )

    def set_transition(self, transition: TrinaryCondition) -> None:
        match transition.kind:
            case TransitionKind.START:
                self._start_condition = transition
            case TransitionKind.PAUSE:
                self._pause_condition = transition
            case TransitionKind.END:
                self._end_condition = transition
            case TransitionKind.RESET:
                self._reset_condition = transition
            case _:
                raise ValueError(f"Unknown transition kind: {transition.kind}")

    @property
    def life_cycle_variable(self) -> LifeCycleVariable:
        return self._life_cycle_variable

    @property
    def observation_variable(self) -> ObservationVariable:
        return self._observation_variable

    @property
    def motion_statechart(self) -> MotionStatechart:
        if not hasattr(self, "_motion_statechart"):
            raise AttributeError(
                f"Motion statechart not set for {self.__class__.__name__} {self.name}"
            )
        return self._motion_statechart

    @motion_statechart.setter
    def motion_statechart(self, motion_statechart: MotionStatechart) -> None:
        self._motion_statechart = motion_statechart

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        Describe this node by returning its constraints and the observation expression.
        Called exactly once during motion statechart compilation.
        """
        return NodeArtifacts(
            constraints=ConstraintCollection(),
            observation=cas.Expression(self.observation_variable),
        )

    def on_tick(self) -> Optional[float]:
        """
        Triggered when the node is ticked.
        .. warning:: Only happens while the node is in state RUNNING.
        :return: An optional observation state overwrite
        """

    def on_start(self):
        """
        Triggered when the node transitions from NOT_STARTED to RUNNING.
        """

    def on_pause(self):
        """
        Triggered when the node transitions from RUNNING to PAUSED.
        """

    def on_unpause(self):
        """
        Triggered when the node transitions from PAUSED to RUNNING.
        """

    def on_end(self):
        """
        Triggered when the node transitions from RUNNING to DONE.
        """

    def on_reset(self):
        """
        Triggered when the node transitions from any state to NOT_STARTED.
        """

    @property
    def world(self) -> World:
        return self.motion_statechart.world

    def __hash__(self):
        return hash(self.name)

    @property
    def life_cycle_state(self) -> LifeCycleValues:
        return LifeCycleValues(self.motion_statechart.life_cycle_state[self])

    @property
    def observation_state(self) -> float:
        return self.motion_statechart.observation_state[self]

    @property
    def start_condition(self) -> cas.Expression:
        return self._start_condition.expression

    @start_condition.setter
    def start_condition(self, expression: cas.Expression) -> None:
        self._start_condition.update_expression(expression, self)

    @property
    def pause_condition(self) -> cas.Expression:
        return self._pause_condition.expression

    @pause_condition.setter
    def pause_condition(self, expression: cas.Expression) -> None:
        self._pause_condition.update_expression(expression, self)

    @property
    def end_condition(self) -> cas.Expression:
        return self._end_condition.expression

    @end_condition.setter
    def end_condition(self, expression: cas.Expression) -> None:
        self._end_condition.update_expression(expression, self)

    @property
    def reset_condition(self) -> cas.Expression:
        return self._reset_condition.expression

    @reset_condition.setter
    def reset_condition(self, expression: cas.Expression) -> None:
        self._reset_condition.update_expression(expression, self)

    def to_json(self) -> Dict[str, Any]:
        json_data = super().to_json()
        for field_ in fields(self):
            if not field_.name.startswith("_") and field_.init:
                value = getattr(self, field_.name)
                json_data[field_.name] = self._attribute_to_json(value)
        return json_data

    def _attribute_to_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return self._dict_to_json(value)
        if isinstance(value, list):
            return self._list_to_json(value)
        if isinstance(value, SubclassJSONSerializer):
            return value.to_json()
        return value

    def _list_to_json(self, list_attr: list) -> list:
        return [self._attribute_to_json(value) for value in list_attr]

    def _dict_to_json(self, dict_attr: Dict[Any, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in dict_attr.items():
            json_value = self._attribute_to_json(value)
            result[key] = json_value
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        node_kwargs = {}
        del data["type"]
        for field_name, field_data in data.items():
            if isinstance(field_data, dict) and "type" in field_data:
                field_data = SubclassJSONSerializer.from_json(field_data, **kwargs)
            node_kwargs[field_name] = field_data
        return cls(**node_kwargs)

    def formatted_name(self, quoted: bool = False) -> str:
        formatted_name = string_shortener(
            original_str=str(self.name), max_lines=4, max_line_length=25
        )
        result = (
            f"{formatted_name}\n"
            f"----start_condition----\n"
            f"{str(self._start_condition)}\n"
            f"----pause_condition----\n"
            f"{str(self._pause_condition)}\n"
            f"----end_condition----\n"
            f"{str(self._end_condition)}\n"
            f"----reset_condition----\n"
            f"{str(self._reset_condition)}"
        )
        if quoted:
            return '"' + result + '"'
        return result

    def __repr__(self) -> str:
        return str(self.name)


GenericMotionStatechartNode = TypeVar(
    "GenericMotionStatechartNode", bound=MotionStatechartNode
)


@dataclass(eq=False, repr=False)
class Goal(MotionStatechartNode):
    nodes: List[MotionStatechartNode] = field(default_factory=list)
    _plot_style: str = field(default="filled", init=False)
    _plot_shape: str = field(default="none", init=False)

    @property
    def motion_statechart(self) -> MotionStatechart:
        if not hasattr(self, "_motion_statechart"):
            raise AttributeError(
                f"Motion statechart not set for {self.__class__.__name__} {self.name}"
            )
        return self._motion_statechart

    @motion_statechart.setter
    def motion_statechart(self, motion_statechart: MotionStatechart) -> None:
        self._motion_statechart = motion_statechart
        self._link_child_nodes_with_motion_statechart()

    def add_node(self, node: MotionStatechartNode) -> None:
        self.nodes.append(node)
        node.parent_node = self
        self._link_child_nodes_with_motion_statechart()

    def _link_child_nodes_with_motion_statechart(self) -> None:
        if hasattr(self, "_motion_statechart"):
            for node in self.nodes:
                node.motion_statechart = self.motion_statechart
                if not self.motion_statechart.has_node(node):
                    self.motion_statechart.add_node(node)

    def arrange_in_sequence(self, nodes: List[MotionStatechartNode]) -> None:
        first_node = nodes[0]
        first_node.end_condition = first_node.observation_variable
        for node in nodes[1:]:
            node.start_condition = first_node.observation_variable
            node.end_condition = node.observation_variable
            first_node = node

    def apply_goal_conditions_to_children(self):
        for node in self.nodes:
            self.apply_start_condition_to_node(node)
            self.apply_pause_condition_to_node(node)
            self.apply_end_condition_to_node(node)
            self.apply_reset_condition_to_node(node)
            if isinstance(node, Goal):
                node.apply_goal_conditions_to_children()

    def apply_start_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_const_trinary_true(node.start_condition):
            node.start_condition = self.start_condition
            return
        node.start_condition = cas.trinary_logic_and(
            node.start_condition, self.start_condition
        )

    def apply_pause_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_const_trinary_false(node.pause_condition):
            node.pause_condition = self.pause_condition
        elif not cas.is_const_trinary_false(node.pause_condition):
            node.pause_condition = cas.trinary_logic_or(
                node.pause_condition, self.pause_condition
            )

    def apply_end_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_const_trinary_false(node.end_condition):
            node.end_condition = self.end_condition
        elif not cas.is_const_trinary_false(self.end_condition):
            node.end_condition = cas.trinary_logic_or(
                node.end_condition, self.end_condition
            )

    def apply_reset_condition_to_node(self, node: MotionStatechartNode):
        if cas.is_const_trinary_false(node.reset_condition):
            node.reset_condition = self.reset_condition
        elif not cas.is_const_trinary_false(node.pause_condition):
            node.reset_condition = cas.trinary_logic_or(
                node.reset_condition, self.reset_condition
            )


@dataclass(eq=False, repr=False)
class ThreadPayloadMonitor(MotionStatechartNode, ABC):
    """
    Payload monitor that evaluates _compute_observation in a background thread.

    - compute_observation triggers an async evaluation and immediately returns.
    - Until the first successful completion, returns TrinaryUnknown.
    - Afterwards, returns the last successfully computed value.
    """

    # Internal threading primitives
    _request_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _thread: threading.Thread = field(init=False, repr=False)

    # Cache of last successful result from _compute_observation
    _has_result: bool = field(default=False, init=False, repr=False)
    _last_result: float = field(
        default=float(cas.TrinaryUnknown.to_np()), init=False, repr=False
    )

    def __post_init__(self):
        super().__post_init__()
        # Start a daemon worker thread that computes observations when requested
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.__class__.__name__}-worker",
            daemon=True,
        )
        self._thread.start()

    def compute_observation(
        self,
    ) -> float:
        # Signal the worker to compute a fresh value if it is not already signaled.
        self._request_event.set()
        # Return the last known result (initialized to Unknown until first success)
        return self._last_result

    def _worker_loop(self):
        while not self._stop_event.is_set():
            # Wait until a request is made (wake periodically to check for stop)
            triggered = self._request_event.wait(timeout=0.1)
            if not triggered:
                continue
            # Clear early to allow new requests while we compute
            self._request_event.clear()
            try:
                result = self._compute_observation()
                # Accept only valid trinary values (floats expected)
                self._last_result = result
                self._has_result = True
            except Exception:
                # On failure, keep previous result and mark as having no new value
                pass


@dataclass(eq=False, repr=False)
class EndMotion(MotionStatechartNode):
    _plot_style: str = field(default="filled, rounded", init=False)
    _plot_shape: str = field(default="rectangle", init=False)
    _plot_boarder_styles: List[str] = field(
        default_factory=lambda: ["rounded"], kw_only=True
    )

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=cas.TrinaryTrue)


@dataclass(eq=False, repr=False)
class CancelMotion(MotionStatechartNode):
    exception: Exception = field(kw_only=True)
    observation_expression: cas.Expression = field(
        default_factory=lambda: cas.TrinaryTrue, init=False
    )
    _plot_extra_boarder_styles: List[str] = field(
        default_factory=lambda: ["dashed, rounded"], kw_only=True
    )
    _plot_style: str = field(default="filled, rounded", init=False)
    _plot_shape: str = field(default="rectangle", init=False)

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=cas.TrinaryTrue)

    def on_tick(self) -> Optional[float]:
        raise self.exception
