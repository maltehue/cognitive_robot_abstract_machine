import numpy as np
from random_events.product_algebra import SimpleEvent

from random_events.interval import singleton, open_closed, open, closed

from random_events.variable import Continuous

from krrood.entity_query_language.factories import (
    match_variable,
    variable,
    entity,
    match,
    variable_from,
    and_,
    or_,
)
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.ormatic.dao import to_dao
from krrood.probabilistic_knowledge.parameterizer import Parameterizer
from krrood.probabilistic_knowledge.probable_variable import (
    QueryToRandomEventTranslator,
    is_disjunctive_normal_form,
)
from ..dataset.example_classes import Position, Pose, Orientation, Positions
from ..dataset.ormatic_interface import *  # type: ignore


def test_parameterizer_with_where():
    pose_variable = variable(Pose, None)

    q = entity(pose_variable).where(
        pose_variable.position.y > 0,
        pose_variable.position.x == 0,
        pose_variable.position.y < 10,
        pose_variable.position.z >= -1,
        pose_variable.position.z <= 1,
        pose_variable.orientation.x != 1,
    )
    t = QueryToRandomEventTranslator(q)
    r = t.translate()

    result_by_hand = SimpleEvent(
        {
            Continuous("Pose.orientation.x"): ~singleton(1.0),
            Continuous("Pose.position.y"): open(0.0, 10),
            Continuous("Pose.position.z"): closed(-1.0, 1.0),
            Continuous("Pose.position.x"): singleton(0.0),
        }
    )

    assert result_by_hand == r


def test_dnf_checking():
    pose_variable = variable(Pose, None)

    q = entity(pose_variable).where(
        and_(
            or_(
                pose_variable.position.y > 0,
                pose_variable.position.x == 0,
            ),
            or_(
                pose_variable.position.z >= -1,
                pose_variable.position.x == 0,
            ),
        )
    )

    assert not is_disjunctive_normal_form(q)

    q = entity(pose_variable).where(
        or_(
            and_(
                pose_variable.position.y > 0,
                pose_variable.position.x == 0,
            ),
            and_(
                pose_variable.position.z >= -1,
                pose_variable.position.z <= 1,
            ),
        )
    )
    assert is_disjunctive_normal_form(q)
