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
)
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.ormatic.dao import to_dao
from krrood.probabilistic_knowledge.parameterizer import Parameterizer
from krrood.probabilistic_knowledge.probable_variable import (
    QueryToRandomEventTranslator,
)
from ..dataset.example_classes import Position, Pose, Orientation
from ..dataset.ormatic_interface import *  # type: ignore


def test_parameterizer_with_where():
    pose = Pose(
        position=Position(..., ..., ...),
        orientation=Orientation(..., ..., ..., None),
    )

    pose_dao_variable = variable(PoseDAO, [to_dao(pose)])
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
