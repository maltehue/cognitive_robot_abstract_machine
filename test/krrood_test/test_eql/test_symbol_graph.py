import os
import sys
import threading

import pytest
from typing_extensions import List

from krrood.entity_query_language.factories import entity, variable, an
from krrood.symbol_graph.symbol_graph import SymbolGraph
from ..dataset.example_classes import KRROODPosition

SWEEP_TIMEOUT = 30.0
"""
How long a thread waits for the others while creating and sweeping instances.
"""

try:
    import pydot
    import pygraphviz
except ImportError:
    pydot = None
    pygraphviz = None


@pytest.mark.skipif(
    not (pydot and pygraphviz), reason="pydot and graphviz not installed"
)
def test_visualize_symbol_graph():
    SymbolGraph().clear()
    symbol_graph = SymbolGraph()
    symbol_graph.to_dot("symbol_graph.svg", format_="svg", graph_type="type")
    assert len(symbol_graph._class_diagram.wrapped_classes) >= 59
    if os.path.exists("symbol_graph.svg"):
        os.remove("symbol_graph.svg")


def test_memory_leak():
    """
    Test if the SymbolGraph does not artificially keep objects alive that would be
    garbage collected.
    """

    def create_data():
        point = KRROODPosition(1, 2, 3)
        return point

    create_data()

    q = an(entity(variable(KRROODPosition, domain=None)))
    result = list(q.evaluate())

    assert result == []

    assert len(SymbolGraph().wrapped_instances) == 0


# %% concurrent bookkeeping

SWEEPING_THREADS = 4
"""
How many threads create and sweep instances at the same time.
"""

SWEEPS_PER_THREAD = 10
"""
How often each thread creates a batch of instances and sweeps the graph.
"""

INSTANCES_PER_SWEEP = 200
"""
How many instances a thread lets die before it sweeps.
"""


def test_several_threads_may_create_and_sweep_instances():
    """
    Two threads sweeping the same dead instance must not fight over the bookkeeping.
    """
    symbol_graph = SymbolGraph()
    errors: List[BaseException] = []
    start = threading.Barrier(parties=SWEEPING_THREADS)

    def create_and_sweep() -> None:
        start.wait(timeout=SWEEP_TIMEOUT)
        try:
            for _ in range(SWEEPS_PER_THREAD):
                for _ in range(INSTANCES_PER_SWEEP):
                    KRROODPosition(1, 2, 3)
                symbol_graph.remove_dead_instances()
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=create_and_sweep, daemon=True)
        for _ in range(SWEEPING_THREADS)
    ]
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=SWEEP_TIMEOUT)
    finally:
        sys.setswitchinterval(switch_interval)

    assert errors == []
    assert [
        wrapped_instance
        for wrapped_instance in symbol_graph.wrapped_instances
        if wrapped_instance.instance_type is KRROODPosition
    ] == []
