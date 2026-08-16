"""
Collection settings for the motion-statechart tests.
"""

import importlib.util

collect_ignore = []

# ``test_pouring_learned`` calls pytest.importorskip at module scope. Raising Skipped while that
# module is imported aborts collection for everything pytest was asked to collect alongside it, so
# running this directory would report zero collected tests and still exit zero — a run that silently
# tests nothing. Excluding the module when its dependencies are absent keeps the rest collectable.
if (
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("l4casadi") is None
):
    collect_ignore.append("test_pouring_learned.py")
