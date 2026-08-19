from __future__ import division

import inspect
import logging
import os
import sys
from functools import cached_property
from typing import Type, Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_all_classes_in_module(
    module_name: str, parent_class: Optional[Type] = None
) -> Dict[str, Type]:
    """
    :param module_name: e.g. giskardpy.goals
    :param parent_class: e.g. Goal
    :return:
    """
    classes = {}
    module = __import__(module_name, fromlist="dummy")
    for class_name, class_type in inspect.getmembers(module, inspect.isclass):
        if (
            parent_class is None
            or issubclass(class_type, parent_class)
            and module_name in str(class_type)
        ):
            classes[class_name] = class_type
    return classes


def create_path(path):
    """
    Creates the directory a file path points into, if any.

    A bare file name has no directory part and needs nothing created; ``exist_ok``
    guards against a concurrent creation of the same directory.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def clear_cached_properties(instance: Any):
    """
    Clears the cache of all cached_property attributes of an instance.

    Args:
        instance: The instance for which to clear all cached_property caches.
    """
    for attr in dir(instance):
        if isinstance(getattr(type(instance), attr, None), cached_property):
            if attr in instance.__dict__:
                del instance.__dict__[attr]


def string_shortener(original_str: str, max_lines: int, max_line_length: int) -> str:
    if len(original_str) < max_line_length:
        return original_str
    lines = []
    start = 0
    for _ in range(max_lines):
        end = start + max_line_length
        lines.append(original_str[start:end])
        if end >= len(original_str):
            break
        start = end

    result = "\n".join(lines)

    # Check if string is cut off and add "..."
    if len(original_str) > start:
        result = result + "..."

    return result


def is_running_in_pytest():
    return "pytest" in sys.modules
