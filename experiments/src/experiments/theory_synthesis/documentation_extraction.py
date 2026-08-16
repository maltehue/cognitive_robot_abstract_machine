"""
Reads the documentation this repository keeps below each dataclass field.

Field docstrings are string literals placed under the field they document, which Python
does not retain at runtime — but the source does. Extracting them lets a prompt
verbalize an interface from its single source of truth, so the vocabulary a synthesizer
is offered can never drift from the code it must target.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from typing_extensions import Dict, Type


def field_documentation(dataclass_type: Type) -> Dict[str, str]:
    """
    The docstring below each field of a dataclass, plus each property's docstring.

    :param dataclass_type: The dataclass whose fields are documented.
    :return: First docstring line by field name, across the class hierarchy.
    """
    documentation: Dict[str, str] = {}
    for klass in reversed(dataclass_type.__mro__):
        if klass is object:
            continue
        documentation.update(_own_field_documentation(klass))
        for attribute_name, attribute in vars(klass).items():
            if isinstance(attribute, property) and attribute.__doc__:
                documentation[attribute_name] = _first_line(attribute.__doc__)
    return documentation


def _own_field_documentation(klass: Type) -> Dict[str, str]:
    """
    The docstrings below the fields one class defines itself.

    :param klass: The class whose source is read.
    :return: First docstring line by field name; empty if the source is unavailable.
    """
    try:
        source = textwrap.dedent(inspect.getsource(klass))
    except (OSError, TypeError):
        return {}
    [class_definition] = ast.parse(source).body
    documentation: Dict[str, str] = {}
    body = class_definition.body
    for statement, following in zip(body, body[1:]):
        is_documented_field = (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        )
        if is_documented_field:
            documentation[statement.target.id] = _first_line(following.value.value)
    return documentation


def _first_line(docstring: str) -> str:
    """
    The first non-empty line of a docstring, as the one-line summary.

    :param docstring: The full docstring.
    :return: Its first non-empty line.
    """
    for line in docstring.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""
