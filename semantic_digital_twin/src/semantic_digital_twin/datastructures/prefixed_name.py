from dataclasses import dataclass

from typing_extensions import Optional, Self

from krrood.entity_query_language.predicate import Symbol

NAME_SEPARATOR = "/"
"""
What stands between a prefix and the name it scopes when a name is written as one string.
"""


@dataclass
class PrefixedName(Symbol):
    name: str
    """
    The local name identifying the entity.
    """

    prefix: Optional[str] = None
    """
    Optional namespace that disambiguates the name from equally named entities in other scopes.
    """

    @classmethod
    def from_string(cls, text: str) -> Self:
        """
        The name a string spells, split at its last :data:`NAME_SEPARATOR`.

        This reads back what :meth:`__str__` writes, so a text without a separator names
        an entity without a prefix.

        :param text: The written name.
        """
        prefix, separator, name = text.rpartition(NAME_SEPARATOR)
        if separator == "":
            return cls(name=text)
        return cls(name=name, prefix=prefix)

    def __hash__(self):
        return hash((self.prefix, self.name))

    def __str__(self):
        if self.prefix is None or self.prefix == "":
            return self.name
        return f"{self.prefix}{NAME_SEPARATOR}{self.name}"

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.prefix}{NAME_SEPARATOR}{self.name}')"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.prefix == other.prefix and self.name == other.name

    def __lt__(self, other):
        return str(self) < str(other)

    def __le__(self, other):
        return str(self) <= str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __ge__(self, other):
        return str(self) >= str(other)
