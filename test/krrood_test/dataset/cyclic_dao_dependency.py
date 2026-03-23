from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from krrood.ormatic.dao import AlternativeMapping


@dataclass(eq=False)
class IssueDependency:
    """
    A domain class whose hash depends on a relationship field.
    """

    name: str
    parent: Optional[IssueMain]

    def __hash__(self):
        # This will fail with AttributeError if parent is not set,
        # which happens during the discovery phase of ORMatic
        # for AlternativeMappings if we call hash() in to_domain_object.
        return hash((self.name, self.parent))


@dataclass
class IssueMain:
    """
    A domain class that uses an AlternativeMapping.
    """

    name: str
    dependencies: List[IssueDependency] = field(default_factory=list)


@dataclass
class PlanReproduction:
    """
    A class that discovers IssueDependency before IssueMain.
    """

    dependency: IssueDependency


@dataclass(eq=False)
class IssueMainMapping(AlternativeMapping[IssueMain]):
    """
    Mapping that triggers the bug by calling hash() on its dependencies
    during to_domain_object().
    """

    name: str
    dependencies: List[IssueDependency]

    @classmethod
    def from_domain_object(cls, obj: IssueMain):
        return cls(name=obj.name, dependencies=obj.dependencies)

    def to_domain_object(self) -> IssueMain:
        # Triggering hash() on dependencies that are not yet filled
        for dep in self.dependencies:
            hash(dep)
        result = IssueMain(name=self.name, dependencies=self.dependencies)
        for dep in self.dependencies:
            dep.parent = result
        return result
