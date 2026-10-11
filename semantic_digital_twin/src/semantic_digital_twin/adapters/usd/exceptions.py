from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

from semantic_digital_twin.exceptions import ParsingError


@dataclass
class UnsupportedUsdPhysicsJointTypeError(ParsingError):
    """
    Raised when a stage contains a physics joint of a type this parser does not build a
    Connection for.

    Silently skipping it would build a World missing whatever link that joint's body1 is
    the only connection to.
    """

    joint_path: str = field(kw_only=True)
    """
    The unsupported joint prim's stage path.
    """

    joint_type: str = field(kw_only=True)
    """
    The unsupported joint prim's USD type name.
    """

    supported_types: List[str] = field(kw_only=True, default_factory=list)
    """
    The joint types that can be mapped to connections.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a joint of unsupported type "
            f"'{self.joint_type}' at '{self.joint_path}'."
        )

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


@dataclass
class UsdPhysicsJointMissingChildBodyError(ParsingError):
    """
    Raised when a physics joint's ``body1`` relationship (its child link) has no target.

    Unlike ``body0``, where an unset target is the USD convention for "the stage's own
    root frame", ``body1`` has no such meaning here - every joint is expected to connect
    a link into the world, so a joint with no child leaves nothing to build a Connection
    to.
    """

    joint_path: str = field(kw_only=True)
    """
    The joint prim's stage path.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a joint at '{self.joint_path}' with no "
            f"body1 target."
        )

    def suggest_correction(self) -> str:
        return "Inspect the joint prim's body1 relationship on the stage."


@dataclass
class UnsupportedUsdGeometryTypeError(ParsingError):
    """
    Raised when a stage contains a renderable geometric primitive
    (:class:`~pxr.UsdGeom.Gprim`) of a type this parser does not build a Shape for.

    Silently skipping it would build a World missing that piece of a link's geometry,
    the same way silently skipping an unsupported joint would build one missing a link.
    """

    prim_path: str = field(kw_only=True)
    """
    The unsupported geometry prim's stage path.
    """

    geometry_type: str = field(kw_only=True)
    """
    The unsupported geometry prim's USD type name.
    """

    supported_types: List[str] = field(kw_only=True, default_factory=list)
    """
    The geometry types that can be mapped to shapes.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has geometry of unsupported type "
            f"'{self.geometry_type}' at '{self.prim_path}'."
        )

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


# %% scene topology


@dataclass
class UsdSceneTopologyError(ParsingError):
    """
    Raised when scene relationships cannot form a supported connection tree.
    """

    prim_path: str = field(kw_only=True)
    """
    The prim whose relationship cannot be represented.
    """

    def error_message(self) -> str:
        """
        Identify the unsupported scene relationship.
        """
        return f"Stage '{self.file_path}' has unsupported scene topology at '{self.prim_path}'."

    def suggest_correction(self) -> str:
        """
        Describe the supported scene structure.
        """
        return "Use valid single-target joints forming a tree, without closed loops or multiple parents."


@dataclass
class UsdScaledJointFrameError(ParsingError):
    """
    Raised when a joint endpoint has scale or shear that its anchors cannot represent.
    """

    prim_path: str = field(kw_only=True)
    """
    The joint endpoint with a non-rigid world transform.
    """

    def error_message(self) -> str:
        """
        Identify the unsupported joint frame.
        """
        return f"Stage '{self.file_path}' has a scaled or sheared joint frame at '{self.prim_path}'."

    def suggest_correction(self) -> str:
        """
        Describe how to make joint anchors representable.
        """
        return "Bake scale into geometry and author joints in unscaled rigid frames."


# %% inferred mass properties


@dataclass
class UsdInertialGeometryError(ParsingError):
    """
    Raised when unspecified mass properties lack closed collision geometry.
    """

    prim_path: str = field(kw_only=True)
    """
    Prim requiring geometric mass-property inference.
    """

    def error_message(self) -> str:
        """
        Identify the prim whose mass properties cannot be inferred.
        """
        return f"Cannot infer unspecified mass properties at '{self.prim_path}' in '{self.file_path}'."

    def suggest_correction(self) -> str:
        """
        Describe valid authored or geometric mass properties.
        """
        return "Author a finite centre and nonzero inertia, or provide closed collision geometry with positive volume."
