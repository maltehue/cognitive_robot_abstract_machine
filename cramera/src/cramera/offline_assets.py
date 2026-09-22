"""Validate the local dependency graph of a portable recorded scene."""

from __future__ import annotations

import json
import re
import shlex
import struct
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

from typing_extensions import Any, ClassVar

from cramera.mesh_format import MeshFormat
from cramera.onboard.bundle_urdf import BundledAssets

# %% scene references and failures


class OfflineAssetProblem(StrEnum):
    """Reasons a recording cannot supply an asset independently."""

    MISSING = "Referenced file is missing"
    """A required file does not exist within the scene."""
    EXTERNAL = "Reference requires an external location"
    """A URL or absolute path is not portable with the scene."""
    OUTSIDE_BUNDLE = "Reference escapes the scene directory"
    """A relative reference or symbolic link leaves the exported scene."""
    DECLARED_MISSING = "The scene declares unresolved assets"
    """The original bundler reported assets it could not include."""
    INVALID_REFERENCE = "Reference does not name an asset"
    """A malformed location or material declaration cannot be resolved."""


@dataclass
class OfflineAssetError(ValueError):
    """A referenced recording dependency cannot be supplied offline."""

    source: Path
    """File declaring the failed reference."""

    reference: str
    """Asset location as declared in the source file."""

    problem: OfflineAssetProblem
    """Reason the reference cannot be used offline."""

    def __str__(self) -> str:
        """Identify the declaration and the unavailable asset."""
        return f"{self.problem}: {self.source}: {unquote(self.reference)!r}"


class SceneAssetField(StrEnum):
    """Scene bundle fields carrying file references."""

    SCENE = "scene.json"
    """Manifest defining the recorded scene."""
    MODELS = "models"
    """Robot and environment descriptions."""
    URDF = "urdf"
    """Model description reference."""
    OBJECTS = "objects"
    """Loose object descriptions."""
    MESH = "mesh"
    """Loose object geometry reference."""
    MATERIAL = "mtl"
    """Explicit object material library."""
    TRAJECTORY = "trajectory"
    """Recorded world states."""
    STATECHARTS = "statecharts"
    """Optional recorded controller states."""
    MISSING = "missingAssets"
    """Unresolved references reported during recording."""


class AssetFormat(StrEnum):
    """Dependency-bearing asset formats beyond mesh formats."""

    URDF = ".urdf"
    """Robot or environment description."""
    MATERIAL = ".mtl"
    """Wavefront material library."""
    GLTF = ".gltf"
    """JSON glTF mesh with optional external buffers and images."""


class AssetField(StrEnum):
    """File-reference fields in model formats."""

    URI = "uri"
    """glTF file or embedded resource location."""
    BUFFERS = "buffers"
    """glTF binary payload descriptions."""
    IMAGES = "images"
    """glTF image descriptions."""
    FILENAME = "filename"
    """URDF mesh or texture location."""
    MESH = "mesh"
    """URDF geometry element."""
    TEXTURE = "texture"
    """URDF image element."""


class MaterialDirective(StrEnum):
    """Wavefront declarations that refer to other files."""

    LIBRARY = "mtllib"
    """OBJ material-library declaration."""
    MAP_PREFIX = "map_"
    """Prefix identifying material texture maps."""
    BUMP = "bump"
    """Bump image."""
    NORMAL = "norm"
    """Normal image."""
    DISPLACEMENT = "disp"
    """Displacement image."""
    DECAL = "decal"
    """Decal image."""
    REFLECTION = "refl"
    """Reflection image."""


class TextureOption(StrEnum):
    """Wavefront options preceding a texture filename."""

    BLEND_HORIZONTAL = "-blendu"
    """Horizontal blending switch."""
    BLEND_VERTICAL = "-blendv"
    """Vertical blending switch."""
    COLOR_CORRECTION = "-cc"
    """Color-correction switch."""
    CLAMP = "-clamp"
    """Texture clamping switch."""
    BOOST = "-boost"
    """Mipmap sharpness value."""
    RESOLUTION = "-texres"
    """Texture resolution."""
    BUMP_MULTIPLIER = "-bm"
    """Bump multiplier."""
    CHANNEL = "-imfchan"
    """Image channel."""
    TYPE = "-type"
    """Reflection mapping type."""
    COLOR_SPACE = "-colorspace"
    """Image color space."""
    RANGE = "-mm"
    """Base and gain pair."""
    SCALE = "-s"
    """One to three scale coordinates."""
    OFFSET = "-o"
    """One to three offset coordinates."""
    TURBULENCE = "-t"
    """One to three turbulence coordinates."""


@dataclass
class AssetReference:
    """An asset location together with the file declaring it."""

    source: Path
    """File relative to which the asset is located."""
    location: str
    """Declared file URL or relative path."""


# %% dependency declarations


@dataclass
class AssetDependencies:
    """Read the file references declared by a model, mesh or material."""

    path: Path
    """Local asset whose dependencies are inspected."""

    MATERIAL_OPTIONS: ClassVar[dict[TextureOption, int]] = {
        TextureOption.BLEND_HORIZONTAL: 1,
        TextureOption.BLEND_VERTICAL: 1,
        TextureOption.COLOR_CORRECTION: 1,
        TextureOption.CLAMP: 1,
        TextureOption.BOOST: 1,
        TextureOption.RESOLUTION: 1,
        TextureOption.BUMP_MULTIPLIER: 1,
        TextureOption.CHANNEL: 1,
        TextureOption.TYPE: 1,
        TextureOption.COLOR_SPACE: 1,
        TextureOption.RANGE: 2,
    }
    """Fixed argument counts defined for Wavefront texture options."""

    VECTOR_OPTIONS: ClassVar[set[TextureOption]] = {
        TextureOption.SCALE,
        TextureOption.OFFSET,
        TextureOption.TURBULENCE,
    }
    """Texture options taking one to three numeric coordinates."""

    TEXTURE_DIRECTIVES: ClassVar[set[MaterialDirective]] = {
        MaterialDirective.BUMP,
        MaterialDirective.NORMAL,
        MaterialDirective.DISPLACEMENT,
        MaterialDirective.DECAL,
        MaterialDirective.REFLECTION,
    }
    """Texture directives whose names do not start with ``map_``."""

    NUMBER: ClassVar[re.Pattern[str]] = re.compile(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    )
    """Numeric arguments accepted by Wavefront vector options."""

    GLB_HEADER: ClassVar[struct.Struct] = struct.Struct("<4sII")
    """glTF binary header containing magic, version and file length."""

    GLB_CHUNK: ClassVar[struct.Struct] = struct.Struct("<II")
    """glTF chunk header containing payload length and type."""

    GLB_MAGIC: ClassVar[bytes] = b"glTF"
    """Identifier of a binary glTF file."""

    GLB_JSON: ClassVar[int] = 0x4E4F534A
    """Chunk type containing the glTF JSON document."""

    COLLADA_IMAGES: ClassVar[str] = ".//{*}library_images/{*}image//{*}init_from"
    """Image declarations, excluding effect parameters referring to image IDs."""

    def references(self) -> list[str]:
        """Return direct file references while leaving embedded assets in place."""
        suffix = self.path.suffix.lower()
        if suffix == AssetFormat.URDF:
            return self.urdf_references()
        if suffix == MeshFormat.DAE:
            return self.collada_references()
        if suffix == MeshFormat.OBJ:
            return self.object_references()
        if suffix == AssetFormat.MATERIAL:
            return self.material_references()
        if suffix in (AssetFormat.GLTF, MeshFormat.GLB):
            document = self.gltf_document()
            return [
                entry[AssetField.URI]
                for category in (AssetField.BUFFERS, AssetField.IMAGES)
                for entry in document.get(category, [])
                if AssetField.URI in entry
            ]
        return []

    def urdf_references(self) -> list[str]:
        """Read geometry and image filenames from an XML model description."""
        root = ElementTree.parse(self.path).getroot()
        return [
            element.attrib[AssetField.FILENAME]
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] in (AssetField.MESH, AssetField.TEXTURE)
            and AssetField.FILENAME in element.attrib
        ]

    def collada_references(self) -> list[str]:
        """Read image locations without interpreting effect IDs as filenames."""
        root = ElementTree.parse(self.path).getroot()
        return [
            "".join(element.itertext()).strip()
            for element in root.findall(self.COLLADA_IMAGES)
        ]

    def object_references(self) -> list[str]:
        """Read quoted or multiple material libraries declared by OBJ geometry."""
        references = []
        for line in self.path.read_text().splitlines():
            match = BundledAssets.MATERIAL_LIBRARY_PATTERN.fullmatch(line.strip())
            if match is not None:
                libraries = shlex.split(match.group(1), comments=True)
                if not libraries:
                    raise OfflineAssetError(
                        self.path, line, OfflineAssetProblem.INVALID_REFERENCE
                    )
                references.extend(libraries)
            elif line.strip() == MaterialDirective.LIBRARY:
                raise OfflineAssetError(
                    self.path, line, OfflineAssetProblem.INVALID_REFERENCE
                )
        return references

    def material_references(self) -> list[str]:
        """Read texture filenames following Wavefront material-map options."""
        references = []
        for line in self.path.read_text().splitlines():
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            match = BundledAssets.TEXTURE_MAP_PATTERN.fullmatch(line.strip())
            if (
                match is not None
                or tokens[0].lower() in self.TEXTURE_DIRECTIVES
                or tokens[0].lower().startswith(MaterialDirective.MAP_PREFIX)
            ):
                references.append(self.texture_reference(tokens[1:]))
        return references

    def texture_reference(self, tokens: list[str]) -> str:
        """Separate a texture filename from its mapping options.

        :param tokens: Tokens after a Wavefront texture directive.
        :return: Filename, including any unquoted spaces.
        """
        index = 0
        while index < len(tokens) and tokens[index].startswith("-"):
            option = tokens[index].lower()
            index += 1
            if option in self.VECTOR_OPTIONS:
                first = index
                while index < min(first + 3, len(tokens)) and self.NUMBER.fullmatch(
                    tokens[index]
                ):
                    index += 1
                if index == first:
                    raise OfflineAssetError(
                        self.path,
                        " ".join(tokens),
                        OfflineAssetProblem.INVALID_REFERENCE,
                    )
            elif option in self.MATERIAL_OPTIONS:
                index += self.MATERIAL_OPTIONS[option]
            else:
                raise OfflineAssetError(
                    self.path, " ".join(tokens), OfflineAssetProblem.INVALID_REFERENCE
                )
        if index >= len(tokens):
            raise OfflineAssetError(
                self.path, " ".join(tokens), OfflineAssetProblem.INVALID_REFERENCE
            )
        return " ".join(tokens[index:])

    def gltf_document(self) -> dict[str, Any]:
        """Read a glTF JSON document from a text file or binary container."""
        if self.path.suffix.lower() == AssetFormat.GLTF:
            return json.loads(self.path.read_text())
        data = self.path.read_bytes()
        magic, version, length = self.GLB_HEADER.unpack_from(data)
        if magic != self.GLB_MAGIC or version != 2 or length != len(data):
            raise OfflineAssetError(
                self.path, self.path.name, OfflineAssetProblem.INVALID_REFERENCE
            )
        offset = self.GLB_HEADER.size
        while offset < len(data):
            length, kind = self.GLB_CHUNK.unpack_from(data, offset)
            offset += self.GLB_CHUNK.size
            if offset + length > len(data):
                raise OfflineAssetError(
                    self.path, self.path.name, OfflineAssetProblem.INVALID_REFERENCE
                )
            if kind == self.GLB_JSON:
                return json.loads(data[offset : offset + length])
            offset += length
        raise OfflineAssetError(
            self.path, self.path.name, OfflineAssetProblem.INVALID_REFERENCE
        )


# %% portable scene validation


@dataclass
class OfflineSceneAssets:
    """Verify that a recorded scene supplies its complete local asset graph."""

    directory: Path
    """Scene directory that must contain every required file."""

    def resolve_reference(self, source: Path, reference: str) -> Path | None:
        """Resolve a portable reference or accept an embedded glTF resource.

        :param source: File declaring the reference.
        :param reference: Relative URL or embedded glTF data URI.
        :return: Existing local file, or None for embedded data.
        :raises OfflineAssetError: If the reference cannot travel with the scene.
        """
        parsed = urlsplit(reference.strip())
        if parsed.scheme == "data" and source.suffix.lower() in (
            AssetFormat.GLTF,
            MeshFormat.GLB,
        ):
            return None
        location = unquote(parsed.path).replace("\\", "/")
        if (
            parsed.scheme
            or parsed.netloc
            or location.startswith("/")
            or PureWindowsPath(location).drive
        ):
            raise OfflineAssetError(source, reference, OfflineAssetProblem.EXTERNAL)
        if not location or "\x00" in location:
            raise OfflineAssetError(
                source, reference, OfflineAssetProblem.INVALID_REFERENCE
            )
        path = (source.parent / location).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise OfflineAssetError(
                source, reference, OfflineAssetProblem.OUTSIDE_BUNDLE
            )
        if not path.is_file():
            raise OfflineAssetError(source, reference, OfflineAssetProblem.MISSING)
        return path

    def validate(self) -> None:
        """Check every declared asset and its transitive local dependencies.

        :raises OfflineAssetError: If any required asset is unavailable offline.
        """
        manifest = self.directory / SceneAssetField.SCENE
        manifest = self.resolve_reference(manifest, SceneAssetField.SCENE)
        assert manifest is not None
        document = json.loads(manifest.read_text())
        missing = document.get(SceneAssetField.MISSING, [])
        if missing:
            raise OfflineAssetError(
                manifest, str(missing[0]), OfflineAssetProblem.DECLARED_MISSING
            )
        references = [document[SceneAssetField.TRAJECTORY]]
        if SceneAssetField.STATECHARTS in document:
            references.append(document[SceneAssetField.STATECHARTS])
        references.extend(
            model[SceneAssetField.URDF]
            for model in document.get(SceneAssetField.MODELS, [])
        )
        references.extend(
            item[key]
            for item in document.get(SceneAssetField.OBJECTS, [])
            for key in (SceneAssetField.MESH, SceneAssetField.MATERIAL)
            if key in item
        )
        pending = [AssetReference(manifest, reference) for reference in references]
        visited: set[Path] = set()
        while pending:
            reference = pending.pop()
            path = self.resolve_reference(reference.source, reference.location)
            if path is None or path in visited:
                continue
            visited.add(path)
            pending.extend(
                AssetReference(path, dependency)
                for dependency in AssetDependencies(path).references()
            )
