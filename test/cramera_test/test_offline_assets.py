"""
Recorded scenes retain every referenced model and material without a network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cramera.offline_assets import (
    OfflineAssetError,
    OfflineAssetProblem,
    OfflineSceneAssets,
)

# %% recorded asset fixture


@pytest.fixture
def offline_scene(tmp_path: Path) -> Path:
    """
    Copy a recorded scene with nested material and texture references.

    :param tmp_path: Isolated directory for the scene under validation.
    :return: Root of the copied scene bundle.
    """
    source = Path(__file__).parent / "dataset" / "offline_assets"
    return Path(shutil.copytree(source, tmp_path / "scene"))


def scene_with_reference(directory: Path, reference: str) -> None:
    """
    Replace the first object mesh reference while preserving the scene schema.

    :param directory: Scene bundle to adjust for a regression.
    :param reference: Mesh location declared by that scene.
    """
    path = directory / "scene.json"
    data = json.loads(path.read_text())
    data["objects"][0]["mesh"] = reference
    path.write_text(json.dumps(data))


# %% complete dependency traversal


def test_complete_scene_is_self_contained(offline_scene: Path) -> None:
    """
    Nested materials, encoded spaces and embedded glTF assets remain local.

    :param offline_scene: Complete scene with image and material identifiers.
    """
    assert OfflineSceneAssets(offline_scene).validate() is None


@pytest.mark.parametrize(
    "reference",
    [
        "scene.json",
        "robot.urdf",
        "motion data.json",
        "states.json",
        "meshes/part.dae",
        "meshes/object.obj",
        "meshes/materials/surface.mtl",
        "meshes/materials/second material.mtl",
        "textures/paint coat.svg",
        "meshes/geometry.bin",
    ],
)
def test_missing_referenced_file_is_rejected(
    offline_scene: Path, reference: str
) -> None:
    """
    Every transitively required file is checked, including texture side assets.

    :param offline_scene: Scene whose dependency is removed.
    :param reference: Relative path of the required file.
    """
    missing = offline_scene / reference
    missing.unlink()
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.MISSING
    assert missing.name in str(failure.value)


def test_unreferenced_broken_files_are_ignored(offline_scene: Path) -> None:
    """
    An unrelated malformed asset cannot invalidate the recorded dependency graph.

    :param offline_scene: Complete scene receiving an unreferenced file.
    """
    (offline_scene / "unused.gltf").touch()
    assert OfflineSceneAssets(offline_scene).validate() is None


def test_optional_statecharts_can_be_absent(offline_scene: Path) -> None:
    """
    Recordings without controller snapshots require no statechart file.

    :param offline_scene: Scene whose optional snapshot reference is removed.
    """
    path = offline_scene / "scene.json"
    data = json.loads(path.read_text())
    del data["statecharts"]
    path.write_text(json.dumps(data))
    (offline_scene / "states.json").unlink()
    assert OfflineSceneAssets(offline_scene).validate() is None


def test_declared_missing_assets_are_rejected(offline_scene: Path) -> None:
    """
    A bundler's unresolved references must remain an explicit validation failure.

    :param offline_scene: Scene with an unresolved original mesh.
    """
    path = offline_scene / "scene.json"
    data = json.loads(path.read_text())
    data["missingAssets"] = ["package://missing/robot.stl"]
    path.write_text(json.dumps(data))
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.DECLARED_MISSING


# %% portable references


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.com/mesh.obj",
        "//example.com/mesh.obj",
        "file:///tmp/mesh.obj",
        "/tmp/mesh.obj",
        "package://robot/mesh.obj",
        "C:\\models\\mesh.obj",
        "%2F%2Fexample.com/mesh.obj",
    ],
)
def test_external_locations_are_rejected(offline_scene: Path, reference: str) -> None:
    """
    A portable recording cannot rely on network or machine-specific asset paths.

    :param offline_scene: Scene whose object location is replaced.
    :param reference: External asset reference to reject.
    """
    scene_with_reference(offline_scene, reference)
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.EXTERNAL


@pytest.mark.parametrize(
    "reference", ["../outside.obj", "%2e%2e/outside.obj", "meshes/../../outside.obj"]
)
def test_escaping_paths_are_rejected(offline_scene: Path, reference: str) -> None:
    """
    Decoded relative references must remain beneath the exported scene root.

    :param offline_scene: Scene containing the escaping reference.
    :param reference: Relative reference leaving the bundle.
    """
    scene_with_reference(offline_scene, reference)
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.OUTSIDE_BUNDLE


def test_symlink_outside_bundle_is_rejected(offline_scene: Path) -> None:
    """
    A symbolic link cannot make an external asset look bundled.

    :param offline_scene: Scene receiving an escaping symbolic link.
    """
    external = offline_scene.parent / "outside.obj"
    external.touch()
    linked = offline_scene / "linked.obj"
    linked.symlink_to(external)
    scene_with_reference(offline_scene, linked.name)
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.OUTSIDE_BUNDLE


@pytest.mark.parametrize("reference", ["", "#material", "meshes/%00bad.obj"])
def test_invalid_file_references_are_rejected(
    offline_scene: Path, reference: str
) -> None:
    """
    Empty, fragment-only and null-byte locations cannot name a portable file.

    :param offline_scene: Scene whose mesh reference is invalid.
    :param reference: Unusable file location.
    """
    scene_with_reference(offline_scene, reference)
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.INVALID_REFERENCE


# %% material and binary formats


@pytest.mark.parametrize(
    "reference",
    [
        "materials/unknown_option.mtl",
        "materials/empty_vector.mtl",
        "materials/missing_filename.mtl",
        "materials/empty_directive.mtl",
        "empty_library.obj",
    ],
)
def test_incomplete_material_declarations_are_rejected(
    offline_scene: Path, reference: str
) -> None:
    """
    A missing texture filename cannot silently pass offline validation.

    :param offline_scene: Bundle containing isolated invalid material fixtures.
    :param reference: Invalid material or mesh dependency to inspect.
    """
    scene_with_reference(offline_scene, f"meshes/{reference}")
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.INVALID_REFERENCE


def test_external_texture_is_rejected_transitively(offline_scene: Path) -> None:
    """
    Material dependencies obey the same locality rules as top-level meshes.

    :param offline_scene: Bundle containing a material with an external image.
    """
    scene_with_reference(offline_scene, "meshes/materials/external_texture.mtl")
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.EXTERNAL
    assert failure.value.source.name == "external_texture.mtl"


def test_binary_gltf_dependencies_are_checked(offline_scene: Path) -> None:
    """
    A GLB's JSON chunk may still reference local images and buffers.

    :param offline_scene: Bundle containing equivalent JSON and binary glTF assets.
    """
    scene_with_reference(offline_scene, "meshes/object.glb")
    assert OfflineSceneAssets(offline_scene).validate() is None


@pytest.mark.parametrize(
    "filename", ["invalid_header.glb", "invalid_chunk.glb", "no_json.glb"]
)
def test_invalid_binary_gltf_is_rejected(offline_scene: Path, filename: str) -> None:
    """
    Invalid binary containers cannot hide uninspected external dependencies.

    :param offline_scene: Bundle receiving a malformed GLB reference.
    :param filename: Malformed binary fixture within the meshes directory.
    """
    scene_with_reference(offline_scene, f"meshes/{filename}")
    with pytest.raises(OfflineAssetError) as failure:
        OfflineSceneAssets(offline_scene).validate()
    assert failure.value.problem is OfflineAssetProblem.INVALID_REFERENCE
