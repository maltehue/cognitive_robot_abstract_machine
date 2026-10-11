from __future__ import annotations

import math
from pathlib import Path

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

import semantic_digital_twin.adapters.usd.exceptions as usd_exceptions

import semantic_digital_twin.adapters.usd.scene_parser as usd_parser

import numpy as np
import pytest

from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from PIL import Image

from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedUsdGeometryTypeError,
    UnsupportedUsdPhysicsJointTypeError,
    UsdPhysicsJointMissingChildBodyError,
)
from semantic_digital_twin.adapters.usd.parser import (
    USDParser,
    UsdMeshShapeBuilder,
    UsdAxis,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.adapters.usd.semantics import UsdSceneSemantics
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor, Table
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdSemanticLabels
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Sphere,
    Scale,
)
from semantic_digital_twin.world_description.world_entity import Body

from .usd_stages import (
    SceneLabel,
    SceneTaxonomy,
    build_labelled_scene,
    SceneJointType,
    CollisionScenePath,
    CollisionSceneFile,
    build_collision_scene,
    write_composed_collision_scene,
    MixedScenePath,
    ComponentScenePath,
    build_mixed_scene,
    build_component_scene,
    ScenePrimPath,
    build_stage_with_placed_objects,
    build_stage_with_scaled_root,
    PXR_AVAILABLE,
    USD_SEMANTICS_AVAILABLE,
    build_jointless_stage_with_a_default_prim,
    build_jointless_stage_with_multiple_top_level_prims,
    build_jointless_stage_with_unsupported_geometry,
    build_single_joint_stage,
    build_single_joint_stage_with_mass,
    build_single_joint_stage_with_semantic_labels,
    build_stage_with_ambiguous_root_and_a_joint,
    build_stage_with_joint_missing_body1,
    build_stage_with_mesh_targeted_body0,
    build_stage_with_primitive_shapes,
    build_stage_with_scaled_mesh,
    build_stage_with_textured_mesh,
)

if PXR_AVAILABLE:
    from pxr import Gf, Kind, Usd, UsdGeom, UsdPhysics

if USD_SEMANTICS_AVAILABLE:
    from pxr import UsdSemantics

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)


def parse(stage) -> World:
    return USDParser(stage=stage, prefix="object").parse()


# %% joints


def test_parse_connects_a_revolute_joint():
    stage = build_single_joint_stage("RevoluteJoint")
    world = parse(stage)

    assert len(world.bodies) == 2  # root + child
    [connection] = world.connections
    assert isinstance(connection, RevoluteConnection)


def test_parse_connects_a_prismatic_joint():
    stage = build_single_joint_stage("PrismaticJoint")
    world = parse(stage)

    [connection] = world.connections
    assert isinstance(connection, PrismaticConnection)


def test_parse_connects_a_fixed_joint():
    stage = build_single_joint_stage("FixedJoint")
    world = parse(stage)

    [connection] = world.connections
    assert isinstance(connection, FixedConnection)


def test_parse_inverts_the_joints_local_pos1_into_connection_t_child():
    # UsdPhysics.Joint documents localPos1/localRot1 as the joint frame's pose relative
    # to body1 (child_T_connection) - the inverse of what Connection needs
    # (connection_T_child). A pure translation makes a missing inversion visible as a
    # sign flip: authoring local_pos1=(1, 0, 0) must place the child's own origin at
    # (-1, 0, 0) relative to the connection frame, not (1, 0, 0).
    stage = build_single_joint_stage("FixedJoint", local_pos1=(1.0, 0.0, 0.0))
    world = parse(stage)

    [connection] = world.connections
    translation = connection.connection_T_child_expression.to_np()[:3, 3]
    np.testing.assert_allclose(translation, [-1.0, 0.0, 0.0], atol=1e-6)


def test_parse_raises_on_an_unsupported_joint_type_instead_of_building_an_incomplete_world():
    # An unrecognized joint type name must not be silently skipped like a non-joint
    # prim would be, or a stage using only unsupported joints builds a World with no
    # bodies but the root - not an error, just quietly missing every link.
    stage = build_single_joint_stage("SphericalJoint")

    with pytest.raises(UnsupportedUsdPhysicsJointTypeError) as excinfo:
        parse(stage)

    assert excinfo.value.joint_type == "PhysicsSphericalJoint"


def test_parse_swaps_inverted_revolute_limits():
    # Seen on a real ArtVIP object (one blade of a pair of scissors): the USD data
    # itself authors lower > upper.
    stage = build_single_joint_stage("RevoluteJoint", lower_limit=30.0, upper_limit=0.0)
    world = parse(stage)

    [connection] = world.connections
    assert connection.raw_dof.limits.lower.position == pytest.approx(0.0)
    assert connection.raw_dof.limits.upper.position == pytest.approx(math.radians(30.0))


def test_parse_resolves_a_mesh_targeted_body0_to_its_enclosing_link():
    # body0 pointing directly at a link's mesh prim (rather than its enclosing Xform)
    # must not create a second, disconnected Body for the same physical link.
    stage = build_stage_with_mesh_targeted_body0()
    world = parse(stage)

    assert len(world.bodies) == 3  # root + carcass + handle
    assert world.root is not None


def test_parse_raises_on_a_joint_with_no_body1_target():
    stage = build_stage_with_joint_missing_body1()

    with pytest.raises(UsdPhysicsJointMissingChildBodyError) as excinfo:
        parse(stage)

    assert excinfo.value.joint_path == "/object/joint"


# %% joint-less stages


def test_parse_builds_a_single_body_world_for_a_stage_without_joints():
    stage = build_jointless_stage_with_a_default_prim()
    world = parse(stage)

    assert len(world.bodies) == 1
    assert len(world.connections) == 0
    assert len(world.root.visual.shapes) == 1


def test_parse_connects_a_joint_to_a_synthetic_root_with_an_ambiguous_root_prim():
    # A joint's unset body0 never depended on the stage having an identifiable root
    # prim: it works the same against a synthetic root as against a real one.
    stage = build_stage_with_ambiguous_root_and_a_joint()
    world = parse(stage)

    [connection] = world.connections
    assert isinstance(connection, FixedConnection)
    assert connection.parent is world.root


def test_parse_attaches_multiple_top_level_prims_to_a_synthetic_root():
    # A joint-less stage's ambiguous root is not an error the way one with a joint
    # graph to name is: each top-level prim becomes its own freely posable body instead.
    stage = build_jointless_stage_with_multiple_top_level_prims()
    world = parse(stage)

    assert len(world.bodies) == 3  # synthetic root + object_a + object_b
    non_root_bodies = [body for body in world.bodies if body is not world.root]
    assert {body.name.name for body in non_root_bodies} == {"object_a", "object_b"}
    for body in non_root_bodies:
        assert len(body.visual.shapes) == 1

    [connection_a, connection_b] = world.connections
    assert isinstance(connection_a, Connection6DoF)
    assert isinstance(connection_b, Connection6DoF)
    assert connection_a.parent is world.root
    assert connection_b.parent is world.root


def test_parse_raises_on_unsupported_geometry_instead_of_silently_dropping_it():
    # A Cone prim is a UsdGeom.Gprim this parser does not build a Shape for - it must
    # not be silently skipped like a non-geometry Xform/Scope/Camera prim would be, or
    # a stage using it builds a World quietly missing that piece of geometry.
    stage = build_jointless_stage_with_unsupported_geometry()

    with pytest.raises(UnsupportedUsdGeometryTypeError) as excinfo:
        parse(stage)

    assert excinfo.value.geometry_type == "Cone"


# %% shapes


def test_create_mesh_shape_applies_a_non_uniform_scale():
    stage = build_stage_with_scaled_mesh(scale=(2.0, 3.0, 4.0))
    mesh_prim = stage.GetPrimAtPath("/object/scaled/mesh")

    shape = UsdMeshShapeBuilder(mesh_prim, Gf.Matrix4d(1)).build()

    vertices = shape.unscaled_mesh.vertices
    np.testing.assert_allclose(vertices.max(axis=0), [2.0, 3.0, 0.0], atol=1e-5)


def test_parse_builds_a_box_for_a_cube_prim():
    stage = build_stage_with_primitive_shapes(cube_scale=(1.0, 2.0, 3.0))
    world = parse(stage)

    [box] = [shape for shape in world.root.visual.shapes if isinstance(shape, Box)]
    np.testing.assert_allclose([box.scale.x, box.scale.y, box.scale.z], [2.0, 4.0, 6.0])


def test_parse_builds_a_sphere_for_a_sphere_prim():
    stage = build_stage_with_primitive_shapes()
    world = parse(stage)

    [sphere] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Sphere)
    ]
    assert sphere.radius == pytest.approx(0.5)


def test_parse_builds_a_cylinder_for_a_z_axis_cylinder_prim():
    stage = build_stage_with_primitive_shapes(cylinder_axis="Z")
    world = parse(stage)

    [cylinder] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Cylinder)
    ]
    assert cylinder.width == pytest.approx(1.0)
    assert cylinder.height == pytest.approx(2.0)


def test_parse_aligns_an_x_axis_cylinder_prim_to_the_shapes_local_z_axis():
    stage = build_stage_with_primitive_shapes(cylinder_axis="X")
    world = parse(stage)

    [cylinder] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Cylinder)
    ]
    rotation = cylinder.origin.to_np()[:3, :3]
    local_z_in_link_frame = rotation @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(local_z_in_link_frame, [1.0, 0.0, 0.0], atol=1e-6)


def test_parse_applies_scale_along_the_cylinders_own_axis_not_the_shapes():
    # scale is authored in the prim's own local axes, unaffected by the alignment
    # rotation that reorients an X/Y-axis cylinder to the shape's local Z: for an
    # X-axis cylinder, scale[0] is height and scale[1]/scale[2] are radial, not
    # scale[2]/scale[0:2] as for the default Z-axis case.
    stage = build_stage_with_primitive_shapes(
        cylinder_axis="X", cylinder_scale=(5.0, 1.0, 1.0)
    )
    world = parse(stage)

    [cylinder] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Cylinder)
    ]
    assert cylinder.height == pytest.approx(2.0 * 5.0)
    assert cylinder.width == pytest.approx(1.0)


# %% inertials


def test_parse_inertial_reads_mass_api_properties():
    stage = build_single_joint_stage_with_mass(
        mass=2.0, center_of_mass=(0.1, 0.2, 0.3), diagonal_inertia=(1.0, 2.0, 3.0)
    )
    link_prim = stage.GetPrimAtPath("/object/child")
    body = Body(name=PrefixedName("test_body"))

    inertial = USDParser._parse_inertial(link_prim, body)

    assert inertial is not None
    assert inertial.mass == pytest.approx(2.0)
    np.testing.assert_allclose(
        inertial.center_of_mass.to_np()[:3], [0.1, 0.2, 0.3], atol=1e-6
    )


def test_parse_inertial_is_none_without_mass_api():
    stage = build_single_joint_stage("FixedJoint")
    link_prim = stage.GetPrimAtPath("/object/child")
    body = Body(name=PrefixedName("test_body"))

    assert USDParser._parse_inertial(link_prim, body) is None


def test_parse_sets_body_inertial_from_mass_api():
    stage = build_single_joint_stage_with_mass(mass=2.0)
    world = parse(stage)

    [child] = [body for body in world.bodies if body is not world.root]
    assert child.inertial.mass == pytest.approx(2.0)


# %% semantics


@pytest.mark.skipif(
    not USD_SEMANTICS_AVAILABLE, reason="UsdSemantics not available in this usd-core"
)
def test_read_semantic_labels_reads_every_taxonomy():
    stage = build_single_joint_stage_with_semantic_labels()
    link_prim = stage.GetPrimAtPath("/object/child")

    labels_by_taxonomy = USDParser._read_semantic_labels(link_prim)

    assert labels_by_taxonomy == {
        "class": ["chair", "furniture"],
        "category": ["seating"],
    }


def test_read_semantic_labels_is_empty_without_any_applied():
    stage = build_single_joint_stage("FixedJoint")
    link_prim = stage.GetPrimAtPath("/object/child")

    assert USDParser._read_semantic_labels(link_prim) == {}


@pytest.mark.skipif(
    not USD_SEMANTICS_AVAILABLE, reason="UsdSemantics not available in this usd-core"
)
def test_parse_attaches_one_usd_semantic_labels_annotation_per_taxonomy():
    stage = build_single_joint_stage_with_semantic_labels()
    world = parse(stage)

    [child] = [body for body in world.bodies if body is not world.root]
    annotations = [
        annotation
        for annotation in world.semantic_annotations
        if isinstance(annotation, UsdSemanticLabels)
    ]
    labels_by_taxonomy = {
        annotation.taxonomy: annotation.labels for annotation in annotations
    }
    assert all(annotation.root is child for annotation in annotations)
    assert labels_by_taxonomy == {
        "class": ["chair", "furniture"],
        "category": ["seating"],
    }


def test_parse_attaches_no_annotation_for_an_unlabelled_body():
    stage = build_single_joint_stage("FixedJoint")
    world = parse(stage)

    assert not any(
        isinstance(annotation, UsdSemanticLabels)
        for annotation in world.semantic_annotations
    )


# %% materials


@pytest.fixture
def texture_file(tmp_path):
    path = tmp_path / "wood.png"
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(path)
    return str(path)


def test_diffuse_texture_path_resolves_the_bound_texture(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")

    resolved = UsdMeshShapeBuilder._diffuse_texture_path(mesh_prim)

    assert resolved == texture_file


def test_diffuse_texture_path_is_none_without_a_bound_material(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh_prim = stage.DefinePrim("/object/unbound_mesh", "Mesh")

    resolved = UsdMeshShapeBuilder._diffuse_texture_path(unbound_mesh_prim)

    assert resolved is None


def test_uv_coordinates_reads_the_per_point_st_primvar(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")

    uv = UsdMeshShapeBuilder._uv_coordinates(mesh_prim)

    np.testing.assert_array_equal(uv, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_uv_coordinates_is_none_without_an_st_primvar(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh_prim = stage.DefinePrim("/object/unbound_mesh", "Mesh")

    uv = UsdMeshShapeBuilder._uv_coordinates(unbound_mesh_prim)

    assert uv is None


def test_create_mesh_shape_applies_the_bound_texture(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")

    shape = UsdMeshShapeBuilder(mesh_prim, Gf.Matrix4d(1)).build()

    mesh = shape.unscaled_mesh
    assert mesh.visual.kind == "texture"
    assert mesh.visual.uv is not None


def test_create_mesh_shape_has_no_texture_without_a_bound_material(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh = UsdGeom.Mesh.Define(stage, "/object/unbound_mesh")
    unbound_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    unbound_mesh.CreateFaceVertexCountsAttr([3])
    unbound_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    shape = UsdMeshShapeBuilder(unbound_mesh.GetPrim(), Gf.Matrix4d(1)).build()

    assert shape.unscaled_mesh.visual.kind != "texture"


# %% triangulation


def test_triangulate_keeps_a_triangle_as_is():
    faces = UsdMeshShapeBuilder._triangulate([3], [0, 1, 2])
    np.testing.assert_array_equal(faces, [[0, 1, 2]])


def test_triangulate_fans_a_quad_into_two_triangles():
    faces = UsdMeshShapeBuilder._triangulate([4], [0, 1, 2, 3])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])


def test_triangulate_fans_a_pentagon_into_three_triangles():
    faces = UsdMeshShapeBuilder._triangulate([5], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3], [0, 3, 4]])


def test_triangulate_handles_multiple_faces_of_different_sizes():
    # A triangle followed by a quad, sharing no vertex indices.
    faces = UsdMeshShapeBuilder._triangulate([3, 4], [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 4, 5], [3, 5, 6]])


# %% authored scene placement


@pytest.mark.parametrize("up_axis", [UsdAxis.Y, UsdAxis.Z])
@pytest.mark.parametrize("meters_per_unit", [0.01, 1.0])
def test_parse_preserves_independent_object_poses_in_stage_coordinates(
    up_axis: UsdAxis, meters_per_unit: float
) -> None:
    """
    Independent objects retain their authored translation and rotation.
    """
    stage = build_stage_with_placed_objects(up_axis, meters_per_unit)

    world = parse(stage)

    for path in (ScenePrimPath.FIRST, ScenePrimPath.SECOND):
        prim = stage.GetPrimAtPath(path)
        expected = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        body = world.get_body_by_name(prim.GetName())
        np.testing.assert_allclose(
            body.global_transform.to_np(), np.array(expected).T, atol=1e-9
        )


def test_parse_bakes_root_scale_into_mesh_geometry() -> None:
    """
    A body frame excludes scale while its mesh retains the root's authored scale.
    """
    stage = build_stage_with_scaled_root()
    mesh_prim = stage.GetPrimAtPath(ScenePrimPath.MESH)
    expected_transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    expected_vertices = np.array(
        [
            expected_transform.Transform(Gf.Vec3d(point))
            for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
        ]
    )

    world = parse(stage)
    mesh = next(shape for shape in world.root.visual.shapes if isinstance(shape, Mesh))

    np.testing.assert_allclose(
        mesh.unscaled_mesh.vertices, expected_vertices, atol=1e-9
    )


def test_parse_bakes_root_scale_into_primitive_dimensions() -> None:
    """
    Primitive dimensions include scale authored on their enclosing root.
    """
    stage = build_stage_with_scaled_root()
    root_scale = Gf.Transform(
        UsdGeom.Xformable(
            stage.GetPrimAtPath(ScenePrimPath.ROOT)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).GetScale()
    cube_size = (
        UsdGeom.Cube(stage.GetPrimAtPath(ScenePrimPath.CUBE)).GetSizeAttr().Get()
    )

    world = parse(stage)
    box = next(shape for shape in world.root.visual.shapes if isinstance(shape, Box))

    np.testing.assert_allclose(
        [box.scale.x, box.scale.y, box.scale.z],
        np.array(root_scale) * cube_size,
        atol=1e-9,
    )


def test_parse_places_scaled_mesh_vertices_in_the_authored_world_frame() -> None:
    """
    The body's rigid pose and baked mesh scale reconstruct the authored geometry.
    """
    stage = build_stage_with_placed_objects(UsdAxis.Z, 1.0)
    prim = stage.GetPrimAtPath(ScenePrimPath.FIRST)
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(2, 3, 4))
    mesh_prim = next(child for child in prim.GetChildren() if child.IsA(UsdGeom.Mesh))
    expected_transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    expected_vertices = np.array(
        [
            expected_transform.Transform(Gf.Vec3d(point))
            for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
        ]
    )

    world = parse(stage)
    body = world.get_body_by_name(prim.GetName())
    mesh = next(shape for shape in body.visual.shapes if isinstance(shape, Mesh))
    vertices = mesh.unscaled_mesh.vertices
    homogeneous_vertices = np.column_stack((vertices, np.ones(len(vertices))))
    world_vertices = (
        body.global_transform.to_np() @ mesh.origin.to_np() @ homogeneous_vertices.T
    ).T[:, :3]

    np.testing.assert_allclose(world_vertices, expected_vertices, atol=1e-9)


# %% scene object ownership


def test_scene_parser_keeps_nested_objects_and_geometry_separate() -> None:
    """
    Grouping transforms place independently named bodies without copied geometry.
    """
    stage = build_mixed_scene()
    world = usd_parser.USDSceneParser(stage=stage, prefix="scene").parse()

    assert len(world.bodies) == 4
    assert len({body.name for body in world.bodies}) == len(world.bodies)
    for path in (MixedScenePath.FIRST, MixedScenePath.SECOND, MixedScenePath.CHILD):
        body = world.get_body_by_name(path.lstrip("/"))
        assert len(body.visual.shapes) == 1
        expected = UsdGeom.Xformable(
            stage.GetPrimAtPath(path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        np.testing.assert_allclose(
            body.global_transform.to_np(), np.array(expected).T, atol=1e-9
        )
    assert all(
        isinstance(connection, FixedConnection) for connection in world.connections
    )


def test_scene_parser_preserves_scaled_world_geometry() -> None:
    """
    Grouping scale is baked into geometry while body frames remain rigid.
    """
    stage = build_mixed_scene()
    UsdGeom.Xformable(stage.GetPrimAtPath(MixedScenePath.GROUP)).AddScaleOp().Set(
        Gf.Vec3f(2, 3, 4)
    )
    world = usd_parser.USDSceneParser(stage=stage, prefix="scene").parse()
    body = world.get_body_by_name(MixedScenePath.FIRST.lstrip("/"))
    mesh_prim = next(
        prim
        for prim in stage.GetPrimAtPath(MixedScenePath.FIRST).GetChildren()
        if prim.IsA(UsdGeom.Mesh)
    )
    expected_transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    expected = np.array(
        [
            expected_transform.Transform(Gf.Vec3d(point))
            for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
        ]
    )
    [mesh] = body.visual.shapes
    vertices = np.column_stack(
        (mesh.unscaled_mesh.vertices, np.ones(len(mesh.unscaled_mesh.vertices)))
    )
    actual = (body.global_transform.to_np() @ mesh.origin.to_np() @ vertices.T).T[:, :3]
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_scene_parser_excludes_guide_geometry() -> None:
    """
    Guide geometry is absent from both collision and visual geometry.
    """
    stage = build_mixed_scene()
    prim = next(
        prim
        for prim in stage.GetPrimAtPath(MixedScenePath.SECOND).GetChildren()
        if prim.IsA(UsdGeom.Gprim)
    )
    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    world = usd_parser.USDSceneParser(stage=stage).parse()
    assert {str(body.name.name) for body in world.bodies if body is not world.root} == {
        MixedScenePath.FIRST.lstrip("/"),
        MixedScenePath.CHILD.lstrip("/"),
    }


def test_scene_parser_keeps_component_geometry_groups_in_one_body() -> None:
    """
    A component owns geometry beneath its internal grouping transforms.
    """
    stage = build_component_scene()
    world = usd_parser.USDSceneParser(stage=stage).parse()
    [body] = [body for body in world.bodies if body is not world.root]

    assert body.name.name == ComponentScenePath.OBJECT.lstrip("/")
    assert len(body.visual.shapes) == 2


# %% authored scene collision geometry


def test_scene_parser_separates_visible_geometry_from_guide_collision_proxies() -> None:
    """
    A guide collider belongs to collision geometry without becoming visible.
    """
    stage = build_collision_scene()
    world = usd_parser.USDSceneParser(stage=stage).parse()
    [body] = [body for body in world.bodies if body is not world.root]

    [visual] = body.visual.shapes
    [collision] = body.collision.shapes
    assert isinstance(visual, Sphere)
    assert isinstance(collision, Box)
    expected_size = (
        UsdGeom.Cube(stage.GetPrimAtPath(CollisionScenePath.PROXY)).GetSizeAttr().Get()
    )
    assert collision.scale.x == expected_size


def test_scene_parser_keeps_objects_with_only_guide_collision_geometry() -> None:
    """
    An object with only an authored collision proxy remains addressable.
    """
    stage = build_collision_scene()
    stage.RemovePrim(CollisionScenePath.VISUAL)
    world = usd_parser.USDSceneParser(stage=stage).parse()
    [body] = [body for body in world.bodies if body is not world.root]

    assert body.name.name == CollisionScenePath.OBJECT.lstrip("/")
    assert body.visual.shapes == []
    [collision] = body.collision.shapes
    assert isinstance(collision, Box)


def test_scene_parser_does_not_replace_disabled_colliders_with_visual_geometry() -> (
    None
):
    """
    An explicitly disabled collider does not trigger visual collision fallback.
    """
    stage = build_collision_scene()
    UsdPhysics.CollisionAPI(
        stage.GetPrimAtPath(CollisionScenePath.PROXY)
    ).CreateCollisionEnabledAttr(False)
    world = usd_parser.USDSceneParser(stage=stage).parse()
    [body] = [body for body in world.bodies if body is not world.root]

    assert body.collision.shapes == []


def test_scene_parser_preserves_collision_proxies_in_a_loaded_payload(tmp_path) -> None:
    """
    Stage composition retains proxy ownership and its distinct geometry.
    """
    source = write_composed_collision_scene(tmp_path)
    world = usd_parser.USDSceneParser.from_file(str(source)).parse()
    [body] = [body for body in world.bodies if body is not world.root]

    assert body.name.name == CollisionScenePath.OBJECT.lstrip("/")
    [visual] = body.visual.shapes
    [collision] = body.collision.shapes
    assert isinstance(visual, Sphere)
    assert isinstance(collision, Box)


# %% scene articulations


@pytest.mark.parametrize(
    "joint_type, connection_type",
    [
        (SceneJointType.FIXED, FixedConnection),
        (SceneJointType.REVOLUTE, RevoluteConnection),
        (SceneJointType.PRISMATIC, PrismaticConnection),
    ],
)
def test_scene_parser_retains_joints_and_unrelated_objects(
    joint_type: SceneJointType, connection_type: type
) -> None:
    """
    A joint controls only its child while unrelated objects stay in the scene.
    """
    stage = build_mixed_scene(joint_type)
    world = usd_parser.USDSceneParser(stage=stage, prefix="scene").parse()
    child = world.get_body_by_name(MixedScenePath.CHILD.lstrip("/"))
    parent = world.get_body_by_name(MixedScenePath.FIRST.lstrip("/"))
    connection = next(
        connection for connection in world.connections if connection.child is child
    )
    assert type(connection) is connection_type
    assert connection.parent is parent
    assert len(world.bodies) == 4
    assert (
        len(world.get_body_by_name(MixedScenePath.SECOND.lstrip("/")).visual.shapes)
        == 1
    )
    if connection_type is not FixedConnection:
        joint = (
            UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(MixedScenePath.JOINT))
            if connection_type is RevoluteConnection
            else UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(MixedScenePath.JOINT))
        )
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        if connection_type is RevoluteConnection:
            lower, upper = math.radians(lower), math.radians(upper)
        assert connection.raw_dof.limits.lower.position == pytest.approx(lower)
        assert connection.raw_dof.limits.upper.position == pytest.approx(upper)
        unrelated = world.get_body_by_name(MixedScenePath.SECOND.lstrip("/"))
        initial = unrelated.global_transform.to_np().copy()
        connection.position = upper
        assert not np.allclose(
            child.global_transform.to_np(), parent.global_transform.to_np()
        )
        np.testing.assert_allclose(unrelated.global_transform.to_np(), initial)


def test_scene_parser_rejects_multiple_joints_for_one_child() -> None:
    """
    Two independent constraints cannot become one tree connection.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    joint = UsdPhysics.FixedJoint.Define(stage, MixedScenePath.OTHER_JOINT)
    joint.CreateBody1Rel().SetTargets([MixedScenePath.CHILD])
    with pytest.raises(usd_exceptions.UsdSceneTopologyError):
        usd_parser.USDSceneParser(stage=stage).parse()


def test_scene_parser_rejects_closed_joint_loops() -> None:
    """
    A joint back to an ancestor would close the scene's connection tree.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    joint = UsdPhysics.FixedJoint.Define(stage, MixedScenePath.OTHER_JOINT)
    joint.CreateBody0Rel().SetTargets([MixedScenePath.CHILD])
    joint.CreateBody1Rel().SetTargets([MixedScenePath.FIRST])
    with pytest.raises(usd_exceptions.UsdSceneTopologyError):
        usd_parser.USDSceneParser(stage=stage).parse()


def test_scene_parser_rejects_unsupported_joint_types() -> None:
    """
    An unsupported joint is diagnosed instead of flattening its object.
    """
    stage = build_mixed_scene(SceneJointType.SPHERICAL)
    with pytest.raises(UnsupportedUsdPhysicsJointTypeError):
        usd_parser.USDSceneParser(stage=stage).parse()


def test_scene_parser_rejects_scaled_joint_frames() -> None:
    """
    Joint anchors cannot silently lose an authored frame scale.
    """
    stage = build_mixed_scene(SceneJointType.REVOLUTE)
    UsdGeom.Xformable(stage.GetPrimAtPath(MixedScenePath.GROUP)).AddScaleOp().Set(
        Gf.Vec3f(2, 2, 2)
    )
    with pytest.raises(usd_exceptions.UsdScaledJointFrameError):
        usd_parser.USDSceneParser(stage=stage).parse()


@pytest.mark.parametrize("targets", [[], [MixedScenePath.CHILD, MixedScenePath.SECOND]])
def test_scene_parser_rejects_missing_or_ambiguous_joint_children(
    targets: list[str],
) -> None:
    """
    A child endpoint must select exactly one object.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    UsdPhysics.Joint(
        stage.GetPrimAtPath(MixedScenePath.JOINT)
    ).GetBody1Rel().SetTargets(targets)
    expected_error = (
        usd_exceptions.UsdSceneTopologyError
        if targets
        else UsdPhysicsJointMissingChildBodyError
    )
    with pytest.raises(expected_error):
        usd_parser.USDSceneParser(stage=stage).parse()


def test_scene_parser_rejects_nonexistent_joint_targets() -> None:
    """
    A broken reference is diagnosed before creating scene connections.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    stage.RemovePrim(MixedScenePath.CHILD)
    with pytest.raises(usd_exceptions.UsdSceneTopologyError):
        usd_parser.USDSceneParser(stage=stage).parse()


def test_scene_parser_keeps_joint_links_separate_inside_one_component() -> None:
    """
    Joint endpoints remain separate bodies within a larger model component.
    """
    stage = build_mixed_scene(SceneJointType.REVOLUTE)
    Usd.ModelAPI(stage.GetPrimAtPath(MixedScenePath.FIRST)).SetKind(
        Kind.Tokens.component
    )
    world = usd_parser.USDSceneParser(stage=stage).parse()
    parent = world.get_body_by_name(MixedScenePath.FIRST.lstrip("/"))
    child = world.get_body_by_name(MixedScenePath.CHILD.lstrip("/"))

    assert child.parent_connection.parent is parent
    assert isinstance(child.parent_connection, RevoluteConnection)
    assert len(parent.visual.shapes) == 1
    assert len(child.visual.shapes) == 1


# %% scene support semantics


@pytest.mark.skipif(not USD_SEMANTICS_AVAILABLE, reason="USD label schemas unavailable")
@pytest.mark.parametrize(
    "label, annotation_type",
    [
        (SceneLabel.FLOOR, Floor),
        (SceneLabel.TABLE, Table),
        (SceneLabel.WORKBENCH, Table),
    ],
)
def test_scene_parser_identifies_labelled_support_surfaces(
    label, annotation_type
) -> None:
    """
    A labelled object exposes the corresponding core support annotation.
    """
    stage = build_labelled_scene(label)
    world = usd_parser.USDSceneParser(stage=stage).parse()
    [annotation] = world.get_semantic_annotations_by_type(annotation_type)

    assert annotation.root.name.name == CollisionScenePath.OBJECT.lstrip("/")


@pytest.mark.skipif(not USD_SEMANTICS_AVAILABLE, reason="USD label schemas unavailable")
def test_scene_parser_preserves_inherited_labels_on_joint_link_bodies() -> None:
    """
    A joint link retains labels authored on its enclosing component.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    labels = UsdSemantics.LabelsAPI.Apply(
        stage.GetPrimAtPath(MixedScenePath.FIRST), SceneTaxonomy.CLASS
    )
    labels.CreateLabelsAttr([SceneLabel.UNKNOWN])
    world = usd_parser.USDSceneParser(stage=stage).parse()
    child = world.get_body_by_name(MixedScenePath.CHILD.lstrip("/"))
    [annotation] = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(UsdSemanticLabels)
        if annotation.root is child
    ]

    assert annotation.taxonomy == SceneTaxonomy.CLASS
    assert annotation.labels == list(labels.GetLabelsAttr().Get())


@pytest.mark.skipif(not USD_SEMANTICS_AVAILABLE, reason="USD label schemas unavailable")
def test_scene_parser_preserves_unknown_labels_without_inventing_support_types() -> (
    None
):
    """
    Unknown vocabulary remains available without a guessed support annotation.
    """
    stage = build_labelled_scene(SceneLabel.UNKNOWN)
    expected = (
        UsdSemantics.LabelsAPI.Get(
            stage.GetPrimAtPath(CollisionScenePath.OBJECT), SceneTaxonomy.CLASS
        )
        .GetLabelsAttr()
        .Get()
    )
    world = usd_parser.USDSceneParser(stage=stage).parse()
    [annotation] = world.get_semantic_annotations_by_type(UsdSemanticLabels)

    assert annotation.labels == list(expected)
    assert world.get_semantic_annotations_by_type(Floor) == []
    assert world.get_semantic_annotations_by_type(Table) == []


@pytest.mark.skipif(not USD_SEMANTICS_AVAILABLE, reason="USD label schemas unavailable")
def test_scene_parser_uses_the_nearest_label_for_support_classification() -> None:
    """
    A link's own table label takes precedence over its ancestor's floor label.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    UsdSemantics.LabelsAPI.Apply(
        stage.GetPrimAtPath(MixedScenePath.FIRST), SceneTaxonomy.CLASS
    ).CreateLabelsAttr([SceneLabel.FLOOR])
    UsdSemantics.LabelsAPI.Apply(
        stage.GetPrimAtPath(MixedScenePath.CHILD), SceneTaxonomy.CLASS
    ).CreateLabelsAttr([SceneLabel.TABLE])
    world = usd_parser.USDSceneParser(stage=stage).parse()
    child = world.get_body_by_name(MixedScenePath.CHILD.lstrip("/"))

    [table] = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(Table)
        if annotation.root is child
    ]
    assert table.root is child
    assert [
        annotation
        for annotation in world.get_semantic_annotations_by_type(Floor)
        if annotation.root is child
    ] == []


@pytest.mark.skipif(not USD_SEMANTICS_AVAILABLE, reason="USD label schemas unavailable")
def test_scene_parser_keeps_labelled_object_geometry_together_without_model_kind() -> (
    None
):
    """
    An authored object label owns visual and collision groups without model metadata.
    """
    stage = build_labelled_scene(SceneLabel.FLOOR)
    Usd.ModelAPI(stage.GetPrimAtPath(CollisionScenePath.OBJECT)).SetKind("")
    world = usd_parser.USDSceneParser(stage=stage).parse()
    [body] = [body for body in world.bodies if body is not world.root]

    assert body.name.name == CollisionScenePath.OBJECT.lstrip("/")
    [visual] = body.visual.shapes
    [collision] = body.collision.shapes
    assert isinstance(visual, Sphere)
    assert isinstance(collision, Box)


@pytest.mark.skipif(not USD_SEMANTICS_AVAILABLE, reason="USD label schemas unavailable")
def test_scene_parser_support_priority_does_not_depend_on_taxonomy_order() -> None:
    """
    A direct support label wins over an inherited label in another taxonomy.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    UsdSemantics.LabelsAPI.Apply(
        stage.GetPrimAtPath(MixedScenePath.FIRST), SceneTaxonomy.CLASS
    ).CreateLabelsAttr([SceneLabel.FLOOR])
    child_prim = stage.GetPrimAtPath(MixedScenePath.CHILD)
    UsdSemantics.LabelsAPI.Apply(child_prim, SceneTaxonomy.CLASS).CreateLabelsAttr(
        [SceneLabel.UNKNOWN]
    )
    UsdSemantics.LabelsAPI.Apply(child_prim, SceneTaxonomy.CATEGORY).CreateLabelsAttr(
        [SceneLabel.TABLE]
    )
    world = usd_parser.USDSceneParser(stage=stage).parse()
    child = world.get_body_by_name(MixedScenePath.CHILD.lstrip("/"))

    [table] = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(Table)
        if annotation.root is child
    ]
    assert table.root is child
    assert [
        annotation
        for annotation in world.get_semantic_annotations_by_type(Floor)
        if annotation.root is child
    ] == []


@pytest.mark.skipif(not USD_SEMANTICS_AVAILABLE, reason="USD label schemas unavailable")
def test_scene_parser_support_vocabulary_is_configurable_and_case_insensitive() -> None:
    """
    A configured external label is matched without regard to authored case.
    """
    stage = build_labelled_scene(SceneLabel.UNKNOWN.upper())
    semantics = UsdSceneSemantics(support_labels={SceneLabel.UNKNOWN: Floor})
    world = usd_parser.USDSceneParser(stage=stage, semantics=semantics).parse()
    [floor] = world.get_semantic_annotations_by_type(Floor)

    assert floor.root.name.name == CollisionScenePath.OBJECT.lstrip("/")


# %% scene materialization


def test_scene_placement_moves_static_objects_and_articulations_together() -> None:
    """
    A stage pose affects root connections once while retaining local joint frames.
    """
    stage = build_mixed_scene(SceneJointType.REVOLUTE)
    authored = usd_parser.USDSceneParser(stage=stage).parse()
    world_T_scene = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=4, y=-2, z=1, yaw=np.pi / 2
    )
    placed = usd_parser.USDSceneParser(stage=stage, world_T_scene=world_T_scene).parse()

    for body in authored.bodies:
        if body is authored.root:
            continue
        actual = placed.get_body_by_name(body.name)
        np.testing.assert_allclose(
            actual.global_transform.to_np(),
            world_T_scene.to_np() @ body.global_transform.to_np(),
            atol=1e-12,
        )


def test_usd_world_specification_preserves_world_frame_spawn_poses(
    tmp_path: Path,
) -> None:
    """
    Starting objects use the world frame independently of the scene placement.
    """
    stage = build_mixed_scene(SceneJointType.FIXED)
    scene_file = tmp_path / CollisionSceneFile.SCENE
    stage.GetRootLayer().Export(str(scene_file))
    world_T_scene = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, z=2)
    object_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=-1, y=2, z=4)
    object_specification = BodySpecification.box(
        "transport_object", Scale(0.1, 0.1, 0.1), parent_T_self=object_pose
    )
    specification = WorldSpecification.from_usd_scene(
        str(scene_file), world_T_scene=world_T_scene, objects=[object_specification]
    )
    world = specification.to_domain_object()

    np.testing.assert_allclose(
        world.get_body_by_name(object_specification.name).global_transform.to_np(),
        object_pose.to_np(),
    )
    authored = usd_parser.USDSceneParser(stage=stage).parse()
    body = authored.get_body_by_name(MixedScenePath.SECOND.lstrip("/"))
    actual = world.get_body_by_name(MixedScenePath.SECOND.lstrip("/"))
    np.testing.assert_allclose(
        actual.global_transform.to_np(),
        world_T_scene.to_np() @ body.global_transform.to_np(),
    )


def test_usd_world_specification_materializes_fresh_scenes(tmp_path: Path) -> None:
    """
    Repeated materialization creates independent entities and retains articulation.
    """
    stage = build_mixed_scene(SceneJointType.REVOLUTE)
    scene_file = tmp_path / CollisionSceneFile.SCENE
    stage.GetRootLayer().Export(str(scene_file))
    specification = WorldSpecification.from_usd_scene(str(scene_file))
    first = specification.to_domain_object()
    second = specification.to_domain_object()

    assert {body.id for body in first.bodies}.isdisjoint(
        body.id for body in second.bodies
    )
    child = second.get_body_by_name(MixedScenePath.CHILD.lstrip("/"))
    assert isinstance(child.parent_connection, RevoluteConnection)
