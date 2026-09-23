from __future__ import annotations

from pathlib import Path

from typing_extensions import Optional

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False

try:
    from pxr import UsdSemantics

    USD_SEMANTICS_AVAILABLE = True
except ImportError:
    USD_SEMANTICS_AVAILABLE = False


def _define_link(stage: Usd.Stage, path: str) -> None:
    """
    Define a link ``Xform`` at ``path`` with a single quad mesh, so it has visual
    geometry the way every real ArtVIP link does.
    """
    UsdGeom.Xform.Define(stage, path)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])


def build_single_joint_stage(
    joint_type: str,
    *,
    axis: str = "Z",
    local_pos0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot1: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    lower_limit: float = -90.0,
    upper_limit: float = 0.0,
) -> Usd.Stage:
    """
    A minimal in-memory stage with a root link ("carcass") and a child link ("child")
    connected by one joint of ``joint_type``, in the same shape ``USDParser.parse``
    reads: ``Xform`` links each holding one ``Mesh``, and a joint prim with
    body0/body1 relationships and localPos/localRot/axis/limit attributes.

    :param joint_type: A ``UsdPhysics`` joint type name, e.g. ``"RevoluteJoint"``,
        ``"PrismaticJoint"``, ``"FixedJoint"``, or ``"SphericalJoint"`` for an
        unsupported-type test.
    :param axis: The joint's local-frame axis token.
    :param local_pos0: The joint frame's translation relative to body0.
    :param local_rot0: The joint frame's rotation (w, x, y, z) relative to body0.
    :param local_pos1: The joint frame's translation relative to body1.
    :param local_rot1: The joint frame's rotation (w, x, y, z) relative to body1.
    :param lower_limit: The joint's lower limit (degrees for Revolute, meters for
        Prismatic; ignored for Fixed/Spherical).
    :param upper_limit: The joint's upper limit.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    _define_link(stage, "/object/child")

    joint_class = getattr(UsdPhysics, joint_type)
    joint = joint_class.Define(stage, "/object/joint")
    # body0 is left with no targets: the USD convention for "the object's own root
    # frame", so the built connection's parent is the object's root body.
    joint.CreateBody1Rel().SetTargets(["/object/child"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(*local_rot0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(*local_rot1))
    if joint_type in ("RevoluteJoint", "PrismaticJoint"):
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(lower_limit)
        joint.CreateUpperLimitAttr(upper_limit)

    return stage


def build_stage_with_joint_missing_body1() -> Usd.Stage:
    """
    A minimal in-memory stage with a single ``FixedJoint`` whose ``body1`` relationship
    has no target - unlike ``body0``, an unset ``body1`` has no "the stage's own frame"
    meaning for ``USDParser._describe_joint``, since every joint is expected to connect
    a link into the world.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    joint = UsdPhysics.FixedJoint.Define(stage, "/object/joint")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    return stage


def build_stage_with_mesh_targeted_body0() -> Usd.Stage:
    """
    Reproduces a layout seen on real ArtVIP basket objects: one link ("carcass") is
    connected to the root by a fixed joint targeting its enclosing Xform, and a second
    joint's body0 targets that same link's Mesh prim directly instead of the Xform. Both
    should resolve to the same link body, not create a second, disconnected one.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    _define_link(stage, "/object/carcass")
    _define_link(stage, "/object/handle")

    fixed = UsdPhysics.FixedJoint.Define(stage, "/object/carcass/joint")
    fixed.CreateBody1Rel().SetTargets(["/object/carcass"])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    fixed.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/object/handle/joint")
    hinge.CreateBody0Rel().SetTargets(["/object/carcass/mesh"])
    hinge.CreateBody1Rel().SetTargets(["/object/handle"])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLowerLimitAttr(-30.0)
    hinge.CreateUpperLimitAttr(30.0)
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    return stage


def build_stage_with_textured_mesh(texture_file_path: str) -> Usd.Stage:
    """
    A minimal in-memory stage with a single quad mesh, per-point ``st`` UV.

    coordinates, and a material whose ``diffuseColor`` is driven by a texture read
    from ``texture_file_path`` - the layout
    ``USDParser._diffuse_texture_path``/``_uv_coordinates`` read.

    :param texture_file_path: Path to the texture image the material's
        ``UsdUVTexture`` node reads.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/object/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    _bind_textured_material(stage, "/object", texture_file_path)

    return stage


def _bind_textured_material(
    stage: Usd.Stage, object_path: str, texture_file_path: str
) -> None:
    """
    Define a ``UsdPreviewSurface`` material under ``object_path`` whose diffuse colour
    is read from ``texture_file_path``, and bind it to the mesh beside it.
    """
    material = UsdShade.Material.Define(stage, f"{object_path}/material")
    pbr_shader = UsdShade.Shader.Define(stage, f"{object_path}/material/PBRShader")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        pbr_shader.ConnectableAPI(), "surface"
    )

    texture_shader = UsdShade.Shader.Define(
        stage, f"{object_path}/material/diffuseTexture"
    )
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file_path)
    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture_shader.ConnectableAPI(), "rgb"
    )

    mesh_prim = stage.GetPrimAtPath(f"{object_path}/mesh")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)


def build_stage_with_scaled_mesh(scale: tuple[float, float, float]) -> Usd.Stage:
    """
    A minimal in-memory stage with a quad mesh under a child ``Xform`` authored with
    ``scale`` - the shape of decorative props seen on real ArtVIP scene furniture (e.g.
    a fruit platter's leaf), whose local-to-world transform carries scale on top of its
    translation and rotation.

    :param scale: The child ``Xform``'s authored scale.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object")
    scaled = UsdGeom.Xform.Define(stage, "/object/scaled")
    scaled.AddScaleOp().Set(Gf.Vec3f(*scale))
    mesh = UsdGeom.Mesh.Define(stage, "/object/scaled/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    return stage


def build_jointless_stage_with_a_default_prim() -> Usd.Stage:
    """
    A minimal in-memory stage with a default prim and no physics joints at all - a
    plain, non-articulated static USD asset (e.g. a decorative prop), the shape a stage
    with nothing for :class:`~pxr.UsdPhysics.Joint` to connect takes.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/prop")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/prop/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    return stage


def build_stage_with_ambiguous_root_and_a_joint() -> Usd.Stage:
    """
    A minimal in-memory stage with no default prim, two top-level prims, and a physics
    joint targeting one of them - exercising the synthetic-root fallback in combination
    with a joint graph, rather than the joint-less case it is otherwise exercised with.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/object_a")
    _define_link(stage, "/object_b")
    joint = UsdPhysics.FixedJoint.Define(stage, "/object_b/joint")
    joint.CreateBody1Rel().SetTargets(["/object_b"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    return stage


def build_jointless_stage_with_multiple_top_level_prims() -> Usd.Stage:
    """
    A minimal in-memory stage with no default prim and two top-level prims, each its
    own labelled link with visual geometry, and no physics joints at all - the shape of
    a USD "scene" file collecting several independent objects rather than one
    articulated asset.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    _define_link(stage, "/object_a")
    _define_link(stage, "/object_b")

    return stage


def build_jointless_stage_with_unsupported_geometry() -> Usd.Stage:
    """
    A minimal in-memory stage with a default prim holding a ``Cone`` prim - a
    :class:`~pxr.UsdGeom.Gprim` this parser does not build a Shape for.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/prop")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cone.Define(stage, "/prop/cone")

    return stage


def build_stage_with_primitive_shapes(
    *,
    cube_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cylinder_axis: str = "Z",
    cylinder_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Usd.Stage:
    """
    A minimal in-memory stage with a root link holding one ``Cube``, one ``Sphere``, and
    one ``Cylinder`` prim - the native USD primitive shapes ArtVIP never uses but a
    general-purpose USD asset can.

    :param cube_scale: The cube prim's authored scale.
    :param cylinder_axis: The cylinder prim's authored axis token.
    :param cylinder_scale: The cylinder prim's authored scale.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/object")
    stage.SetDefaultPrim(root.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/object/cube")
    cube.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    cube.AddScaleOp().Set(Gf.Vec3f(*cube_scale))
    cube.CreateSizeAttr(2.0)

    sphere = UsdGeom.Sphere.Define(stage, "/object/sphere")
    sphere.AddTranslateOp().Set(Gf.Vec3d(0, 1, 0))
    sphere.CreateRadiusAttr(0.5)

    cylinder = UsdGeom.Cylinder.Define(stage, "/object/cylinder")
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
    cylinder.AddScaleOp().Set(Gf.Vec3f(*cylinder_scale))
    cylinder.CreateRadiusAttr(0.5)
    cylinder.CreateHeightAttr(2.0)
    cylinder.CreateAxisAttr(cylinder_axis)

    return stage


def build_single_joint_stage_with_mass(
    *,
    mass: float = 2.0,
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    diagonal_inertia: tuple[float, float, float] = (1.0, 2.0, 3.0),
    principal_axes: Optional[tuple[float, float, float, float]] = (1.0, 0.0, 0.0, 0.0),
) -> Usd.Stage:
    """
    A minimal in-memory stage like :func:`build_single_joint_stage`, but with
    :class:`~pxr.UsdPhysics.MassAPI` applied to the child link.

    :param mass: The child link's authored mass.
    :param center_of_mass: The child link's authored centre of mass.
    :param diagonal_inertia: The child link's authored diagonal inertia.
    :param principal_axes: The child link's authored principal axes as ``(w, x, y, z)``,
        or ``None`` to leave them unauthored the way a file stating only a mass does.
    :return: The built in-memory stage.
    """
    stage = build_single_joint_stage("FixedJoint")
    link_prim = stage.GetPrimAtPath("/object/child")
    mass_api = UsdPhysics.MassAPI.Apply(link_prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal_inertia))
    if principal_axes is not None:
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*principal_axes))

    return stage


def build_single_joint_stage_with_semantic_labels() -> Usd.Stage:
    """
    A minimal in-memory stage like :func:`build_single_joint_stage`, but with
    :class:`~pxr.UsdSemantics.LabelsAPI` labels applied to the child link in two
    taxonomies.

    :return: The built in-memory stage.
    """
    stage = build_single_joint_stage("FixedJoint")
    link_prim = stage.GetPrimAtPath("/object/child")
    UsdSemantics.LabelsAPI.Apply(link_prim, "class").CreateLabelsAttr().Set(
        ["chair", "furniture"]
    )
    UsdSemantics.LabelsAPI.Apply(link_prim, "category").CreateLabelsAttr().Set(
        ["seating"]
    )

    return stage


# %% scene stages


def _define_placed_instance(
    stage: Usd.Stage, path: str, translation: tuple[float, float, float]
) -> None:
    """
    Define a link ``Xform`` at ``path`` holding one quad mesh, translated by
    ``translation`` relative to its parent prim.
    """
    _define_link(stage, path)
    UsdGeom.Xform(stage.GetPrimAtPath(path)).AddTranslateOp().Set(
        Gf.Vec3d(*translation)
    )


def build_scene_stage_with_grouped_instances(up_axis: str = "Z") -> Usd.Stage:
    """
    A minimal in-memory stage shaped like a scanned building.

    A default prim holds category ``Xform`` groups that carry a transform but no
    geometry of their own, each holding the separately placed instances that do - the
    shape a USD scene of independent static objects takes, as opposed to one
    articulated asset.

    :param up_axis: The stage's up axis token, which decides which way is down and so
        where the ground of the scene lies.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, up_axis)
    root = UsdGeom.Xform.Define(stage, "/scene")
    stage.SetDefaultPrim(root.GetPrim())

    wall_group = UsdGeom.Xform.Define(stage, "/scene/Wall")
    wall_group.AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
    _define_placed_instance(stage, "/scene/Wall/wall_a", (1, 0, 0))
    _define_placed_instance(stage, "/scene/Wall/wall_b", (0, 2, 0))

    UsdGeom.Xform.Define(stage, "/scene/Floor")
    _define_placed_instance(stage, "/scene/Floor/floor_a", (0, 0, 3))

    return stage


def build_scene_stage_with_nested_objects() -> Usd.Stage:
    """
    A minimal in-memory stage where one geometry-owning prim sits inside another's
    subtree, so the inner object's nearest enclosing object - not the stage root - is
    what it is placed relative to.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/scene")
    stage.SetDefaultPrim(root.GetPrim())
    _define_placed_instance(stage, "/scene/outer", (1, 0, 0))
    _define_placed_instance(stage, "/scene/outer/inner", (0, 1, 0))

    return stage


def build_scene_stage_with_a_scaled_group() -> Usd.Stage:
    """
    A minimal in-memory stage whose grouping ``Xform`` carries a non-uniform scale, so
    the object it holds is both displaced and scaled by it.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/scene")
    stage.SetDefaultPrim(root.GetPrim())
    group = UsdGeom.Xform.Define(stage, "/scene/group")
    group.AddScaleOp().Set(Gf.Vec3f(2, 3, 4))
    _define_placed_instance(stage, "/scene/group/object", (1, 0, 0))

    return stage


def build_scene_stage_with_semantic_labels() -> Usd.Stage:
    """
    A minimal in-memory stage like :func:`build_scene_stage_with_grouped_instances`, but
    with :class:`~pxr.UsdSemantics.LabelsAPI` labels applied to one instance.

    :return: The built in-memory stage.
    """
    stage = build_scene_stage_with_grouped_instances()
    UsdSemantics.LabelsAPI.Apply(
        stage.GetPrimAtPath("/scene/Wall/wall_a"), "class"
    ).CreateLabelsAttr().Set(["wall"])

    return stage


def build_usdz_package_with_a_textured_mesh(
    directory: Path, texture_file_path: str
) -> str:
    """
    Write a ``.usdz`` package holding a textured mesh and the texture itself.

    A texture travelling inside the package resolves to a path only USD's asset resolver
    can read, unlike one lying beside the stage as a plain file.

    :param directory: The directory to write the package and its source stage to.
    :param texture_file_path: Path to the texture image to package.
    :return: The path of the written package.
    """
    stage = build_stage_with_textured_mesh(texture_file_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/object"))
    source_path = directory / "textured.usda"
    stage.GetRootLayer().Export(str(source_path))

    package_path = directory / "textured.usdz"
    UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source_path)), str(package_path))
    return str(package_path)


def build_stage_with_a_double_sided_mesh(double_sided: bool) -> Usd.Stage:
    """
    A minimal in-memory stage with a two-triangle mesh whose ``doubleSided`` attribute
    is authored either way.

    A scanned surface is a sheet rather than a solid, so it carries this flag to say it
    must be drawn whichever side it is seen from.

    :param double_sided: The value to author on the mesh's ``doubleSided`` attribute.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/object/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    mesh.CreateDoubleSidedAttr(double_sided)

    return stage


def build_scene_stage_with_textured_objects(texture_file_path: str) -> Usd.Stage:
    """
    A minimal in-memory stage shaped like the export of a scanned building.

    Category ``Xform`` groups carry a transform and hold the separately placed
    geometry-owning prims, each of which holds both its mesh and the material bound to
    it - the layout a photogrammetry export writes, and the one an asset library is
    split along.

    :param texture_file_path: Path to the texture image each object's material reads.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/scene")
    stage.SetDefaultPrim(root.GetPrim())

    wall_group = UsdGeom.Xform.Define(stage, "/scene/Wall")
    wall_group.AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
    for name, translation in (("wall_a", (1, 0, 0)), ("wall_b", (0, 2, 0))):
        _define_placed_instance(stage, f"/scene/Wall/{name}", translation)
        _bind_textured_material(stage, f"/scene/Wall/{name}", texture_file_path)

    UsdGeom.Xform.Define(stage, "/scene/Floor")
    _define_placed_instance(stage, "/scene/Floor/floor_a", (0, 0, 3))
    _bind_textured_material(stage, "/scene/Floor/floor_a", texture_file_path)

    return stage


def build_scene_stage_with_a_guide_prim() -> Usd.Stage:
    """
    A minimal in-memory stage whose object holds a guide beside its mesh.

    A guide is geometry a renderer draws nothing for, which is the shape a collision
    proxy authored beside the surface it stands for takes.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/scene").GetPrim())

    _define_placed_instance(stage, "/scene/Wall/wall_a", (1, 0, 0))
    proxy = UsdGeom.Cube.Define(stage, "/scene/Wall/wall_a/collision")
    proxy.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    return stage


def build_scene_stage_with_a_triangle_soup() -> Usd.Stage:
    """
    A minimal in-memory stage whose object is exported as loose triangles.

    Two triangles meet along an edge, but every corner carries its own point, its own
    texture coordinate and the flat normal of the face it belongs to - the shape a
    photogrammetry export takes, where nothing is shared between faces.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/scene").GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/scene/Wall/wall_a/mesh")
    mesh.CreatePointsAttr(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    mesh.CreateNormalsAttr(
        [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, -1), (0, 0, -1), (0, 0, -1)]
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([(0, 0), (1, 0), (1, 1), (0.1, 0.1), (0.9, 0.9), (0.1, 0.9)])

    return stage


def build_stage_with_face_varying_texture_coordinates(
    texture_file_path: str,
) -> Usd.Stage:
    """
    A minimal in-memory stage whose mesh textures each face corner separately.

    Two triangles share the vertices along the edge they meet at, but texture them
    differently there, so the ``st`` primvar holds one coordinate per face corner
    rather than one per point - what a mesh looks like once loose triangles have been
    given one vertex per position.

    :param texture_file_path: Path to the texture image the material's
        ``UsdUVTexture`` node reads.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/object/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])

    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([(0, 0), (1, 0), (1, 1), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)])

    _bind_textured_material(stage, "/object", texture_file_path)

    return stage


def build_scene_stage_with_repeated_container_names() -> Usd.Stage:
    """
    A minimal in-memory stage shaped like a referenced asset library.

    Each object is placed as an actor holding a geometry container, and every asset
    names those the same way, so the prim holding the geometry says nothing about which
    object it belongs to.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/scene").GetPrim())

    for asset in ("sofa", "chair"):
        _define_placed_instance(stage, f"/scene/{asset}/Actor_0000/Geom", (1, 0, 0))

    return stage


def build_scene_stage_with_a_guide_under_a_prim_of_its_own() -> Usd.Stage:
    """
    A minimal in-memory stage whose object holds its collision boxes under prims that
    hold nothing else.

    An asset library writing one box per opening of a wall gives each box a prim of its
    own beside the surface, so the only geometry those prims hold is a guide.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/scene").GetPrim())

    _define_placed_instance(stage, "/scene/Wall/wall_a", (1, 0, 0))
    for name in ("box_0", "box_1"):
        box = UsdGeom.Cube.Define(stage, f"/scene/Wall/wall_a/Collision/{name}/cube")
        box.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    return stage


def build_scene_stage_with_an_object_of_several_faces(
    texture_file_path: str,
) -> Usd.Stage:
    """
    A minimal in-memory stage whose object is made of more than one face.

    Two squares side by side, textured per point, so that a segmentation taking some of
    the faces has something to leave behind.

    :param texture_file_path: Path to the texture image the object's material reads.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/scene").GetPrim())

    UsdGeom.Xform.Define(stage, "/scene/Wall/wall_a")
    mesh = UsdGeom.Mesh.Define(stage, "/scene/Wall/wall_a/mesh")
    mesh.CreatePointsAttr(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 0, 1),
            (0, 0, 1),
            (2, 0, 0),
            (2, 0, 1),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4, 4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 5, 2])
    mesh.CreateNormalsAttr([(0, -1, 0), (0, -1, 0)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(0, 0), (0.5, 0), (0.5, 1), (0, 1), (1, 0), (1, 1)])

    _bind_textured_material(stage, "/scene/Wall/wall_a", texture_file_path)

    return stage


def _define_collision_box(
    stage: Usd.Stage,
    path: str,
    translation: tuple[float, float, float],
    turn: float,
    extents: tuple[float, float, float],
) -> None:
    """
    Define one collision box the way an asset library writes it: an ``Xform`` placing it
    under a ``Cube`` scaled to its extents, marked a guide and collided against.

    :param stage: The stage to define it in.
    :param path: The ``Xform``'s path; the ``Cube`` is its child ``cube``.
    :param translation: Where the box's middle sits relative to its parent.
    :param turn: How far the box is turned about the upright axis, in degrees.
    :param extents: The box's full size along each of its own axes.
    """
    placement = UsdGeom.Xform.Define(stage, path)
    placement.AddTranslateOp().Set(Gf.Vec3d(*translation))
    placement.AddRotateZOp().Set(turn)
    cube = UsdGeom.Cube.Define(stage, f"{path}/cube")
    cube.GetSizeAttr().Set(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*extents))
    cube.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def build_scene_stage_with_authored_collision() -> Usd.Stage:
    """
    A minimal in-memory stage whose wall carries the collision an asset library
    writes: two boxes under a ``Collision`` scope, a door leaf of its own beneath the
    wall with one box of its own, and one guide that is collided against by nothing.

    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/scene").GetPrim())

    _define_placed_instance(stage, "/scene/Wall/wall_a", (1, 0, 0))
    _define_collision_box(
        stage,
        "/scene/Wall/wall_a/Collision/box_0",
        (2.0, 0.0, 1.25),
        30.0,
        (4.0, 0.2, 2.5),
    )
    _define_collision_box(
        stage,
        "/scene/Wall/wall_a/Collision/box_1",
        (5.0, 0.0, 2.0),
        30.0,
        (1.0, 0.2, 1.0),
    )
    UsdGeom.Cube.Define(stage, "/scene/Wall/wall_a/proxy").CreatePurposeAttr().Set(
        UsdGeom.Tokens.guide
    )

    _define_placed_instance(stage, "/scene/Wall/wall_a/door_0", (3.0, 0.0, 0.0))
    _define_collision_box(
        stage,
        "/scene/Wall/wall_a/door_0/Collision/box_0",
        (0.5, 0.0, 1.0),
        0.0,
        (1.0, 0.04, 2.0),
    )
    return stage


def build_scene_stage_with_a_door_on_a_hinge(
    anchored_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    lower: float = -90.0,
    upper: float = 5.0,
) -> Usd.Stage:
    """
    A minimal in-memory stage shaped like a scanned wall with a door hung in it: the
    wall's own surface, a leaf standing beside it, and the revolute joint an asset
    library writes to hold the one to the other.

    :param anchored_at: Where the joint sits in the leaf's own frame. A library writes
        each leaf about its own hinge, which is this left at the origin.
    :param lower: The least the joint turns to, in degrees.
    :param upper: The most it turns to, in degrees.
    :return: The built in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/scene").GetPrim())

    _define_placed_instance(stage, "/scene/Wall/wall_a/Geometry/surface", (0, 0, 0))
    _define_placed_instance(stage, "/scene/Wall/wall_a/Geometry/door_0", (1, 0, 0))

    joint = UsdPhysics.RevoluteJoint.Define(
        stage, "/scene/Wall/wall_a/Geometry/surface/hinge"
    )
    joint.CreateBody0Rel().SetTargets(["/scene/Wall/wall_a/Geometry/surface"])
    joint.CreateBody1Rel().SetTargets(["/scene/Wall/wall_a/Geometry/door_0"])
    joint.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(1, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*anchored_at))
    joint.CreateLowerLimitAttr().Set(lower)
    joint.CreateUpperLimitAttr().Set(upper)
    return stage
