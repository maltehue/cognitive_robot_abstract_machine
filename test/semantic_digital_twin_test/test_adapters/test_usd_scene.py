import numpy as np
import pytest
from PIL import Image

from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedUsdGeometryTypeError,
)
from semantic_digital_twin.adapters.usd.scene_parser import (
    RootPlacement,
    USDSceneParser,
)
from semantic_digital_twin.adapters.usd.stage_parser import Shading
from semantic_digital_twin.pipeline.mesh_decomposition.bounding_box import (
    BoundingBoxDecomposer,
)
from semantic_digital_twin.pipeline.pipeline import Pipeline
from semantic_digital_twin.semantic_annotations.usd_semantics import (
    UsdSemanticLabels,
    UsdStageOrigin,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh
from semantic_digital_twin.world_description.world_entity import Body, Connection

from .usd_stages import (
    PXR_AVAILABLE,
    USD_SEMANTICS_AVAILABLE,
    build_jointless_stage_with_a_default_prim,
    build_jointless_stage_with_multiple_top_level_prims,
    build_jointless_stage_with_unsupported_geometry,
    build_scene_stage_with_a_guide_prim,
    build_scene_stage_with_authored_collision,
    build_scene_stage_with_a_door_on_a_hinge,
    build_scene_stage_with_a_guide_under_a_prim_of_its_own,
    build_scene_stage_with_a_scaled_group,
    build_scene_stage_with_grouped_instances,
    build_scene_stage_with_nested_objects,
    build_scene_stage_with_repeated_container_names,
    build_scene_stage_with_semantic_labels,
    build_stage_with_textured_mesh,
)

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)


def parse(stage, root_placement: RootPlacement = RootPlacement.STAGE_ORIGIN) -> World:
    return USDSceneParser(
        stage=stage, prefix="scene", root_placement=root_placement
    ).parse()


def stage_origin_of(world: World) -> UsdStageOrigin:
    [annotation] = [
        annotation
        for annotation in world.semantic_annotations
        if isinstance(annotation, UsdStageOrigin)
    ]
    return annotation


def body_named(world: World, name: str) -> Body:
    [body] = [body for body in world.bodies if body.name.name == name]
    return body


def connection_to(world: World, child_name: str) -> Connection:
    [connection] = [
        connection
        for connection in world.connections
        if connection.child.name.name == child_name
    ]
    return connection


def translation_of(connection: Connection) -> np.ndarray:
    return connection.parent_T_connection_expression.to_np()[:3, 3]


# %% bodies


def test_parse_builds_one_body_per_geometry_owning_prim():
    # The category groups (Wall, Floor) carry a transform but no geometry of their own,
    # so they place the objects they hold without becoming bodies themselves.
    world = parse(build_scene_stage_with_grouped_instances())

    assert {body.name.name for body in world.bodies} == {
        "scene",
        "wall_a",
        "wall_b",
        "floor_a",
    }


def test_parse_keeps_the_root_prim_as_a_geometry_less_world_root():
    world = parse(build_scene_stage_with_grouped_instances())

    assert world.root.name.name == "scene"
    assert len(world.root.visual.shapes) == 0


def test_parse_gives_each_object_the_geometry_authored_under_it():
    world = parse(build_scene_stage_with_grouped_instances())

    assert len(body_named(world, "wall_a").visual.shapes) == 1


def test_parse_builds_a_single_body_when_the_root_prim_owns_the_geometry():
    world = parse(build_jointless_stage_with_a_default_prim())

    assert len(world.bodies) == 1
    assert len(world.connections) == 0
    assert len(world.root.visual.shapes) == 1


# %% placement


def test_parse_fixes_every_object_to_the_root():
    # A scanned object is rigidly placed where it was measured, not freely posable.
    world = parse(build_scene_stage_with_grouped_instances())

    assert len(world.connections) == 3
    for connection in world.connections:
        assert isinstance(connection, FixedConnection)
        assert connection.parent is world.root


def test_parse_applies_a_grouping_prims_transform_to_the_objects_it_holds():
    # wall_a is translated by (1, 0, 0) inside a Wall group translated by (10, 0, 0):
    # dropping the group as a body must not drop its transform with it.
    world = parse(build_scene_stage_with_grouped_instances())

    np.testing.assert_allclose(
        translation_of(connection_to(world, "wall_a")), [11.0, 0.0, 0.0], atol=1e-6
    )


def test_parse_places_an_object_relative_to_the_object_whose_subtree_holds_it():
    # inner sits inside outer's subtree, so its (0, 1, 0) is relative to outer, not to
    # the world root outer itself is placed against.
    world = parse(build_scene_stage_with_nested_objects())

    connection = connection_to(world, "inner")
    assert connection.parent is body_named(world, "outer")
    np.testing.assert_allclose(translation_of(connection), [0.0, 1.0, 0.0], atol=1e-6)


def test_parse_fixes_top_level_objects_to_a_synthetic_root():
    # With no default prim and several top-level prims there is no prim to treat as the
    # scene's own root, so a synthetic one named after the prefix holds the objects.
    world = parse(build_jointless_stage_with_multiple_top_level_prims())

    assert world.root.name.name == "scene"
    assert {body.name.name for body in world.bodies} == {
        "scene",
        "object_a",
        "object_b",
    }
    for connection in world.connections:
        assert isinstance(connection, FixedConnection)
        assert connection.parent is world.root


# %% scale


def test_parse_places_an_object_at_its_scaled_translation():
    # The group scales the object's own (1, 0, 0) offset to (2, 0, 0).
    world = parse(build_scene_stage_with_a_scaled_group())

    np.testing.assert_allclose(
        translation_of(connection_to(world, "object")), [2.0, 0.0, 0.0], atol=1e-6
    )


def test_parse_bakes_a_group_scale_into_the_objects_shape():
    # A connection is rigid, so a grouping prim's scale has nowhere to go but the
    # shape's own geometry.
    world = parse(build_scene_stage_with_a_scaled_group())

    [shape] = body_named(world, "object").visual.shapes
    np.testing.assert_allclose(
        shape.unscaled_mesh.vertices.max(axis=0), [2.0, 3.0, 0.0], atol=1e-5
    )


# %% geometry


def test_parse_raises_on_unsupported_geometry_instead_of_silently_dropping_it():
    stage = build_jointless_stage_with_unsupported_geometry()

    with pytest.raises(UnsupportedUsdGeometryTypeError) as excinfo:
        parse(stage)

    assert excinfo.value.geometry_type == "Cone"


# %% semantics


@pytest.mark.skipif(
    not USD_SEMANTICS_AVAILABLE, reason="usd-core predates UsdSemantics"
)
def test_parse_attaches_semantic_labels_to_the_object_they_are_authored_on():
    world = parse(build_scene_stage_with_semantic_labels())

    [annotation] = [
        annotation
        for annotation in world.semantic_annotations
        if isinstance(annotation, UsdSemanticLabels)
    ]
    assert annotation.root is body_named(world, "wall_a")
    assert annotation.labels == ["wall"]


# %% root placement


def test_parse_places_the_root_on_the_ground_below_the_center_of_the_scene():
    # The fixture's geometry spans (0, 0, 0) to (12, 3, 3) and the stage is Z-up,
    # so the root goes to (6, 1.5, 0) - centred across, but down on the ground
    # rather than floating at half the scene's height.
    world = parse(
        build_scene_stage_with_grouped_instances(),
        root_placement=RootPlacement.SCENE_GROUND,
    )

    np.testing.assert_allclose(
        translation_of(connection_to(world, "wall_a")), [5.0, -1.5, 0.0], atol=1e-6
    )


def test_parse_moves_the_geometry_the_root_prim_owns():
    # The root body's own shapes are placed against the same root, so moving the
    # root has to move them too. That stage is the default Y-up and its quad spans
    # (0, 0, 0) to (1, 1, 0), so the root goes to (0.5, 0, 0).
    world = parse(
        build_jointless_stage_with_a_default_prim(),
        root_placement=RootPlacement.SCENE_GROUND,
    )

    [shape] = world.root.visual.shapes
    np.testing.assert_allclose(
        shape.unscaled_mesh.vertices.min(axis=0), [-0.5, 0.0, 0.0], atol=1e-5
    )


def test_parse_records_where_the_stage_origin_lies_after_moving_the_root():
    # Moving the root is only reversible if the world says how far it moved.
    world = parse(
        build_scene_stage_with_grouped_instances(),
        root_placement=RootPlacement.SCENE_GROUND,
    )

    annotation = stage_origin_of(world)
    assert annotation.root is world.root
    np.testing.assert_allclose(
        annotation.position.to_np()[:3].flatten(), [-6.0, -1.5, 0.0], atol=1e-6
    )


def test_parse_records_the_stage_origin_at_the_root_without_moving_it():
    world = parse(build_scene_stage_with_grouped_instances())

    np.testing.assert_allclose(
        stage_origin_of(world).position.to_np()[:3].flatten(),
        [0.0, 0.0, 0.0],
        atol=1e-6,
    )


def test_parse_finds_the_ground_along_the_stages_own_up_axis():
    # The same geometry on a Y-up stage has its ground at minimum y instead, putting
    # the root at (6, 0, 1.5) so wall_a lands (5, 0, -1.5) from it.
    world = parse(
        build_scene_stage_with_grouped_instances(up_axis="Y"),
        root_placement=RootPlacement.SCENE_GROUND,
    )

    np.testing.assert_allclose(
        translation_of(connection_to(world, "wall_a")), [5.0, 0.0, -1.5], atol=1e-6
    )


# %% shading


def test_parse_draws_the_scene_unlit_when_asked(tmp_path):
    # Only a textured surface can be drawn unlit, since it is the texture that carries
    # the brightness.
    texture_file = tmp_path / "wood.png"
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(texture_file)

    world = USDSceneParser(
        stage=build_stage_with_textured_mesh(str(texture_file)),
        prefix="scene",
        shading=Shading.UNLIT,
    ).parse()

    [shape] = world.root.visual.shapes
    np.testing.assert_array_equal(
        shape.unscaled_mesh.visual.material.emissiveFactor, [1.0, 1.0, 1.0]
    )


# %% collision geometry


def test_parse_leaves_what_a_scene_collides_as_to_a_decomposition_step():
    # What a scene is collided against is a MeshDecomposer step's choice to make, so
    # the parser reports the surfaces the stage holds and nothing else.
    world = parse(build_scene_stage_with_grouped_instances())
    wall = body_named(world, "wall_a")

    [visual] = wall.visual.shapes
    assert isinstance(visual, Mesh)
    assert wall.collision.shapes == []


def test_parse_leaves_the_geometry_the_root_prim_owns_uncollided():
    world = parse(build_jointless_stage_with_a_default_prim())

    assert world.root.visual.shapes != []
    assert world.root.collision.shapes == []


def test_parse_does_not_read_back_the_surfaces_it_writes():
    # A scanned stage becomes unloadable if every surface is read into memory again,
    # which is what reaches the collision detector the moment one is collided against.
    world = parse(build_scene_stage_with_grouped_instances())

    surfaces = [shape for body in world.bodies for shape in body.visual]
    assert surfaces
    assert all("mesh" not in surface.__dict__ for surface in surfaces)


def test_a_parsed_scene_can_be_enclosed_in_boxes():
    world = parse(build_scene_stage_with_grouped_instances())
    [visual] = body_named(world, "wall_a").visual.shapes
    low, high = visual.bounds

    Pipeline([BoundingBoxDecomposer()]).apply(world)

    [collision] = body_named(world, "wall_a").collision.shapes
    assert isinstance(collision, Box)
    np.testing.assert_allclose(collision.scale.to_np(), high - low, atol=1e-6)


def test_parse_leaves_a_guide_out_of_what_an_object_looks_like():
    # A guide is geometry a renderer draws nothing for, so a collision proxy authored
    # beside a surface is not part of what that object looks like.
    world = parse(build_scene_stage_with_a_guide_prim())

    [shape] = body_named(world, "wall_a").visual.shapes
    assert isinstance(shape, Mesh)


def test_parse_gives_no_body_to_a_prim_holding_nothing_but_guides():
    # A collision box written beside the surface it stands for owns no geometry a
    # renderer draws, so the prim holding it is not an object of the scene.
    world = parse(build_scene_stage_with_a_guide_under_a_prim_of_its_own())

    assert [body.name.name for body in world.bodies if body is not world.root] == [
        "wall_a"
    ]


def test_parse_reads_the_collision_a_stage_authors_into_the_body_it_belongs_to():
    # An asset library writes what a physics engine collides an object against as
    # guide boxes beside its surface; the body is collided as exactly those boxes.
    world = parse(build_scene_stage_with_authored_collision())
    wall = body_named(world, "wall_a")

    [visual] = wall.visual.shapes
    assert isinstance(visual, Mesh)
    first, second = wall.collision.shapes
    assert isinstance(first, Box) and isinstance(second, Box)
    np.testing.assert_allclose(first.scale.to_np(), [4.0, 0.2, 2.5], atol=1e-6)
    np.testing.assert_allclose(first.origin.to_np()[:3, 3], [2.0, 0.0, 1.25], atol=1e-6)
    np.testing.assert_allclose(
        first.origin.to_np()[:3, :3],
        [
            [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
            [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
            [0, 0, 1],
        ],
        atol=1e-6,
    )


def test_parse_gives_a_nested_object_its_own_collision_and_not_its_holder_s():
    world = parse(build_scene_stage_with_authored_collision())

    [leaf_box] = body_named(world, "door_0").collision.shapes
    np.testing.assert_allclose(leaf_box.scale.to_np(), [1.0, 0.04, 2.0], atol=1e-6)
    assert len(body_named(world, "wall_a").collision.shapes) == 2


def test_parse_collides_against_a_guide_only_when_the_stage_says_so():
    # A guide nothing collides against stays what it was: not part of the object.
    world = parse(build_scene_stage_with_a_guide_prim())

    assert body_named(world, "wall_a").collision.shapes == []


# %% naming the bodies a scene is made of


def test_parse_names_every_body_so_it_can_be_addressed():
    # An asset library names every object's geometry container the same, so the leaf
    # name alone leaves most of a scene impossible to look up.
    world = parse(build_scene_stage_with_repeated_container_names())

    names = [body.name.name for body in world.bodies if body is not world.root]

    assert sorted(names) == ["chair_Actor_0000_Geom", "sofa_Actor_0000_Geom"]


def test_parse_keeps_a_name_short_when_it_is_already_unique():
    world = parse(build_scene_stage_with_grouped_instances())

    names = [body.name.name for body in world.bodies if body is not world.root]

    assert sorted(names) == ["floor_a", "wall_a", "wall_b"]


# %% objects a stage states a joint for


def _parsed(stage) -> World:
    """
    :param stage: The stage to read.
    :return: The world it becomes, rooted where it stands.
    """
    return USDSceneParser(stage=stage, prefix="test").parse()


def _leaf_of(world: World) -> Body:
    """
    :param world: The parsed world.
    :return: The body the stage hangs on a hinge.
    """
    return next(body for body in world.bodies if "door_0" in body.name.name)


def test_a_revolute_joint_becomes_a_connection_that_turns():
    world = _parsed(build_scene_stage_with_a_door_on_a_hinge())

    [hinged] = [
        connection
        for connection in world.connections
        if isinstance(connection, RevoluteConnection)
    ]

    assert hinged.child is _leaf_of(world)
    assert "surface" in hinged.parent.name.name


def test_a_revolute_joint_keeps_the_limits_the_stage_states():
    world = _parsed(build_scene_stage_with_a_door_on_a_hinge(lower=-90.0, upper=5.0))

    [hinged] = [
        connection
        for connection in world.connections
        if isinstance(connection, RevoluteConnection)
    ]
    limits = hinged.dof.limits

    # USD states a revolute limit in degrees and a world holds it in radians.
    assert np.isclose(limits.lower.position, np.radians(-90.0))
    assert np.isclose(limits.upper.position, np.radians(5.0))


def test_an_object_with_no_joint_is_still_fixed_where_it_stands():
    world = _parsed(build_scene_stage_with_a_door_on_a_hinge())

    surface = next(body for body in world.bodies if "surface" in body.name.name)

    assert isinstance(surface.parent_connection, FixedConnection)


def test_a_leaf_stands_where_the_stage_puts_it_until_it_is_turned():
    world = _parsed(build_scene_stage_with_a_door_on_a_hinge())
    leaf = _leaf_of(world)

    resting = leaf.visual.shapes[0].mesh_in_frame(world.root).bounds

    # The stage places the leaf a metre along, where its own quad then reaches a metre
    # further; a joint holding it must leave it exactly there.
    assert np.allclose(resting[0], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(resting[1], [2.0, 1.0, 0.0], atol=1e-6)


def test_turning_a_leaf_leaves_its_hinge_where_it_was():
    world = _parsed(build_scene_stage_with_a_door_on_a_hinge())
    leaf = _leaf_of(world)
    [hinged] = [
        connection
        for connection in world.connections
        if isinstance(connection, RevoluteConnection)
    ]
    before = np.asarray(leaf.visual.shapes[0].mesh_in_frame(world.root).vertices)

    world.state[hinged.dof.id].position = np.radians(-45.0)
    world.notify_state_change()
    after = np.asarray(leaf.visual.shapes[0].mesh_in_frame(world.root).vertices)

    moved = np.linalg.norm(after - before, axis=1)
    # The leaf turns about its own origin, which the stage anchored the joint at, so
    # the corner standing there is the one that does not move.
    assert moved.min() < 1e-9
    assert moved.max() > 0.1
