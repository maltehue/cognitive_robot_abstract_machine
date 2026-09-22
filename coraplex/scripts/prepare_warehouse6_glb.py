"""
Turn a raw ``warehouse6.fbx`` into ``resources/worlds/warehouse6.glb``, a coraplex
world resource a demo can load with ``BodySpecification.mesh``.

The raw model, like most game assets, is not ready as it comes:

* it is modelled in centimetres, Y-up -- coraplex works in metres, Z-up
* it is roofed, and a roof hides the hall from every camera above it
* it holds two million faces, more than a viewer wants to carry
* its colour lives in per-part materials, which a decimated mesh would drop

This reads the parts, moves them into metres and Z-up, cuts the roof off, decimates
each part while baking its material colour onto the faces, and writes one GLB.

FBX is Autodesk's own format, which trimesh does not read; convert it to glTF first::

    assimp export warehouse6.fbx warehouse6.glb

Then, from the repository root::

    python coraplex/scripts/prepare_warehouse6_glb.py --src warehouse6.glb
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

CENTIMETRES_PER_METRE = 100.0
"""What a game asset modelled in centimetres has to be divided by."""

ROOF_HEIGHT_METRES = 3.2
"""Height the hall is cut off at: above it are only the roof trusses and the roof."""

FACE_BUDGET = 250_000
"""Faces the hall is decimated towards; the raw scan holds two million."""

KEEP_SMALL_UNDER = 200
"""Parts smaller than this are left as they are: decimating them buys nothing."""

FLOOR_RGBA = np.array([150, 150, 152, 255], np.uint8)
"""
The colour the floor is painted: a light concrete grey.

The model's own floor is modelled all but black, which reads as a hole rather than as
ground; a light grey gives the run a floor to stand on. Change this to repaint it.
"""

FLOOR_MAX_THICKNESS_METRES = 0.1
"""How thin a part has to be, in Z, to be taken for the floor rather than a wall."""

RACK_VALUE_JITTER = 0.22
"""
How far the brightness of the racked goods is spread, up and down.

The racks come as one mesh of thousands of near-identical brown crates, which reads as
one flat brown block; giving each crate its own shade breaks that up. This is the full
width of the spread in HSV value, so each crate lands somewhere within +/- half of it.
"""

RACK_HUE_JITTER = 0.035
"""The same, for hue: a small warm/cool wander so the browns are not all the one tint."""


def is_brownish(rgba: np.ndarray) -> bool:
    """
    Whether a colour is a brown: warm (red the strongest channel), clearly not grey, and
    not so bright it is really an orange or a tan.

    :param rgba: The colour to judge, 0-255.
    """
    red, green, blue = (int(c) for c in rgba[:3])
    return red > green > blue and red >= 40 and (red - blue) >= 20 and red <= 200


def rack_index(parts) -> int:
    """
    Which part is the racked goods: the brown part with the most faces. Returns ``-1``
    when nothing looks brown.

    :param parts: The parts of the hall, in metres and Z-up.
    """
    best, best_faces = -1, 0
    for index, part in enumerate(parts):
        if is_brownish(base_rgba(part)) and len(part.faces) > best_faces:
            best, best_faces = index, len(part.faces)
    return best


def shaded_face_colours(mesh: trimesh.Trimesh, rgba: np.ndarray) -> np.ndarray:
    """
    Per-face colours that give every connected piece of ``mesh`` its own shade of the
    base colour, so a mesh of many identical crates stops reading as one flat block.

    The spread is deterministic -- a low-discrepancy walk keyed on the piece number, not
    a random draw -- so the same mesh always comes out shaded the same way.

    :param mesh: The mesh whose pieces are shaded.
    :param rgba: The base colour every piece is a shade of, 0-255.
    """
    hue, saturation, value = colorsys.rgb_to_hsv(*(rgba[:3] / 255.0))
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces)
    )
    colours = np.empty((len(mesh.faces), 4), np.uint8)
    colours[:, 3] = 255
    for piece in np.unique(labels):
        walk_value = (piece * 0.6180339887) % 1.0
        walk_hue = (piece * 0.7548776662) % 1.0
        shade_value = np.clip(value + (walk_value - 0.5) * RACK_VALUE_JITTER, 0.1, 0.9)
        shade_hue = (hue + (walk_hue - 0.5) * RACK_HUE_JITTER) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(shade_hue, saturation, shade_value)
        colours[labels == piece, :3] = np.round(
            np.array([red, green, blue]) * 255.0
        ).astype(np.uint8)
    return colours


def floor_index(parts) -> int:
    """
    Which part is the floor: the widest part in its X/Y footprint that is thin in Z and
    sits on the ground. Returns ``-1`` when nothing looks like a floor.

    :param parts: The parts of the hall, in metres and Z-up.
    """
    best, best_area = -1, 0.0
    for index, part in enumerate(parts):
        low, high = part.bounds
        extent = high - low
        if extent[2] > FLOOR_MAX_THICKNESS_METRES or low[2] > FLOOR_MAX_THICKNESS_METRES:
            continue
        area = float(extent[0] * extent[1])
        if area > best_area:
            best, best_area = index, area
    return best


def base_rgba(geom: trimesh.Trimesh) -> np.ndarray:
    """
    The one flat colour of a part: its glTF ``baseColorFactor``, or, if it carries none,
    the mean of its vertex colours.

    :param geom: The part to read the colour of.
    """
    material = getattr(geom.visual, "material", None)
    if material is not None:
        colour = getattr(material, "baseColorFactor", None)
        if colour is None:
            colour = getattr(material, "main_color", None)
        if colour is not None:
            colour = np.asarray(colour, dtype=float).ravel()
            if colour.max() <= 1.0 + 1e-6:
                colour = colour * 255.0
            rgba = np.full(4, 255, np.uint8)
            rgba[: len(colour)] = np.clip(colour, 0, 255).astype(np.uint8)
            return rgba
    try:
        return np.asarray(geom.visual.to_color().vertex_colors).mean(0).astype(np.uint8)
    except Exception:
        return np.array([180, 180, 180, 255], np.uint8)


def prepared_hall(source: Path) -> trimesh.Trimesh:
    """
    The hall in metres and Z-up, with its roof cut off, decimated towards the budget and
    each part painted the colour of the material it was modelled with.

    :param source: The glTF the FBX was converted to.
    """
    scene = trimesh.load(source)
    to_metres_z_up = trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]
    ) @ trimesh.transformations.scale_matrix(1.0 / CENTIMETRES_PER_METRE)
    scene.apply_transform(to_metres_z_up)

    parts = scene.dump(concatenate=False)
    total_faces = sum(len(part.faces) for part in parts)
    floor = floor_index(parts)
    rack = rack_index(parts)

    walls = []
    for index, part in enumerate(parts):
        rgba = FLOOR_RGBA if index == floor else base_rgba(part)
        below_roof = part.slice_plane(
            plane_origin=[0, 0, ROOF_HEIGHT_METRES],
            plane_normal=[0, 0, -1],
            cap=True,
        )
        if below_roof is None or len(below_roof.faces) == 0:
            continue
        target = max(int(FACE_BUDGET * len(below_roof.faces) / total_faces), 4)
        if len(below_roof.faces) > KEEP_SMALL_UNDER and target < len(below_roof.faces):
            vertices, faces = fast_simplification.simplify(
                np.asarray(below_roof.vertices, np.float64),
                np.asarray(below_roof.faces, np.int64),
                target_count=target,
            )
            below_roof = trimesh.Trimesh(vertices, faces, process=False)
        if index == rack:
            face_colours = shaded_face_colours(below_roof, rgba)
        else:
            face_colours = np.tile(rgba, (len(below_roof.faces), 1))
        below_roof.visual = trimesh.visual.ColorVisuals(
            below_roof, face_colors=face_colours
        )
        walls.append(below_roof)

    return trimesh.util.concatenate(walls)


def main() -> None:
    default_out = (
        Path(__file__).resolve().parents[1] / "resources" / "worlds" / "warehouse6.glb"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="the warehouse6 model, converted from FBX to glTF with assimp",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="where to write the prepared world resource",
    )
    arguments = parser.parse_args()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)

    hall = prepared_hall(arguments.src)
    hall.export(arguments.out)

    print(
        "%s  %d faces  %s m  bottom %.3f m below its origin"
        % (
            arguments.out.name,
            len(hall.faces),
            [round(float(v), 3) for v in hall.extents],
            round(-float(hall.bounds[0][2]), 3),
        )
    )


if __name__ == "__main__":
    main()
