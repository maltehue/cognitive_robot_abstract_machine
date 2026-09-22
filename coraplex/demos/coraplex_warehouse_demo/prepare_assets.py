"""
Prepare the warehouse demo's meshes from the collections they were downloaded as.

Three things the downloads do not come ready for:

* the hall is roofed, and a roof hides the run from every camera above it
* every crate in the collection is closed, and the run takes something out of one
* the wrench is modelled in centimetres, like most game assets

Run once, from the repository root::

    python coraplex/demos/coraplex_warehouse_demo/prepare_assets.py --downloads ~/Downloads
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d
import trimesh
from typing_extensions import List

ROOF_HEIGHT_METRES = 3.2
"""
Height the hall is cut off at: above it are only the roof trusses and the roof.
"""

WAREHOUSE_FACE_BUDGET = 250_000
"""
Faces the hall is simplified to, so a viewer can carry it: the raw scan holds two
million, of which the roof alone is six hundred thousand.
"""

CENTIMETRES_PER_METRE = 100.0
"""
What a game asset modelled in centimetres has to be divided by.
"""

CRATE_WIDTH_METRES = 0.22
"""
Width the crate is scaled to.

The collection models a pallet crate of 47 cm, which no two- or three-finger hand can
take hold of -- the G1's is 12 cm across. At 22 cm a hand can grasp an edge, and a 20 cm
wrench still fits inside.
"""

LID_THICKNESS_METRES = 0.05
"""
How far below its rim a crate is opened: the lid sits within two to three centimetres
of it, and taking that much off leaves the walls standing.
"""


@dataclass(frozen=True)
class PreparedMesh:
    """
    One mesh the demo loads, and where it came from.
    """

    name: str
    """
    File name the demo loads it under.
    """

    faces: int
    """
    Triangles it ends up with.
    """

    extents: List[float]
    """
    Its size in metres, x/y/z.
    """

    bottom_below_origin: float
    """
    How far its lowest point sits below its own origin, in metres.

    What a demo has to lift a body by for it to rest on a surface rather than sink into
    it: a mesh is modelled around whatever origin its author chose.
    """


def unroofed_hall(source: Path, budget: int) -> trimesh.Trimesh:
    """
    The hall with its roof cut off and simplified enough to be carried around.

    The cut is capped: an uncapped one leaves every wall an open shell, which reads as
    scenery rather than as a building.

    :param source: The scanned hall.
    :param budget: Faces to simplify down to.
    """
    hall = trimesh.load(source, force="mesh")
    walls = hall.slice_plane(
        plane_origin=[0, 0, ROOF_HEIGHT_METRES],
        plane_normal=[0, 0, -1],
        cap=True,
    )
    return simplified(walls, budget)


def simplified(mesh: trimesh.Trimesh, budget: int) -> trimesh.Trimesh:
    """
    The mesh with at most ``budget`` faces, or as it is when it already has fewer.

    :param mesh: The mesh to simplify.
    :param budget: Faces to simplify down to.
    """
    if len(mesh.faces) <= budget:
        return mesh
    geometry = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        open3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    reduced = geometry.simplify_quadric_decimation(target_number_of_triangles=budget)
    return trimesh.Trimesh(
        vertices=np.asarray(reduced.vertices), faces=np.asarray(reduced.triangles)
    )


def opened_crate(source: Path) -> trimesh.Trimesh:
    """
    A crate with its lid taken off, so what is inside can be seen and reached.

    :param source: The closed crate.
    """
    crate = trimesh.load(source, force="mesh")
    rim = crate.bounds[1][2] - LID_THICKNESS_METRES
    opened = crate.slice_plane(plane_origin=[0, 0, rim], plane_normal=[0, 0, -1])
    opened.apply_scale(CRATE_WIDTH_METRES / float(opened.extents[0]))
    return opened


def metric(source: Path, divisor: float) -> trimesh.Trimesh:
    """
    A mesh modelled in another unit, in metres.

    :param source: The mesh as it was modelled.
    :param divisor: What its units go into a metre.
    """
    mesh = trimesh.load(source, force="mesh")
    mesh.apply_scale(1.0 / divisor)
    return mesh


def extracted(archive: Path, member: str, into: Path) -> Path:
    """
    One member of a downloaded archive, on disk.

    :param archive: The archive to read.
    :param member: Path of the member inside it.
    :param into: Directory to extract into.
    """
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open(member) as source:
            target = into / Path(member).name
            target.write_bytes(source.read())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downloads",
        type=Path,
        default=Path.home() / "Downloads",
        help="directory the collections were downloaded to",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "resources",
        help="directory to write the prepared meshes to",
    )
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    staging = arguments.out / "_downloaded"
    staging.mkdir(exist_ok=True)

    crates = (
        arguments.downloads
        / "Stylized_Pallet_Trucks_Hand_Trucks_Crates_Collection_GLB_v1_0.zip"
    )
    crate_source = extracted(
        crates, "GLB/SM_ICrate_Wood/SM_ICrate_Wood_Brown_V01.glb", staging
    )
    wrench_source = extracted(
        arguments.downloads / "PB114_wrench_SM.obj.zip", "PB114_wrench_SM.obj", staging
    )

    prepared = []
    # the crate is written as glTF, which carries the wood texture it was modelled with;
    # the hall and the wrench were downloaded without one, and STL says as much
    for name, mesh in [
        (
            "warehouse_hall.stl",
            unroofed_hall(arguments.downloads / "warehouse6.stl", WAREHOUSE_FACE_BUDGET),
        ),
        ("open_crate.glb", opened_crate(crate_source)),
        ("wrench.stl", metric(wrench_source, CENTIMETRES_PER_METRE)),
    ]:
        mesh.export(arguments.out / name)
        prepared.append(
            PreparedMesh(
                name,
                len(mesh.faces),
                [round(float(v), 3) for v in mesh.extents],
                round(-float(mesh.bounds[0][2]), 3),
            )
        )

    for entry in prepared:
        print(
            "%-22s %8d faces  %s m  bottom %.3f m below its origin"
            % (entry.name, entry.faces, entry.extents, entry.bottom_below_origin)
        )


if __name__ == "__main__":
    main()
