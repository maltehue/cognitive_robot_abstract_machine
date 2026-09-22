"""
Generate a labelled transport-box mesh for the object library.

Writes an ``.obj`` with its ``.mtl`` and texture next to the other object meshes, so a
demo can carry a printed cardboard box instead of an unlabelled shape::

    python scripts/make_transport_box_mesh.py --name screw_box --label SCREWS \\
        --caption "M6 x 60" --caption "100 pcs" \\
        --like coraplex/resources/objects/milk.stl

``--like`` takes the bounding box of an existing mesh, so the new box drops into poses
that were captured for that one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import ClassVar, List, Tuple

KRAFT = (176, 132, 84)
"""
Cardboard brown the box is printed on.
"""

KRAFT_SHADE = (154, 113, 70)
"""
Slightly darker kraft, for the fibre lines and the fold along each edge.
"""

LABEL_WHITE = (243, 240, 234)
"""
Off-white of the printed label.
"""

INK = (38, 40, 46)
"""
Near-black the label is printed in.
"""

ACCENT = (32, 96, 168)
"""
Blue bar across the top of the label.
"""

TAPE = (198, 176, 140)
"""
Packing tape across the lid.
"""

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
"""
The bold face the label is set in.
"""


@dataclass
class BoundingBox:
    """
    The extent a box occupies in its own frame.
    """

    minimum: Tuple[float, float, float]
    """
    Lowest corner, in metres.
    """

    maximum: Tuple[float, float, float]
    """
    Highest corner, in metres.
    """

    @classmethod
    def of_mesh(cls, path: Path) -> BoundingBox:
        """
        The bounding box of an existing mesh file.

        :param path: The mesh whose extent is copied.
        """
        bounds = trimesh.load(str(path), force="mesh").bounds
        return cls(tuple(bounds[0]), tuple(bounds[1]))

    def corners(self) -> np.ndarray:
        """
        The eight corners, ordered so :attr:`TransportBox.FACES` indexes into them.
        """
        (x0, y0, z0), (x1, y1, z1) = self.minimum, self.maximum
        return np.array(
            [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
        )


@dataclass
class TransportBox:
    """
    A cardboard box carrying a printed label, written out as OBJ + MTL + texture.
    """

    name: str
    """
    Base name of the three files written.
    """

    label: str
    """
    The word printed large on the label.
    """

    captions: List[str] = field(default_factory=list)
    """
    Smaller lines printed under the label.
    """

    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox((-0.03, -0.03, -0.09), (0.03, 0.03, 0.11))
    )
    """
    The extent the box fills, in its own frame.
    """

    texture_size: int = 1024
    """
    Edge length of the square texture.
    """

    FACES: ClassVar[
        List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int]]]
    ] = [
        ("side", (0, 1, 5, 4), (0, -1, 0)),
        ("side", (1, 2, 6, 5), (1, 0, 0)),
        ("side", (2, 3, 7, 6), (0, 1, 0)),
        ("side", (3, 0, 4, 7), (-1, 0, 0)),
        ("lid", (4, 5, 6, 7), (0, 0, 1)),
        ("lid", (3, 2, 1, 0), (0, 0, -1)),
    ]
    """
    The six faces: which region of the texture each shows, its corners, and its outward
    normal -- a mesh without normals is rendered unlit, which comes out black.
    """

    REGIONS: ClassVar[dict] = {
        "side": (0.0, 0.0, 0.5, 1.0),
        "lid": (0.5, 0.5, 1.0, 1.0),
    }
    """
    Texture regions as ``(u0, v0, u1, v1)``: the printed panel and the taped lid.
    """

    def texture(self) -> Image.Image:
        """
        The printed cardboard the box is wrapped in.
        """
        size = self.texture_size
        image = Image.new("RGB", (size, size), KRAFT)
        canvas = ImageDraw.Draw(image)
        for y in range(0, size, 7):  # fibre lines, so the cardboard is not flat colour
            canvas.line([(0, y), (size, y)], fill=KRAFT_SHADE, width=1)
        self._draw_panel(canvas, self.REGIONS["side"])
        self._draw_lid(canvas, self.REGIONS["lid"])
        return image

    def _pixels(self, region: Tuple[float, float, float, float]) -> Tuple[int, ...]:
        """
        One texture region in pixels, as ``(left, top, right, bottom)``.

        :param region: The region in texture coordinates.
        """
        u0, v0, u1, v1 = region
        size = self.texture_size
        # texture v runs up from the bottom, pixel y runs down from the top
        return (
            round(u0 * size),
            round((1.0 - v1) * size),
            round(u1 * size),
            round((1.0 - v0) * size),
        )

    def _draw_panel(self, canvas: ImageDraw.ImageDraw, region) -> None:
        """
        The label the four sides show.

        :param canvas: What to draw on.
        :param region: The texture region the sides map to.
        """
        left, top, right, bottom = self._pixels(region)
        width, height = right - left, bottom - top
        margin = round(width * 0.1)
        label_box = (
            left + margin,
            top + round(height * 0.28),
            right - margin,
            bottom - round(height * 0.28),
        )
        canvas.rectangle(label_box, fill=LABEL_WHITE, outline=INK, width=3)
        canvas.rectangle(
            (
                label_box[0],
                label_box[1],
                label_box[2],
                label_box[1] + round(height * 0.05),
            ),
            fill=ACCENT,
        )
        label_font = self._font(round(width * 0.16))
        caption_font = self._font(round(width * 0.075))
        text_y = label_box[1] + round(height * 0.09)
        self._centred(canvas, self.label.upper(), label_font, label_box, text_y)
        text_y += round(width * 0.19)
        for caption in self.captions:
            self._centred(canvas, caption, caption_font, label_box, text_y)
            text_y += round(width * 0.1)

    def _draw_lid(self, canvas: ImageDraw.ImageDraw, region) -> None:
        """
        The taped lid the top and bottom show.

        :param canvas: What to draw on.
        :param region: The texture region the lid faces map to.
        """
        left, top, right, bottom = self._pixels(region)
        width = right - left
        middle = (top + bottom) // 2
        tape_half = round(width * 0.09)
        canvas.rectangle(
            (left, middle - tape_half, right, middle + tape_half),
            fill=TAPE,
            outline=KRAFT_SHADE,
        )
        canvas.line([(left, top), (right, bottom)], fill=KRAFT_SHADE, width=2)
        canvas.line([(left, bottom), (right, top)], fill=KRAFT_SHADE, width=2)
        font = self._font(round(width * 0.11))
        self._centred(
            canvas,
            self.label.upper(),
            font,
            (left, top, right, bottom),
            top + round(width * 0.12),
        )

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont:
        """
        The label face at one size, falling back to Pillow's built-in face.

        :param size: Height in pixels.
        """
        if FONT_PATH.is_file():
            return ImageFont.truetype(str(FONT_PATH), size)
        return ImageFont.load_default(size)

    @staticmethod
    def _centred(canvas, text: str, font, box, y: int) -> None:
        """
        Draw one line centred horizontally in a box.

        :param canvas: What to draw on.
        :param text: The line to draw.
        :param font: The face to set it in.
        :param box: The box to centre within, as ``(left, top, right, bottom)``.
        :param y: Where the line's top edge goes.
        """
        left, _, right, _ = box
        length = canvas.textlength(text, font=font)
        canvas.text((left + (right - left - length) / 2, y), text, font=font, fill=INK)

    def obj_text(self) -> str:
        """
        The box as an OBJ, with one texture coordinate per face corner.
        """
        lines = [
            "# %s -- generated by scripts/make_transport_box_mesh.py" % self.name,
            "mtllib %s.mtl" % self.name,
            "o %s" % self.name,
        ]
        for corner in self.bounding_box.corners():
            lines.append("v %.6f %.6f %.6f" % tuple(corner))
        for region_name, _, _ in self.FACES:
            u0, v0, u1, v1 = self.REGIONS[region_name]
            for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                lines.append("vt %.6f %.6f" % (u, v))
        for _, _, normal in self.FACES:
            lines.append("vn %d %d %d" % normal)
        lines.append("usemtl %s" % self.name)
        for face_index, (_, corners, _) in enumerate(self.FACES):
            texture_indices = [face_index * 4 + offset + 1 for offset in range(4)]
            lines.append(
                "f "
                + " ".join(
                    "%d/%d/%d" % (corner + 1, texture_index, face_index + 1)
                    for corner, texture_index in zip(corners, texture_indices)
                )
            )
        return "\n".join(lines) + "\n"

    def mtl_text(self) -> str:
        """
        The material naming the texture.
        """
        return (
            "\n".join(
                [
                    "newmtl %s" % self.name,
                    "Ka 1.000 1.000 1.000",
                    "Kd 1.000 1.000 1.000",
                    "Ks 0.000 0.000 0.000",
                    "d 1.0",
                    "illum 1",
                    "map_Kd %s.png" % self.name,
                ]
            )
            + "\n"
        )

    def write(self, directory: Path) -> List[Path]:
        """
        Write the three files and return their paths.

        :param directory: Where the object meshes live.
        """
        written = []
        for suffix, content in ((".obj", self.obj_text()), (".mtl", self.mtl_text())):
            path = directory / (self.name + suffix)
            path.write_text(content, encoding="utf-8")
            written.append(path)
        texture_path = directory / (self.name + ".png")
        self.texture().save(texture_path)
        written.append(texture_path)
        return written


def parse_arguments() -> argparse.Namespace:
    """
    What this script accepts on its command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="base name of the written files")
    parser.add_argument(
        "--label", required=True, help="word printed large on the label"
    )
    parser.add_argument(
        "--caption", action="append", default=[], help="smaller line under the label"
    )
    parser.add_argument(
        "--like", type=Path, help="mesh whose bounding box the box takes over"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("coraplex/resources/objects"),
        help="directory to write into (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Write one labelled transport box.
    """
    arguments = parse_arguments()
    box = TransportBox(
        name=arguments.name, label=arguments.label, captions=arguments.caption
    )
    if arguments.like:
        box.bounding_box = BoundingBox.of_mesh(arguments.like)
    for path in box.write(arguments.out):
        print("wrote", path)


if __name__ == "__main__":
    main()
