"""
The scenes index of the published demo site.

The demo site ships a few recordings out of the shared cram-scenes collection. Its index
lists exactly those and names the one to open first, so the viewer's pickers offer only
scenes the site actually carries. Runnable as a script on a machine without cramera
installed, which is what the Pages workflow does.

Usage::

    python3 src/cramera/demo_site.py <collection index> <site index> <default> <scene>...
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing_extensions import Any, Dict, List, Self


class SceneMissing(KeyError):
    """
    A scene the site should carry is not in the collection's index.
    """


@dataclass
class IndexedScene:
    """
    One entry of a scenes index: a recording and the three axes the pickers group it by.
    """

    name: str
    """
    Directory name of the scene bundle.
    """

    robot: str
    """
    Robot the recording shows.
    """

    environment: str
    """
    Environment the recording was made in.
    """

    task: str
    """
    Task the robot performed.
    """

    @classmethod
    def from_json(cls, entry: Dict[str, Any]) -> Self:
        return cls(
            name=entry["name"],
            robot=entry["robot"],
            environment=entry["environment"],
            task=entry["task"],
        )


@dataclass
class ScenesIndex:
    """
    A scenes index as the viewer reads it: the scenes on offer and the one to open first.
    """

    default: str
    """
    Name of the scene the viewer opens without an explicit ``?scene=``.
    """

    scenes: List[IndexedScene]
    """
    The scenes on offer, in picker order.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> Self:
        return cls(
            default=payload["default"],
            scenes=[IndexedScene.from_json(entry) for entry in payload["scenes"]],
        )

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))

    def to_json(self) -> Dict[str, Any]:
        return {
            "default": self.default,
            "scenes": [asdict(scene) for scene in self.scenes],
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n", encoding="utf-8")

    def restricted_to(self, default: str, names: List[str]) -> ScenesIndex:
        """
        The index of a site carrying only ``names``, opening ``default`` first.

        :raises SceneMissing: when a requested scene is not in this index.
        """
        by_name = {scene.name: scene for scene in self.scenes}
        missing = [name for name in [default, *names] if name not in by_name]
        if missing:
            raise SceneMissing(missing)
        return ScenesIndex(default=default, scenes=[by_name[name] for name in names])


def main(arguments: List[str]) -> None:
    """
    Write the demo site's index from the collection's index and the chosen scenes.
    """
    collection, site, default, *names = arguments
    ScenesIndex.read(Path(collection)).restricted_to(default, names).write(Path(site))


if __name__ == "__main__":
    main(sys.argv[1:])
