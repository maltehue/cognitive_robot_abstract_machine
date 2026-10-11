---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

(loading-worlds)=
# Loading worlds from files

This tutorial shows how to load a world description from a file into a `World` object using the `URDFParser`.

First, we need to compose the path to your world file.

```{code-cell} ipython3
import logging
import os

from importlib.resources import files
from pathlib import Path

logging.disable(logging.CRITICAL)
apartment = os.path.join(Path(files("semantic_digital_twin")).parent.parent, "resources", "urdf", "apartment.urdf")

```

Next we need to initialize a parser that reads this file. There are many parsers available. 
To read this specific urdf file, the `https://github.com/code-iai/iai_maps/tree/ros-jazzy/` repository needs to be installed
inside your ROS2 workspace.

```{code-cell} ipython3
from semantic_digital_twin.adapters.urdf import URDFParser  
  
parser = URDFParser.from_file(apartment)  
world = parser.parse()  
```

This constructs a world you can visualize, interact and annotate. Be aware that worlds loaded from files have no semantic annotations and serve as purely kinematic models.
Supported file formates are:
- urdf
- mjcf
- stl


## Separate USD scene objects

Use `USDSceneParser.from_file(path).parse()` to import static scene objects as
separate bodies. Body names retain their full prim paths, with the initial slash
removed. Model components and rigid bodies own their internal geometry; otherwise
the enclosing geometry prim's parent defines the object boundary.

Grouping transforms and geometry scale are preserved. Coordinates remain in the
stage's authored units and axes. Fixed, revolute and prismatic physics joints retain their authored local anchors, axes and limits.
`USDParser` retains its asset-import behavior.


`USDSceneParser` keeps visual geometry separate from `UsdPhysics.CollisionAPI`
geometry. Guide colliders remain available for collision checks without becoming
visible. When an object has no authored collider, its visible geometry is also
used for collision. An explicitly disabled collider suppresses this fallback.


Scene articulations start at zero joint position. Apply configured initial positions
through the world's state API. Unsupported joints, invalid endpoints, multiple
parents, closed loops and scaled joint frames raise explicit parsing errors.
Joint endpoints remain separate bodies even within one model component.


Scene bodies retain labels authored on themselves and their ancestors. Object labels
also establish geometry ownership when model-kind metadata is absent. The default
`UsdSceneSemantics` maps `floor` to `Floor` and `table`/`workbench` to `Table`,
case-insensitively; the nearest recognized label takes precedence. Unknown vocabulary
is retained as `UsdSemanticLabels`. Supply a different `support_labels` mapping to
interpret another vocabulary. Joint type alone does not classify furniture.
