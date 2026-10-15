# Real pouring 2026-10-15: counting on the receiver's rim

The paced corn pour of 2026-10-15 (`new-pouring-corn.zip`; 313 g received, 48.8 g spilled, full cup 361.83 g)
never stopped: the corn hue band barely saw the thin slow stream, and the opening projected from the camera pose
sat 30-38 px above and 9-17 px left of the rim (a few centimetres of pose error). An independent review evaluated
counting rules offline; the detector now moves the opening onto the rim of the blue receiving cup found in the
empty view (`ReceiverOutline`) and counts grains that moved since the last frame in another colour than the
empty view (`MovingGrainSegmentation`), for every material.

- `count_calibration.py <material>::<bag>::<received g>::<full cup g> ...` -> `count_calibration.json`: pixel-seconds
  per bank grain received, with the production detector. Corn 43.1-54.0 over three pours (thin slow and dense fast
  streams), styrofoam 22.1-28.2 over five; the means are the calibrations in `poured_material.py`.
- `stop_replay.py ...` -> `stop_replay.json`: the bags replayed with the demo's live-bank settings, and where the
  controller would have been told the goal was reached.

The heap in the receiver as the stop signal: the detector counts pixels of the receiver's inside that have shown
grains lying still for a second, and `HeapGauge` reads the fill off that area with the share of the surface the
camera sees over the rim (none below about 0.5). The live bank tells the controller the larger of the counted and
the heap fill.

- `heap_calibration.py ...` -> `heap_calibration.json`: heap pixels at the end of each weighed pour over the visible
  share at its weighed fill. Corn 2186 at 0.69 (2806 at 0.87), styrofoam 513-675 at 0.63-0.70 (about 2200 at 0.79 and
  0.91); calibrated at the goal, the gauge reads high above it and stops a pour early rather than late.
- `stop_replay.py` now also reports when the heap alone shows the goal. Near the goal it reads the weighed fill to
  0.02 (corn 0.692/0.693, styrofoam 0.651/0.653, 0.648/0.630); in fast pours the count reaches the goal 0.4-2 s first.
