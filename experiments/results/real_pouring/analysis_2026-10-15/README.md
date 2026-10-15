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
