# Real pouring session 2026-10-14: analysis

Raw data: `~/Downloads/open-loop-no-bank.zip` (open loop with the fitted drain, repose 27 deg, k 2.03)
and `~/Downloads/bag-bank-trace.zip` (the same script with the live bank, its trace in
`live_bank_trace.json`); the 2026-10-12 bags as in `../analysis_2026-10-12/README.md`.

Order: `../analysis_2026-10-12/extract.py <bag dir>` and `kinematics.py <bag dir>` (with ROS sourced) on
both new bags, then `live_pours.py <2026-10-12 folder> <open-loop bag> <live-bank bag>`.
Result: `live_pours.json` (tilt and believed fills of all seven demo pours, what the fitted drain
predicts on each, the bank trace, and the 2026-10-12 fits redone with both new pours added);
written up as section Q of the pouring model verification page.

`segmentation_comparison.py <bag dir> ...` counts recorded pours with both grain segmentations
(whiteness, and any change from the empty view as meant for corn) and writes
`segmentation_comparison.json`.
`replay_bank.py <bag dir> <uncalibrated|fitted>` replays a live-bank bag through the bank offline.
