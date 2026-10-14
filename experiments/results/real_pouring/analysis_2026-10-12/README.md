# Real pouring session 2026-10-12: analysis

Raw data: `~/Downloads/5BagsIntoJeroenCup.zip` (5 demo pours) and `~/Downloads/5CalibrationRamps.zip` (5 ramps); unzip into `demo/` and `ramps/` next to these scripts.

Order: `extract.py <bag dirs>` and `kinematics.py <bag dirs>` (with ROS sourced), then `onset.py`, `demo_flow.py`, `fit.py`, `section_o_data.py`, `section_o_svg.py`.
Results: `fits.json` (one angle: rho 27 deg, k 2.03; start/stop: 75/26 deg, k 1.90), `onset.json`, `demo_flow.json`; written up as section O of the pouring model verification page.
`sim_check_repose_at_initialization.py` is the sim probe that passes the repose angle to the drain built at initialization.
