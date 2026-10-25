# Recorded wall-fill regression evidence

These are original per-frame colour measurements and weighings for corn in the translucent receiver. They are data fixtures, not generated target predictions. Each `.npz` contains 39 frames' signed Lab yellowing, receiver mask, timestamps and legacy geometric lookup: the frames ending three frames before the source cup turns back from the tilt it held at the end of the transfer, which is when the controller reads the wall. The adjacent JSON is the bag's original `weighed_particles.json`, renamed to match the recording. `dataset.json` declares the 26 development and three validation recordings and their feature configuration; fitting it gives the deployed `translucent_corn_wall.json` exactly.

All caches are cut with the receiver modelled standing on the table top (0.875 m in the world root), as the demo models it since 2026-10-25 14:31. The wall reading depends on that height: the same pours cut with the receiver at the earlier 0.925 m read about 0.09 lower against this calibration. Bags recorded before 2026-10-25 14:31 model the receiver at 0.925 m and were cut here as if at 0.875 m; `translucent_corn_wall_lifted_receiver.json` is the calibration fitted at 0.925 m (the previous fixtures), for replaying those bags as they ran.

Sources:

| Archive | Recording starts |
|---|---|
| `/tmp/bank-receiver-corn4` (from `corn-4-bags.zip`) | 2026-10-16 09:28:20, 09:34:34, 09:39:31, 09:44:59 |
| `/tmp/pouring-restarts-previous` | 2026-10-19 11:10:44, 13:18:48, 13:22:38 |
| `/tmp/pouring-goals-03-03-04-01` | 2026-10-19 16:07:31, 16:10:56, 16:25:52 (validation) |
| `goalSweep-24-10-corn.zip` | 2026-10-24 14:48:54, 15:07:09, 15:09:24, 15:12:03, 15:14:54, 15:19:14, 15:23:09, 15:26:28, 15:42:01, 15:46:43 |
| `minClearance03-band01-2026-10-24_16-03-49_corn_translucent_cup_goal30_start100_approach3.zip` | 2026-10-24 16:03:49 |
| `goalSweep-corn-fillsonly.zip` | 2026-10-25 13:37:03, 13:49:14 |
| `goalSweep-corn-noBank.zip` | 2026-10-25 13:52:03, 13:54:01, 13:56:04, 13:58:00 |
| `2026-10-25_14-31-05_corn_translucent_cup_goal30_start100_approach3-graspMiddle.zip` | 2026-10-25 14:31:05 (recorded at 0.875 m) |
| `good-middle-grasp-pour2026-10-25_14-43-35_corn_translucent_cup_goal30_start100_approach3.zip` | 2026-10-25 14:43:35 (recorded at 0.875 m) |

Excluded: 2026-10-19 16:13:33 following the user's correction; 2026-10-25 13:40:23, 13:44:01 and 13:46:53, stopped by hand, whose source never turned back below the returned tilt in the bag; 2026-10-25 14:00:09, which received almost nothing and shows too little colour. The development pours of 2026-10-19 09:37:10, 10:24:17, 10:44:40, 10:47:56, 13:04:58, 15:09:57, 15:27:50, 15:46:48 and of 2026-10-23 09:24-09:47 were in the previous fixtures, but their bags were no longer available to cut at the new height.

Extraction used `experiments/results/real_pouring/analysis_2026-10-19/cache_wall_yellowing.py` for the whole pour with the receiver's height set to 0.875 m (it reproduces the previous fixtures exactly at 0.925 m), followed by `experiments/scripts/retain_held_tilt_frames.py`. The pixelwise temporal minimum is recomputed in the production estimator; the fixture does not contain precomputed new-model estimates.

Leave-one-out on development: bias +0.002, mean absolute error 0.028; over the 24 pours read inside the calibrated range the root mean square error is 0.0295 (the deployed calibrated noise), and the geometric fallback's is 0.0880 (the deployed geometric noise). The three reserved pours read -0.022, -0.003 and -0.007. The two middle-grasp pours, the only ones recorded at 0.875 m, are development pours: held out, they read +0.005 and -0.040, the second beyond the 0.03 bound the reserved pours are tested against. The calibration fitted at 0.925 m reads these caches +0.089 high.
