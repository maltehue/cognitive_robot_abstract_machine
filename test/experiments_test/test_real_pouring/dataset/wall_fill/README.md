# Recorded wall-fill regression evidence

These are original per-frame colour measurements and weighings for corn in the translucent receiver. They are data fixtures, not generated target predictions. Each `.npz` contains 39 frames' signed Lab yellowing, receiver mask, timestamps and legacy geometric lookup: the frames ending three frames before the source cup turns back from the tilt it held at the end of the transfer, which is when the controller reads the wall. The wall reads more while the tilted source stands above the receiver than once it is upright (+3.2 pp on 2026-10-23, +1 pp before, with the upright-fitted calibration). The adjacent JSON is the bag's original `weighed_particles.json`, renamed to match the recording. `dataset.json` declares the 21 development and three validation recordings and their feature configuration.

Sources in `~/Downloads`:

| Archive | Recording starts |
|---|---|
| `corn-4-bags.zip` | 2026-10-16 09:28:20, 09:34:34, 09:39:31, 09:44:59 |
| `2026-10-19_09-37-10.zip` | 2026-10-19 09:37:10 |
| `pour-transparent-new-perception.zip` | 2026-10-19 10:24:17 |
| `2bags-pour-corn-transparent.zip` | 2026-10-19 10:44:40, 10:47:56 |
| `2026-10-19_11-10-44.zip` | 2026-10-19 11:10:44 |
| `2026-10-19_13-04-58.zip` | 2026-10-19 13:04:58 |
| `2-corn-bags-2026-10-19_13.zip` | 2026-10-19 13:18:48, 13:22:38 |
| `2026-10-19_15-09-57.zip` | 2026-10-19 15:09:57 |
| `2026-10-19_15-27-50.zip` | 2026-10-19 15:27:50 |
| `corn-transparent-underpour.zip` | 2026-10-19 15:46:48 |
| `pouring-goals-03-03-04-01.zip` | 2026-10-19 16:07:31, 16:10:56, 16:25:52 |
| `pouring-corn-chatgpt-perception-03-03-02-02-04-04.zip` | 2026-10-23 09:24:30, 09:31:07, 09:36:03, 09:39:13, 09:41:49, 09:47:39 |

The 16:13:33 recording is explicitly excluded following the user's correction. The 2026-10-15 calibration archive was inspected but had no `weighed_particles.json`, so its handover-only weighing was not introduced as a training label. Failed, unweighed recordings and other receiver/material combinations were not mixed into this calibration.

Extraction used `experiments/results/real_pouring/analysis_2026-10-19/cache_wall_yellowing.py` for the whole pour, followed by `experiments/scripts/retain_held_tilt_frames.py`, which keeps the one-second scoring window at the held tilt plus its causal history. No cache value reached the extractor's ±100 clipping limits. The pixelwise temporal minimum is recomputed in the production estimator; the fixture does not contain precomputed new-model estimates.

Latest recordings were excluded from fitting but inspected during development; this is a regression corpus, not a permanently blind test set. See `experiments/results/real_pouring/perception_2026-10-19/README.md` for evaluation limits and reproduction commands.
