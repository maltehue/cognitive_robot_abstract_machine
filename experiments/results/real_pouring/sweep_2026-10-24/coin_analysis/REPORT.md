# COIN-style offline analyses of the 2026-10-24 goal sweeps

Three replay analyses on the recorded corn (10 pours) and styrofoam (14 pours) sweeps,
testing ideas from the COIN model of motor learning (Heald, Lengyel & Wolpert, 2021):
whether a memory across pours would have been justified (v-a), what a cue-selected
memory is worth (v-b), and whether expressing the hypotheses as a probability-weighted
mixture would have forecast better than committing to one (v-c, corn only). Everything
is replay on the recordings; no robot, no library change, nothing committed.

## Environment and provenance

- Code: `/home/malte/libs/cognitive_robot_abstract_machine`, branch
  `pouring-constraint-inspector` at `918254f00e`, the commit the bags name
  (`pour_settings.json: repository.commit`). The bags were recorded with
  `has_uncommitted_changes: true`; the checkout used here also carries uncommitted
  edits in `experiments/src/experiments/real_pouring/live_bank.py` (a `corrects_drain`
  flag defaulting to the recorded behaviour) and in two test files. They do not change
  the replayed bank's arithmetic, but exact equality with the robot's code is not
  guaranteed.
- Python: `/home/malte/venvs/cram2/bin/python`, run from the repository root with
  plain imports (no `PYTHONPATH` override). Verified once:
  `experiments.__file__ = /home/malte/libs/cognitive_robot_abstract_machine/experiments/src/experiments/__init__.py`,
  `giskardpy/src/giskardpy/__init__.py` and `semantic_digital_twin/src/semantic_digital_twin/__init__.py`
  under the same checkout. `MPLCONFIGDIR=/tmp/pouring-mpl`; the bank replay (v-c)
  additionally `ROS_DOMAIN_ID=91 ROS_LOCALHOST_ONLY=1`.
- Data: `experiments/results/real_pouring/sweep_2026-10-24/{bags_corn,bags_styrofoam}`.
  The bag folders there hold no `.mcap`; for v-c the ten corn recordings were unpacked
  from `~/Downloads/goalSweep-24-10-corn.zip` into the session scratchpad
  (`/tmp/claude-1000/-home-malte-Documents-phdThesis/f5955e40-5036-40c0-842f-59ea233cd86e/scratchpad/mcap_corn/goalSweep-24-10/`),
  replay outputs into `.../scratchpad/replays/<pour>.json`. Nothing was added to the
  results tree except this folder.
- Scripts (this folder): `coin_common.py` (loading, targets, frozen-forecast scoring),
  `v_a_memory_across_pours.py`, `v_b_cue_selected_memory.py`, `v_c_replay_bank.py`
  (detector + bank replay per bag), `v_c_mixture_vs_commit.py`, `make_figures.py`.
  Results: `v_a_results.json`, `v_a_concept_values.csv`, `v_a_forecast_rows.csv`,
  `v_b_results.json`, `v_b_forecast_rows.csv`, `v_c_results.json`,
  `v_c_forecast_rows.csv`; figures `fig_va_concept_values.png`,
  `fig_va_vb_forecast_mae.png`, `fig_vb_novelty.png`, `fig_vc_mixture_vs_commit.png`.

## What the sweep's bank actually tracked (read before the numbers)

- Hypotheses in every trace: `H0` nominal, `H2` aim (discharge coefficient free),
  `H5` catch-all. `tracks_onset=True`, so every hypothesis tracks the onset (repose
  angle) within the pour; there is no onset hypothesis of its own and no drain-rate
  hypothesis (`outflow_capacity=False`, `adapts_drain_rate=False`). The trace's
  `log_drain_scale` is the constant log(0.8) = -0.223 (the 0.2 stall probability
  weighting the capacity, `LiveBank._blended`), not an inferred correction. "Drain-rate
  correction if present": not present.
- The concept values the controller ran on are the active hypothesis's controller
  point: repose at the 0.1 quantile of the onset posterior, conditional on the inferred
  flow mode, and *lowered by the material's hysteresis while the mode is "flowing"*
  (corn 21.2 deg + 27.1 deg per share poured out, styrofoam 60 deg). An end-of-pour
  repose correction therefore depends on the flow state at the last reading.
- The frozen forecast (`FrozenForecast`, `DrainGeometry.outflow`) depends on the
  repose angle and the drain rate only. The discharge coefficient sets where the
  stream lands, not how much leaves the source. In v-a/v-b the memories therefore
  differ from each other only through their onset; the aim concept is tabulated and
  replayed through the write-back rule but cannot show in a fill forecast.
- Targets: corn -> the wall readings the bank admitted to its likelihood
  (`receiver_observation`, every ~0.4 s; noise 5.68 grains = calibrated, 13.67 =
  geometric), interpolated, windows with a gap > 0.55 s or a calibration transition
  excluded, as the 2026-10-19 ablation did; styrofoam -> the camera's crossing count
  (`counted_grains`; receiver = source cup, so 1 grain = 1 receiver pp). The two
  substances are thus scored against different sensors; the styrofoam count's
  calibration against the scale (count/weighed 0.8-1.2 over the sweep) is not corrected.
- Conventions kept from the ablation: errors in receiver percentage points; origins
  every 10 readings (about 1 s); horizons 0.5/1.5/3.0 s with 1.5 s as headline; targets
  shifted by 0 and 0.33 s (sensitivity, scoring only); means within each recording then
  across recordings; overlapping windows are not independent samples; no significance
  claims. The source estimate at an origin is the trace's (the active hypothesis's) for
  every model, so only parameters differ. Every model keeps the recorded
  `log_drain_scale` at the origin (the stall factor).

## v-a. Would a memory across pours have been justified?

### (1) End-of-pour concept values, in pour order

`v_a_concept_values.csv`, `fig_va_concept_values.png`. "repose end" is the last written
repose correction; "quiescent" restores the hysteresis where the pour ended with the
stream still seen within 1 s of its end and the restored value lies no more than 5 deg
above the pour's highest onset estimate (only corn 15-07-09 qualifies: 24.0 -> 49.1 deg).
"aim" is the discharge factor on C_d = 0.2 while the aim hypothesis was last in charge;
the end value is 1.0 (unrevised) where the controller had fallen back to `H0`.

| corn pour | goal | dur s | repose end deg | quiescent deg | aim factor | P(H2)>0.5 at s | first commit s | share before commit |
|---|---|---|---|---|---|---|---|---|
| 14-48-54 | 0.3 | 28.6 | 39.3 | 39.3 | 0.168 | 2.3 | 7.4 | 0.26 |
| 15-07-09 | 0.1 | 11.6 | 24.0 | 49.1 | 0.175 | 7.0 | 7.6 | 0.65 |
| 15-09-24 | 0.1 | 14.3 | 46.3 | 46.3 | 0.284 | 7.5 | 8.2 | 0.57 |
| 15-12-03 | 0.2 | 30.4 | 44.8 | 44.8 | 0.174 | 1.5 | 6.9 | 0.23 |
| 15-14-54 | 0.2 | 19.9 | 45.9 | 45.9 | 0.174 | 1.4 | 7.9 | 0.40 |
| 15-19-14 | 0.3 | 28.5 | 45.8 | 45.8 | 0.174 | 6.0 | 6.8 | 0.24 |
| 15-23-09 | 0.4 | 30.8 | 40.0 | 40.0 | 0.182 | 1.6 | 8.9 | 0.29 |
| 15-26-28 | 0.4 | 29.7 | 42.4 | 42.4 | 0.169 | 5.8 | 8.9 | 0.30 |
| 15-42-01 | 0.5 | 46.4 | 56.5 | 56.5 | 0.213 (ended on H0: 1.0) | 6.5 | 9.2 | 0.20 |
| 15-46-43 | 0.5 | 32.1 | 52.3 | 52.3 | 0.159 | 6.0 | 8.6 | 0.27 |

| styrofoam pour | goal | dur s | repose end deg | aim factor | P(H2)>0.5 at s | first commit s | share before commit |
|---|---|---|---|---|---|---|---|
| 16-24-23 | 0.7 | 29.3 | 62.1 | 0.201 | 2.0 | 12.9 | 0.44 |
| 16-27-52 | 0.7 | 24.4 | 58.4 | 0.316 | 1.6 | 15.4 | 0.63 |
| 16-43-53 | 0.5 | 28.2 | 62.2 | 0.752 | 2.0 | 15.3 | 0.54 |
| 16-47-34 | 0.4 | 23.5 | 60.3 | 0.942 | 0.8 | 17.9 | 0.76 |
| 16-51-10 | 0.3 | 33.0 | 56.0 | never (H0) | 0.8 | never | 1.00 |
| 17-07-46 | 0.3 | 30.8 | 55.5 | 0.423 | 1.6 | 14.6 | 0.47 |
| 17-10-54 | 0.4 | 30.8 | 61.2 | 0.226 | 3.4 | 11.4 | 0.37 |
| 17-14-09 | 0.5 | 28.2 | 60.1 | 0.706 | 2.1 | 17.9 | 0.64 |
| 17-17-18 | 0.2 | 29.2 | 60.1 | 0.673 | 24.6 | 25.9 | 0.89 |
| 17-20-37 | 0.2 | 30.5 | 59.9 | 0.607 | 24.6 | 25.6 | 0.84 |
| 17-24-02 | 0.1 | 26.7 | 60.2 | never (H0) | 2.1 | never | 1.00 |
| 17-33-12 | 0.1 | 22.5 | 62.1 | never (H0) | 3.1 | never | 1.00 |
| 17-35-57 | 0.6 | 34.9 | 60.0 | 0.472 | 0.8 | 17.6 | 0.50 |
| 17-39-52 | 0.6 | 35.4 | 58.5 | never (H0) | 0.8 | never | 1.00 |

Consistency across pours (n; mean; median; sd; MAD; range):

- corn onset, end value: 10; 43.7; 45.3; 8.7; 4.1; 24.0-56.5 deg. With the hysteresis
  restored on 15-07-09: 46.2; 45.8; 5.3; 3.4; 39.3-56.5 deg. The two goal-0.5 pours
  (56.5, 52.3) sit above the rest.
- styrofoam onset, end value: 14; 59.8; 60.1; 2.1; 1.4; 55.5-62.2 deg. Tighter than
  corn; all end values are un-lowered onsets (styrofoam's 60 deg hysteresis would put a
  "flowing" value near 0 deg, which never occurs at a pour end).
- corn aim factor: 10; 0.187; 0.174; 0.037; 0.0055; 0.159-0.284 (sd of log 0.17): eight
  of ten within 0.159-0.182, one outlier at 0.284.
- styrofoam aim factor: 10 of 14 (four never committed to aim); 0.53; 0.54; 0.24; 0.19;
  0.20-0.94 (sd of log 0.53): no consistent value, as expected for a failure no
  hypothesis names.
- Aim first leading (P(H2) > 0.5): corn median 5.9 s (1.4-7.5); styrofoam median 2.0 s
  (0.8-24.6). On styrofoam it leads early without being committed (the commit rule
  needs 0.9 on the concept for three readings).
- First commit: corn all 10, mean 8.0 s, median 8.05 s (6.8-9.2); styrofoam 10 of 14,
  mean 17.5 s, median 16.5 s (11.4-25.9).
- Time spent before the first commit (relearning from the uncalibrated default): corn
  mean 34 % of the pour, median 28 % (20-65 %, the two short goal-0.1 pours highest);
  styrofoam mean 72 %, median 70 % (37-100 %; four pours never committed).

### (2) The write-back rule replayed

`recovery.OnsetWriteBack` (centre moves only after `WRITE_BACK_RUN = 3` pours in a
row outside centre +- 1.96 spread on the same side; new centre = their mean, spread =
max(old, their sd)), every pour treated as explained: the consistency check is skipped
because its features (tilt at the first grain, outflow over the first ten grains) and
residual tables exist only for the simulated cup. The rule takes one point estimate per
pour, so the within-pour "95 % of the posterior outside" veto is not part of it and is
not replayed. Centre = the first pour's value.

Spread: the simulation configuration's dynamic onset spread (2.39 deg,
`reference_configuration.json` twin.cup.repose_angle_dynamic, 204 simulated plain-cup
pours) was used as the transferred spread; it is a within-simulation scatter of a twin
cup and probably does not transfer, so the across-pour scatter of the real sequence
(sd, ddof 1) was run as the alternative - noting that this spread is computed from the
very pours it then judges. The configuration has no spread for C_d ("never fitted");
the aim factor was run with its scatter spread (sd of the log factor) in the rule's
logarithmic variant (`DrainRateSpread`), a re-implementation of `OnsetWriteBack.after`
for a general `ParameterSpread`.

| sequence, value | centre (pour 1) | spread | interval | sides, pours 1... | fires |
|---|---|---|---|---|---|
| corn onset, end value | 39.3 deg | sim 2.39 deg | 34.7-44.0 deg | 0 - + + + 0 - 0 + + | **after pour 5** (15-14-54) -> centre 45.6 deg, spread 2.39 deg |
| corn onset, end value | 39.3 deg | scatter 8.7 deg | 22.3-56.3 deg | all 0 except pour 9 (+) | no |
| corn onset, quiescent | 39.3 deg | sim 2.39 deg | 34.7-44.0 deg | 0 + + + 0 0 - 0 + + | after pour 4 -> 46.7 deg |
| corn onset, quiescent | 39.3 deg | scatter 5.3 deg | 28.9-49.7 deg | 0 ... 0 + + | no (run of 2 at the end) |
| corn aim factor | 0.168 | scatter log 0.17 | 0.121-0.234 | 0 0 + 0 0 0 0 0 0 0 | no |
| styrofoam onset | 62.1 deg | sim 2.39 deg | 57.4-66.8 deg | 0 0 0 0 - - 0 0 0 0 0 0 0 0 | no (run of 2: 16-51-10, 17-07-46) |
| styrofoam onset | 62.1 deg | scatter 2.07 deg | 58.1-66.2 deg | same | no |
| styrofoam aim factor | 0.201 | scatter log 0.53 | 0.072-0.563 | 0 0 + + 0 0 + + + 0 (10 valued pours) | after the 9th valued pour (17-20-37, 10th styrofoam pour) -> centre 0.66, spread 0.53 |

Reading: with the simulation's narrow spread the corn onset rule fires after the fifth
corn pour, moving the centre from the first pour's 39.3 deg to 45.6 deg (mean of pours
3-5), 0.1 deg from the simulated twin's own dynamic onset centre of 45.7 deg; the two
goal-0.5 pours (56.5, 52.3) then sit above the moved interval again (run of 2, no second
firing). With the real scatter as spread the interval is so wide that nothing fires. The
styrofoam onset is consistent and never moves. The styrofoam aim "fires" to 0.66 after
three pours at 0.6-0.7, a write-back of noise: the earlier valued pours ran 0.20-0.94 and
the next is 0.47. The sweep never returns to corn, so whether a written-back corn centre
would have been recovered or re-tuned later cannot be tested; one substance switch only.

### (3) What the memory would have saved at pour start

1.5 s frozen forecasts at origins before the pour's first commit (whole pour where it
never committed) and in a fixed first-5 s window; the first pour of each substance has
no memory and is excluded (9 corn, 13 styrofoam pours). Models: `default` = the
uncalibrated drain (repose 0, k 0.8 x 0.8); `memory_end` = median of the earlier
same-substance pours' end repose (and last aim log scale, inert in the forecast);
`memory_quiescent` = same with the hysteresis-restored onset; `memory_rule` = the
write-back rule's centre in force (first pour's value until it fires, then 45.6 deg);
`as_used` = the corrections the robot's bank had written at that origin.
Recording-balanced MAE (bias), receiver pp:

| window, shift | substance | default | memory_end | memory_quiescent | memory_rule | as_used | origins / pours |
|---|---|---|---|---|---|---|---|
| before commit, 0 s | corn | 5.39 (+5.39) | 0.91 (+0.23) | 0.63 (-0.07) | 0.82 (+0.12) | 0.41 (-0.22) | 48 / 9 |
| before commit, 0.33 s | corn | 5.22 (+5.22) | 0.92 (-0.01) | 0.67 (-0.28) | 0.83 (-0.12) | 1.42 (+0.55) | 59 / 9 |
| first 5 s, 0.33 s | corn | 5.55 (+5.55) | 0.42 (+0.42) | 0.16 (+0.16) | 0.31 (+0.31) | 1.18 (+1.18) | 45 / 9 |
| before commit, 0 s | styrofoam | 13.08 (+12.95) | 1.99 (-1.93) | 1.99 | 2.00 (-1.98) | 3.12 (+0.55) | 268 / 13 |
| before commit, 0.33 s | styrofoam | 12.91 (+12.81) | 2.16 (-2.11) | 2.16 | 2.18 (-2.16) | 3.57 (+0.34) | 265 / 13 |
| first 5 s, 0.33 s | styrofoam | 10.90 (+10.90) | 0.25 (-0.25) | 0.25 | 0.25 | 3.15 (+2.88) | 65 / 13 |

Split by whether the camera saw a stream at the origin (before commit, 0.33 s shift):
corn stream-not-seen, 55 origins: default 5.50, memory_end 0.65, as_used 1.28; corn
stream-seen, 4 origins: 1.24, 4.62, 3.39 (the memory without a hysteresis under-predicts
once flow has started). Styrofoam not-seen, 244: 12.92 / 1.94 / 2.98; seen, 21: 12.85 /
5.10 / 11.22.

Reading: the uncalibrated default predicts 5-6 pp (corn) and 11-13 pp (styrofoam) of
inflow per 1.5 s at the start of every pour, when nothing flows; a memory of the earlier
pours' onset removes nearly all of it (0.6-0.9 pp corn, 2 pp styrofoam) before the bank
has seen a grain. The live bank's own corrections (`as_used`) reach a similar level
within the first second because every hypothesis tracks the onset from the first
reading, so the relearning cost in this forecast is small; what the pour spends before
the aim commit (corn 28 %, styrofoam 70 % median) concerns the aim concept, which this
forecast cannot score. The memory's residual error is concentrated where flow starts,
which a fixed onset gets wrong by a few degrees (styrofoam bias -2 pp before commit:
the memory's 60 deg onset is slightly late).

## v-b. Value of a cue-selected memory

`v_b_results.json`, `fig_va_vb_forecast_mae.png`, `fig_vb_novelty.png`. Leave one pour
out; `same_memory` = median end repose and last-aim log scale over the other pours of
the same substance (corn 45.3 deg, factor 0.174; styrofoam 60.1 deg, 0.535, over all
pours); `other_memory` = the median over every pour of the other substance; `default`
and `as_used` as above. Frozen forecasts along the whole held-out pour.
Recording-balanced MAE (bias), receiver pp, 1.5 s:

| substance, origins, shift | default | same memory | other memory | as used | origins / pours |
|---|---|---|---|---|---|
| corn, all, 0 s | 6.04 (+5.92) | 1.91 (-1.48) | 1.86 (-1.63) | 1.62 (+0.34) | 188 / 10 |
| corn, all, 0.33 s | 5.82 (+5.75) | 1.88 (-1.56) | 1.88 (-1.70) | 1.97 (+0.58) | 200 / 10 |
| corn, stream seen, 0 s | 5.83 (+5.50) | 3.54 (-3.18) | 3.60 (-3.24) | 3.08 (+1.92) | 61 / 9 |
| corn, stream seen, 0.33 s | 5.30 (+5.16) | 3.30 (-3.18) | 3.47 (-3.36) | 2.81 (+1.50) | 65 / 10 |
| styrofoam, all, 0 s | 13.04 (+12.83) | 2.76 (-2.71) | 2.99 (-1.42) | 3.66 (+0.76) | 377 / 14 |
| styrofoam, all, 0.33 s | 13.02 (+12.83) | 2.81 (-2.77) | 2.94 (-1.47) | 4.31 (+0.69) | 373 / 14 |
| styrofoam, stream seen, 0 s | 9.56 (+8.68) | 9.11 (-9.11) | 8.35 (-8.22) | 8.20 (+6.94) | 50 / 13 |
| styrofoam, stream seen, 0.33 s | 12.03 (+11.87) | 5.92 (-5.92) | 5.49 (-5.04) | 10.58 (+10.12) | 50 / 13 |

Other horizons, all origins, 0.33 s shift: corn 0.5 s - default 2.31, same 0.90, other
0.88, as used 0.99; corn 3 s - 9.98, 3.63, 3.71, 3.50; styrofoam 0.5 s - 4.80, 0.94,
1.13, 1.35; styrofoam 3 s - 23.28, 5.68, 5.34, 8.62. The hysteresis-restored
same-substance memory is within 0.02 pp of `same_memory` everywhere.

Reading: any memory beats the uncalibrated default by a factor of 3 (corn) to 4-5
(styrofoam) over the whole pour, but the *cue-selected* one is worth little in these
forecasts: the same- and other-substance memories differ only in their onset (45 vs 60
deg), which matters only while the tilt crosses 45-60 deg (about five seconds of a 3
deg/s approach), and over the whole pour the two are within 0.1-0.2 pp of each other;
on corn the styrofoam memory is even marginally better at all origins, because in the
pours that start from the pre-tilt corn's stream appears later than its 45 deg memory
predicts. The cue shows in the early window only: styrofoam first 10 s, same memory 0.71
vs corn memory 1.38 pp (shift 0.33 s; first 5 s: 0.23 vs 0.59); corn first 5 s: 0.12 vs
0.00 (the other-substance memory "wins" by predicting nothing). On the stream-seen
origins every frozen model under-predicts by 3-9 pp: a frozen starting model has no
flow state, hysteresis or bursts, and the styrofoam failure (a release that outlasts
the tilt) is outside every model, so styrofoam's memories are expected to be
inconsistent and poor once flowing.

Novelty check (the corn memory's recording-level error on each styrofoam pour against
the ten corn leave-one-out errors; threshold = their 95th percentile, which with n = 10
is essentially their maximum):

| window, shift | corn LOO p95 (max) | corn memory on first styrofoam pour (16-24-23) | flagged |
|---|---|---|---|
| whole pour, 0.33 s | 2.68 (2.69) | 3.86 | yes; 9 of 14 styrofoam pours |
| whole pour, 0 s | 2.71 (2.74) | 3.91 | yes; 8 of 14 |
| first 10 s, 0.33 s | 1.86 (2.38) | 1.45 | no; 1 of 14 |
| first 5 s, 0.33 s | 0.18 (0.19) | 0.55 | yes; 14 of 14 |

The corn memory's error on the first styrofoam pour exceeds everything it did on corn
over the whole pour and in the first 5 s (where the corn memory predicts flow from 45
deg on and styrofoam shows none - a clean early novelty signal, but at a threshold of
0.18 pp, below what the wall sensor resolves on corn), while over the first 10 s
styrofoam is not distinguishable from corn (the corn pours' own early bursts raise the
corn band to 1.9 pp). Five of fourteen styrofoam pours are not flagged over the whole
pour. One switch only, no return to corn, and the two substances are scored on
different sensors.

## v-c. Mixture vs commit (COIN's expression rule), corn

`v_c_results.json`, `fig_vc_mixture_vs_commit.png`. The ten corn bags were replayed
through the detector and the bank (`v_c_replay_bank.py`: the recording's own settings
and drain, seed 0, translucent-cup corn perception, `PourReplay.of`'s recipe) to expose
each hypothesis's controller point at each reading. Agreement with the robot's trace:
the active hypothesis matches at 99-100 % of readings in eight pours, 90 % in 15-42-01
and 62 % in 15-46-43 (the replay dropped back to `H0` at 27.7 s where the robot stayed
on `H2`); first commit within 0.3 s in every pour; mean |dP(H2)| 0.05-0.17 (max up to
0.87 at single readings); replayed count at the end within 2 % of the recorded one. The
written repose correction differs by 3-7 deg on average (11.7 deg in 15-07-09) because
the flow-state flips (+-21 deg of hysteresis) fall on different readings; the onset
estimates themselves agree.

Models, all stall-weighted (k x 0.8) and scored with the same source estimate:
`as_used_recorded` = the robot's written corrections (blended), `active_replayed` = the
replay's written corrections, `active_point` = the committed hypothesis's unblended
controller point, `mixture` = the probability-weighted mean of the three hypotheses'
controller points (repose and k; C_d does not enter), `most_probable` = the point of
the most probable hypothesis without the commit rule. Targets: the replayed detector's
wall readings at every reading (as the 2026-10-19 ablation scored them). Origins every
10 readings up to the recorded trace's end. Recording-balanced MAE (bias), receiver pp:

| horizon, shift, origins | as_used_recorded | active_replayed | active_point | mixture | most_probable | origins / pours |
|---|---|---|---|---|---|---|
| 1.5 s, 0.33 s, all | 1.86 (+0.35) | 2.04 (+0.69) | 2.00 (+0.65) | 2.00 (+0.65) | 2.01 (+0.66) | 223 / 10 |
| 1.5 s, 0.33 s, stream seen | 2.60 (+1.15) | 3.11 (+2.06) | 3.29 (+2.27) | 3.32 (+2.25) | 3.31 (+2.24) | 57 / 10 |
| 1.5 s, 0.33 s, before commit | 1.60 (+0.74) | 1.62 (+0.69) | 1.32 (+0.41) | 1.34 (+0.44) | 1.34 (+0.48) | 61 / 10 |
| 1.5 s, 0.33 s, after commit | 1.94 (+0.19) | 2.16 (+0.73) | 2.21 (+0.77) | 2.20 (+0.76) | 2.21 (+0.77) | 162 / 10 |
| 1.5 s, 0 s, all | 1.56 (+0.22) | 1.82 (+0.59) | 1.84 (+0.62) | 1.84 (+0.62) | 1.85 (+0.64) | 217 / 10 |
| 0.5 s, 0.33 s, all | 0.96 (+0.17) | 1.01 (+0.29) | 0.98 (+0.27) | 0.98 (+0.27) | 0.98 (+0.27) | 246 / 10 |
| 3.0 s, 0.33 s, all | 2.98 (+0.60) | 3.35 (+1.23) | 3.29 (+1.15) | 3.29 (+1.16) | 3.30 (+1.17) | 191 / 10 |

Paired per pour (1.5 s, 0.33 s): the mixture is better than the committed point in 5 of
10 pours over all origins (7 of 10 after the commit, 2 of 10 before), always by less
than 0.05 pp; the most probable hypothesis in 2 of 10.

Reading: the mixture and the commit are indistinguishable here, and for a structural
reason rather than for lack of data. The three hypotheses differ in the discharge
coefficient (H2) and the noise model (H5), neither of which enters the outflow; all
three track the same onset, so their forecast-relevant parameters agree: the mixture's
onset differs from the committed hypothesis's by 1.7 deg on average (1.0 deg before the
commit, 1.9 after; max 20 deg at single readings where H5's slower-tracking onset still
carries weight). The posterior itself is mixed at 56 % of readings (leading probability
below 0.9), so COIN's expression rule would have been active, but it had nothing to
express differently for the fill forecast. The committed point as the robot actually
used it (blended, 0.1 s time constant) does slightly better than any unblended point
(1.86 vs 2.00 pp); that is smoothing, not expression. A test of mixture versus commit
needs hypotheses that disagree on a forecast-relevant parameter (onset or drain rate
hypotheses, as the simulation bank has). Styrofoam was not replayed (budget; and its
failure lies outside every hypothesis).

## Caveats, all in one place

- The sweep never returns to corn: no recovery, savings or re-tuning test is possible;
  one substance switch only, and its two substances are scored on different sensors.
- Styrofoam's failure is a missing concept (a release that outlasts the tilt), so its
  aim memories are inconsistent (0.20-0.94) and its forecasts once flowing are poor
  under every model; its onset memory is nevertheless the most consistent value in the
  sweep (60.1 +- 2 deg).
- No drain-rate concept was tracked (`log_drain_scale` constant); the aim concept
  cannot show in a fill forecast (the outflow does not depend on C_d). Forecast-based
  comparisons between memories, and between mixture and commit, therefore reduce to the
  onset.
- The controller's repose correction depends on the inferred flow state (hysteresis
  lowered); one corn pour (15-07-09) ended flowing and its end value (24 deg) is not a
  quiescent onset. The "restored" variant is a heuristic (stream seen within 1 s of the
  end, restored value within 5 deg of the pour's highest estimate).
- The write-back rule's consistency check was skipped (no residual tables for the real
  cup); the simulated onset spread may not transfer; the scatter spread is computed from
  the judged pours themselves; the aim rule has no configured spread and is a
  re-implementation of the onset rule on a log scale.
- Corn wall readings exist only every ~0.4 s and only while the bank admitted them
  (v-a/v-b); v-c uses the replayed detector's readings at every reading instead, so
  v-c's "as_used_recorded" numbers are not directly comparable with v-b's "as_used".
  The styrofoam count is not re-calibrated against the scale.
- Frozen forecasts assume all outflow reaches the receiver and have no flow-state model;
  overlapping windows are not independent; recording-level means over 9-14 pours; no
  significance claims. The 0.33 s shift is a sensitivity, not a measured delay.
- v-c replays the bank with seed 0 and this checkout's code on the recordings; it agrees
  with the robot's run on the active hypothesis and commit times but is not bit-exact
  (particle filter, the robot's uncommitted edits unknown).
