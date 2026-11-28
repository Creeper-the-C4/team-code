# Task 3: WNOJ smoothing and withheld-sensor reconstruction (stage A)

**Reference type: `withheld_sensor_reference`.** Every accuracy number compares a
reconstruction with the recording of a sensor that was hidden from the model. It is not an
independent optical motion-capture ground truth.

## Method (`src/task3/`)

| File | Content |
|---|---|
| `so3_lie.py` | Vectorised Exp/Log (with an exact branch near π), right/left Jacobians and their inverses (Taylor series at small angles), xyzw quaternions, hemisphere continuity, geodesic angle, chordal mean |
| `wnoj_prior.py` | White-noise-on-jerk (3rd-order IWP) local error-state prior on SO(3): transition Φ(Δt), exact Q(Δt), Cholesky whitening, analytic Jacobians (checked to < 1e-8). Construction of Barfoot, Tong & Särkkä 2014; Anderson & Barfoot 2015; Tang, Yoon & Barfoot 2019, with a first-order local approximation of J_l⁻¹ that is exact as Δt → 0 |
| `smoother.py` | Gauss–Newton MAP per segment. Factors: WNOJ prior, own gyro, own quaternion (or joint targets). Block-tridiagonal normal equations are solved by banded Cholesky in O(T); sparse and dense paths are cross-checks |
| `kinematic_tree.py` | 17-node tree over the dataset sensors. Forward kinematics with bone vectors from a standing (N-pose) calibration and `body_dimensions_m.csv` |
| `withheld_eval.py` | Reconstruction of withheld segments plus V1/V2/V3 metrics |

**State and residual conventions**
- **State per frame:** world rotation R, world angular velocity ω (the dataset gyro is world-frame rad/s), and angular acceleration α.
- **Perturbations:** left perturbation `R <- Exp(d) R`.
- **Orientation residual:** follows the frozen V4 convention `e_R = Log(R_target^T R)`.

**Reconstruction protocol, per chunk inside one trial (no chunk crosses a trial boundary):**
1. Each observed sensor is smoothed **from its own channels only**.
2. Withheld segments are rebuilt in tree order from observed or already-rebuilt neighbours.
   Joint factors use `R_target = R_neighbour @ mean relative rotation`, with a per-axis
   spread. These joint statistics are computed from **training participants only**.
3. Baselines:
   - `copy_parent`: the parent's orientation;
   - `mean_relative`: the parent's orientation times the training mean relative rotation.
4. A unit test verifies that writing NaN into the withheld sensor's channels leaves the
   reconstruction bitwise identical.

**Metrics:**
- **V1:** geodesic angle against the withheld sensor's quaternion, in degrees (median,
  RMSE, P95, subject-equal median, and per activity).
- **V2:** angular-velocity RMSE against its gyro, in rad/s.
- **V3:** forward-kinematics end-point displacement in metres. It is derived from the
  standing calibration, so it is not a measured position. Translation is not estimated in
  this stage.

**Hyperparameters.** There are at most two: the WNOJ jerk density `qc` and the joint-spread
multiplier `joint_scale`. Each fold chooses them on its **validation** participants over the
grid {1e2, 1e3, 1e4} × {0.5, 1, 2}. Test participants are evaluated once.

**Acceleration (two guards).** Acceleration factors stay gated until the P0
`diag_accel_unit.py` verdict:
1. **CLI:** `--accel-unit none|disabled` is the default; `g` or `mps2` is refused.
2. **Factor registry:** `smoother.FACTOR_TYPES` contains no acceleration factor and
   `ACCEL_FACTOR_REGISTERED = False`. No code path reads the accelerometer channels; a unit
   test shows that writing NaN into every Acc channel leaves all outputs bitwise identical.

`summary.json` echoes `"accel_unit": "none"` and `"accel_factor_registered": false`.

## Run on the cloud cache

```bash
PY=/root/autodl-tmp/C_Problem/.venv/bin/python
DATA=/root/autodl-tmp/C_Problem/data/processed/independent_v1
$PY scripts/task3/run_task3_wnoj.py --data-root "$DATA" \
    --splits results/preprocessing/independent_v1/subject_splits_6fold.json \
    --body-dims results/preprocessing/independent_v1/body_dimensions_m.csv \
    --out results/task3_v4 \
    --fold-subsets-json results/task2_v4/F1H_backward/per_fold_selected_subsets.json
```

| Option | Default | Meaning |
|---|---|---|
| `--schemes` | `H1` | Always runs. H1 withholds one sensor at a time (`--h1-sensors`, default Head, UpperBack, RightForeArm, LeftUpperArm, RightShank, LeftThigh; `all` for all 17) |
| `--observed-sensors` / `--withhold` | — | H2: one fixed layout |
| `--fold-subsets-json` | — | H3: each fold's Task 2 layout `folds[str(fold)][str(k)]` of size `--h3-k` (alias `--subset-k`, default 3 4 5 6), read from `run_sensor_selection.py` output. Each fold reads only its own layout, which prevents Task 2→3 cross-fold leakage |
| `--max-trials-per-activity` | 2 | Trials per activity per participant (0 = all) |
| `--max-seconds-per-trial` | 30 | Seconds used per trial |
| `--chunk-seconds` | 10 | Chunk length |
| `--factors` | `both` | Observed-segment factors: `gyro`, `orientation` or `both` |
| `--workers` | min(8, CPUs) | Parallel worker processes |
| `--no-calibrate` | off | Skip the per-fold hyperparameter grid |
| `--no-ablation` | off | Skip the ablation |
| `--resume` | off | Reuse finished folds from `_parts/` |
| `--overwrite` | off | Recompute even if `summary.json` exists |

**Outputs** (in `--out`, all small):

| File | Content |
|---|---|
| `summary.json` | Reference type, acceleration echo, per-fold hyperparameters, explicit `input_masks` per fold and scheme, solver diagnostics (`convergence_rate`, `failure_rate`, `mean_iterations`, `max_iterations`, largest final orientation update), observed-segment residuals, all metrics |
| `per_sensor_metrics.csv` | Per scheme × withheld sensor × method: `input_mask`, `rmse_deg`, `median_deg`, `p95_deg`, subject-equal median, V2 RMSE |
| `per_activity_metrics.csv` | Same metrics per activity |
| `per_subject_metrics.csv` | Same metrics per participant |
| `observed_residuals.csv` | `gyro_residual` (rad/s) and `orientation_residual` (deg) of the smoothed observed segments against their own recordings |
| `ablation_summary.csv` | Drift over 2/5/10 s for gyro-only, orientation-only and both |
| `README.md` | Generated summary | The ablation covers
gyro-only, orientation-only and both, over 2, 5 and 10 s horizons. In gyro-only mode the
first frame of each horizon is anchored to the measured quaternion. `_parts/*.npz` holds
per-fold intermediates for `--resume`; `.npz` files are git-ignored.

**Runtime.** About 20 s per participant per stage on one core (calibration, evaluation and
ablation) at the default caps. All six folds should take roughly 10–15 min with 8 workers.
To smoke-test first, add `--folds 0 --max-trials-per-activity 1`.

## Tests and benchmark

```bash
python -m pytest -q tests/test_task3_wnoj.py
python scripts/task3/benchmark_wnoj.py --out results/task3_offline_v1
```

The tests cover:

- **SO(3) operators:** Lie-group identities, including `J_r J_r^-1 = I` and Exp/Log near π.
- **Jacobians:** the WNOJ and orientation Jacobians match finite differences to < 1e-8.
- **Normal equations:** the block normal equations equal J^T J.
- **Solvers:** banded, sparse and dense solves agree to < 1e-9.
- **Scaling:** the normal matrix grows linearly in T.
- **Smoother:** smoothing reduces orientation noise.
- **Acceleration gate:** `--accel-unit` is gated, and no acceleration factor is ever
  registered or read (NaN accelerometer channels leave outputs unchanged).
- **Withheld reconstruction:** it beats `copy_parent` on a known kinematic chain and does
  not depend on the withheld channels.
- **H3 parsing:** layout files are parsed, including the Task 2 output format, and in a
  CLI run each fold uses exactly its own `S_k`, with the withheld set equal to its
  complement.
- **CLI:** the command runs end to end on the synthetic fixture, `--resume` skips finished
  work, and `--accel-unit g` is refused.

## Limitations (stage A)

- **Sensor-to-segment alignment:** unknown. FK uses a standing calibration, so V3 is a
  derived quantity.
- **Joint statistics:** global (not per activity), because activity labels are unknown at
  deployment.
- **Gyro and orientation factors:** treated as independent. The P0
  `diag_orientation_independence.py` result decides whether that assumption must be
  revisited.
- **Solver convergence:** the strict criterion is an orientation update below 1e-7 rad.
  Withheld segments with two disagreeing joint targets form large-residual problems, where
  Gauss–Newton converges only linearly. They get up to 30 iterations, and the largest final
  update is reported (about 1e-4 rad on the synthetic fixture, i.e. below 0.01°).
- **Stage B:** the 6D Sensor-Dropout BiGRU is a separate optional step.

## Models T3-M0 to T3-M3 and the external-prior interface (P0-calibrated update)

| ID | Method | Acceleration |
|---|---|---|
| T3-M0 | `copy_parent`: the parent's orientation | no |
| T3-M1 | `mean_relative`: parent orientation × training mean relative rotation | no |
| T3-M2 | `wnoj`: SO(3) WNOJ with joint factors from observed or rebuilt neighbours | no (`--accel-unit none`) |
| T3-M3 | `wnoj_acc`: T3-M2 plus a local robust link-acceleration factor | `--accel-unit g` only |

These Task 3 IDs are unrelated to the Task 4 spine models M1–M3.

**How T3-M3 works** (`src/task3/accel_factor.py`):

- **Where it applies.** Only on a withheld segment w with an observed parent p *and* an
  observed child c. Otherwise the factor is skipped, and T3-M3 equals T3-M2.
- **Residual.** `e_a = da_obs − (G_w) R_w b_w`, with `G = [α]× + [ω]×²`. The observed link
  acceleration is `da_obs = a(J_wc) − a(J_pw)`. Joint accelerations come from the parent and
  child sensors through their WNOJ-smoothed (ω, α) and half-bone lever arms; this assumes
  mid-segment sensors.
- **Units.** Accelerations are the dataset's gravity-compensated free acceleration in g
  (P0 verdict), converted with `ACCEL_G_TO_SI = 9.80665`. Gravity is never removed again.
- **Robustness.** Impact frames are masked (|da_obs| > 30 m/s² or parent/child |α| >
  150 rad/s²); the remaining frames use IRLS Huber weights (k = 1.345). The analytic
  Jacobian is checked to < 1e-8.
- **Tuning.** `sigma_acc` is chosen per fold on validation participants from {1, 2, 4} m/s².
- **Reporting.** `summary.json` echoes `accel_schema`, `accel_factor_registered: true` and
  `model_ids`, and T3-M2 and T3-M3 are reported side by side.
- **Default mode.** With `--accel-unit none`, accelerometer channels are still never read:
  NaN in every Acc channel leaves the output bitwise identical.

**Caveat.** P0 T3 fitted knee lever arms of only about 7 cm, against about 30 cm expected
from anatomy, so the half-bone lever-arm assumption is uncertain. Report T3-M3 against
T3-M2 as measured, including "no gain" if that is the result.

**External prior (T3-M5).** `withheld_eval.smooth_with_external_prior(r_prior_dict,
gyro_obs_or_none, dt, qc, sigma_prior_deg=4.0)` post-smooths per-frame rotation predictions
(for example from a BiGRU) under the WNOJ prior. It can optionally add gyro factors, but only
for sensors whose gyro is observed.
