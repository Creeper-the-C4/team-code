# P0 diagnostics (run before any Task 3 SI-unit physics)

Three read-only checks on the `independent_v1` cache. They read data only through
`scripts/preprocessing/independent_data_access.py`, never modify the cache, and write
small JSON + Markdown reports.

| Script | Question | Decision rule (fixed before running) |
|---|---|---|
| `diag_accel_unit.py` | What is stored in the acceleration channels (free acceleration or specific force), and in which unit? | **T1** jump signature on the torso sensors, three hypotheses (never a forced binary choice): a free-acceleration signature (segment mean ≈ 0) with a flight plateau in [-1.25, -0.75] → `g`, or in [-12, -7.5] → `mps2`; near-zero flat flight between landings with a positive segment mean → `specific_force`; anything else → `filtered_or_unresolved`. **T2** foot zero-velocity stride integration: stairs stride rise in [0.20, 0.50] m **and** walking stride in [0.80, 2.00] m. **Verdict:** `g`/`mps2` only if T1 and T2 agree. T2 fixes the scale and is insensitive to a constant bias such as gravity; T1 decides whether gravity was removed. **Supporting evidence only:** T3 knee joint centre (Seel et al. 2014, Eq. (7); Seel's o_i = -r_i), T4 static tilt, T6 gyro range and spike widths. |
| `diag_orientation_independence.py` | Are quaternions per-sensor outputs or skeleton-solved? Are gyro and quaternion errors correlated? | T5: knee/elbow relative angular velocity in the proximal frame, compared via second-moment ratio sqrt(l2/l1) from the gyro and from quaternion differences. `skeleton_constrained_suspected` if the quaternion ratio is < 0.05 and the gyro ratio is > 2× the quaternion ratio. Also: gyro-vs-quaternion residual fit, residual autocorrelation, thigh–shank residual correlation. |
| `diag_subject_protocol.py` | Batch heterogeneity, split validity, label durations, walking cadence | Trial structure grouped as P01–P24 / P25 / P26–P30 (the paper does not state that these are separate collection batches; the grouping is descriptive only); split audit (disjoint roles, each subject tested once, class coverage); within-trial run lengths, giving tau for the sticky HMM `p_stay = exp(-dt/tau)`; step frequency per speed class from the thigh gyro in its own frame (8 s chunks, autocorrelation + parabolic interpolation) and the slow < moderate < brisk check; kinematic speed `2L sin(dθ/2)·f_step` against the protocol's fixed treadmill speeds 0.9/1.2/1.5 m/s; optional location of `rowing -> sitting` errors inside trials. |

## Run on the cloud cache

```bash
cd <team-code checkout>
PY=/root/autodl-tmp/C_Problem/.venv/bin/python
DATA=/root/autodl-tmp/C_Problem/data/processed/independent_v1
OUT=results/p0_diagnostics
$PY scripts/diagnostics/diag_accel_unit.py --data-root $DATA --out $OUT
$PY scripts/diagnostics/diag_orientation_independence.py --data-root $DATA --out $OUT
$PY scripts/diagnostics/diag_subject_protocol.py --data-root $DATA --out $OUT \
    --splits results/preprocessing/independent_v1/subject_splits_6fold.json \
    --body-dims results/preprocessing/independent_v1/body_dimensions_m.csv \
    --oof /root/autodl-tmp/C_Problem/results/task1_lightgbm_baseline_v1/oof_predictions.parquet
```

The defaults read up to 3 trials per activity per subject (`--max-trials`). Use
`--subjects 1 7 25 27` for a quick look. Only numpy, scipy, pandas and pyarrow are needed,
all already in the cloud environment.

## Synthetic fixture (tests only)

`make_synthetic_fixture.py` writes a small cache in the same layout from simple world-frame
kinematics with a known ground truth: chosen acceleration unit, exact knee hinge, ballistic
jumps, stationary stance feet, and xyzw quaternions with random sign flips. It exists only to
test the scripts. **Never use it for results.**

```bash
python -m pytest -q tests/test_diag_p0.py
```

The tests check:

- A g fixture gives verdict `g`, an m/s² fixture gives `mps2`, and a specific-force fixture
  gives `specific_force`.
- A heavily smoothed fixture never flips the verdict.
- A hinge-perfect quaternion with a wobbling gyro is flagged.
- Planted cadences are recovered and ordered, and kinematic speeds are ordered.
- A broken split file fails the audit.
