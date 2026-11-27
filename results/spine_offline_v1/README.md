# spine_offline_v1 — Task 4 spine observability (offline, model-based)

**Status: Model-based prediction.** Everything here comes from synthetic deadlift postures
and a range-of-motion prior. No recorded data from the main dataset is used, and no
measured gain is claimed for any unmeasured sensor location (problem red line).

## Question

The existing layout has a LowerBack IMU on L5 and an UpperBack IMU between the scapulae,
with no sensor in between. For deadlift lumbar-flexion monitoring, how much of the spine's
internal motion can those two sensors observe, and what does one extra sensor (or a bend
sensor) add?

## Models (`src/spine/spine_kinematics.py`)

Serial chain from the pelvis (sacrum) to the upper back. Every joint is a 3-DoF rotation
`theta = (FE, AR, LB)` in the dataset's frame convention (X left, Y up, Z forward).

| Model | Joints | Joint DoF |
|---|---|---:|
| M1 | L5/S1; one lumped joint for L4/L5 through T6/T7 (rigid trunk above L5) | 6 |
| M2 (main) | L5/S1; lumped lumbar L4/L5..L1/L2; lumped T12/L1 plus thoracic (rigid thorax) | 9 |
| M3 | L5/S1, L4/L5, L3/L4, L2/L3, L1/L2, T12/L1 per level; lumped T11/T12..T6/T7 | 21 |

## Sensor configurations (`src/spine/spine_observation.py`)

| Config | Sensors |
|---|---|
| C0 | L5 + UpperBack (existing) |
| C1 / C1′ / C1″ | C0 + one IMU at L1 / T12 / sacrum |
| C2a / C2b | C0 + flexible bend sensor along L5–L1: one channel / one channel per joint. C2a is an in-plane observation of integrated curvature (the net tangent turning angle in the sagittal plane over the span), linearised in this chain model as the sum of the FE components of the joints in the span. It is not a general 3-D sum of joint flexions |

C3 (plantar-pressure insoles) is a kinetic extension and is analysed separately, not
ranked with orientation sensors.

## Information (`src/spine/spine_information.py`)

- **I_meas = J^T R^-1 J** is built from sensor rows only, after marginalising the pelvis
  world orientation (Schur complement). Only this matrix measures a sensor's contribution.
- **H_post = I_meas + H_prior** adds the range-of-motion prior. It shows what becomes
  solvable *with* anatomical assumptions and is **not** a sensor gain.
- Three statistics are kept strictly apart:
  1. **numerical rank** of I_meas: structural, independent information directions;
  2. **measurement-only 5°-precision eigen-directions**: eigenvalues of I_meas with
     `1/sqrt(lambda) < 5°`;
  3. **posterior standard deviation**: from H_post, so it includes the prior.
- **Trajectory analysis.** M3, one repetition, 41 frames. It adds IMU free-acceleration rows
  (the second time difference of the mount-point position). Accelerations are kept
  dimensionless, `kappa = sigma_a * tau^2 / L`, until the dataset's acceleration unit is
  resolved.
- **Slow-mode basis.** Joint trajectories are projected onto slow modes. The basis is
  DCT-II: `b_k(t_n) = cos(pi k t_n)`, `t_n = (n + 0.5)/N` on dimensionless repetition time
  [0, 1], `k = 0..K-1`. Columns are normalised to unit Euclidean norm (orthonormal), and the
  even (Neumann) extension is implicit at both ends. `trajectory_information.json` records
  `basis_definition`, `basis_normalization`, `boundary_condition` and `rank_tolerance` for
  every case.
- **K sensitivity.** K = 6 is the main setting, with a sweep over K = 4, 6, 8, 10.

## Conventions

- **Orientation residual.** `e_R(q) = Log(R_meas^T R_pred(q))^vee`, evaluated at the
  linearisation point.
- **Perturbations.** Pelvis orientation uses the right retraction `R Exp(d)`. Joint angles
  are rotation vectors with an additive chart; the analytic Jacobian carries the SO(3)
  right Jacobian.
- **Log branch.** The branch near pi is never reached, because spine rotations stay well
  below 60°.
- **Units.** States are dimensionless radians. The trajectory analysis uses spine-span
  length units and repetition time units, with kappa dimensionless.
- **Information matrix.** `J^T R^-1 J` is the Gauss–Newton PSD approximation: exact at zero
  residual, with second-order terms dropped.
- **Gauge.** L5 is the relative-orientation reference node. The pelvis (root) world
  orientation is a nuisance/reference variable eliminated by the Schur complement. The L5
  IMU alone cannot separate the pelvis orientation from the internal L5/S1 joint angle,
  which is why L5/S1 stays unobserved without a sacrum IMU.
- **Rank metadata.** Every (model × configuration) records `state_dimension`,
  `measurement_dimension`, `gauge_dimension` (3, the pelvis world orientation),
  `theoretical_rank_upper_bound = min(3n, 3·(distinct IMU segments − 1) + bend channels)`,
  `numerical_rank` and `rank_tolerance` (1e-9 × largest eigenvalue). The bound is attained
  in all 16 representable cases (`summary.md`).
- **Jacobian check.** float64 central differences at h = 1e-5, 1e-6, 1e-7 give a maximum
  relative error of 3e-12 to 6e-10 (`jacobian_check.json`). The error grows as h shrinks,
  which is the round-off (eps/h) regime, so the analytic Jacobian is exact to working
  precision.

## Main predictions (`summary.md`, figures)

1. **Orientation-only rank follows the analytic bounds.** C0 gives 3 for every model;
   one more IMU gives 6; a single bend channel gives +1.
2. **Which sensor helps depends on which segment is assumed flexible.**
   - In M2 the T12 IMU sits on the same rigid thorax as UpperBack, so it adds nothing.
   - In M1 neither L1 nor T12 adds anything.
   - Only M3 separates them. Any statement that "L1 beats T12" has to name its model.
3. **Identifiability from sensors alone** (lumbar flexion, M2/M3):
   - **C1 (L1 IMU)** identifies L1–L5 flexion.
   - **C1″ (sacrum IMU)** identifies L5/S1 flexion.
   - **C2a (bend sensor L5–L1)** identifies L1–L5 flexion with one channel.
   - **No single added sensor identifies total L1–S1 flexion.** That needs L1 + sacrum,
     or the bend sensor + sacrum.
4. **Bend sensor spanning L5–UpperBack.** It would duplicate the two existing IMUs, so it
   is excluded by construction. C2a only helps because it spans the L5–L1 sub-interval,
   and it is not representable in M1.
5. **Acceleration gives weak observability.** Over one repetition at K = 6, adding C0's
   free acceleration raises the measurement-only 5°-precision eigen-directions from 18 to
   22–28 (kappa 2.0–0.1). Adding an L1 IMU raises them to 36.
   - The conclusion holds for every K in {4, 6, 8, 10}.
   - With orientation only, the count is exactly 3K for C0 and 6K for C1.
   - With acceleration it is 3K + (0…18), always below the 6K of the L1 IMU.
   - The numerical rank is larger than 3K/6K because the Jacobian varies with posture.
     Those extra directions are very weak (sensor-only std > 30°).
6. **Bend-sensor noise.** σ_bend ∈ {0.5, 1, 2, 3, 5}°:
   - C2a measures L1–L5 flexion with measurement-only std equal to σ_bend.
   - M3 C2b gives 2σ_bend, because four channels add up.
   - 2° is only a nominal design scenario, not a literature-wide standard deviation
     (`sensitivity.json`).
7. **Prior scale.** The ROM-derived nominal prior scale is swept at 0.5×, 1.0× and 1.5×.
   - Quantities that sensors alone cannot identify scale with the prior: L5/S1 under C1
     goes from 2.12° to 6.38°.
   - Quantities that are identified barely move: L1–L5 under C1 goes from 2.13° to 2.70°.
   - This separates the measurement contribution from the prior directly.

## Assumptions to check before citing

- **ROM prior.** Per-level ranges are the representative values of White & Panjabi,
  *Clinical Biomechanics of the Spine*, 2nd ed. (1990). The L5/S1 values (17°/3°/1°) and the
  L1–L5 values have been checked against a secondary reproduction of the original table.
  The mapping σ_FE = ROM/4 and σ_one-side = ROM/2 is a **ROM-derived nominal prior scale**,
  not an empirical variance, and is swept at 0.5× / 1.0× / 1.5×.
- **Noise.** Orientation noise is 2° per axis and bend noise 2° per channel. Both are
  nominal design values; bend noise is swept from 0.5° to 5°. Rank results do not depend on
  noise; the posterior std values do.
- **Geometry.** Vertebral levels have equal height and the sensor sits at a posterior
  skin offset of 0.12 span units. These only affect the acceleration analysis.
- **Posture.** The deadlift posture is synthetic: 25° lumbar and 10° thoracic flexion
  distributed by FE range, plus a 50° hip hinge. Ranks are generic in posture.

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy sympy matplotlib pytest
.venv/bin/python -m pytest -q tests/test_spine_offline.py
PYTHONPATH=src .venv/bin/python -m spine.run_offline --out results/spine_offline_v1
```

Tests:

- Analytic orientation and position Jacobians match central differences (< 1e-6), with a
  float64 multi-step (1e-5, 1e-6, 1e-7) relative-error check.
- The rank metadata bound holds and is attained.
- The SO(3) right Jacobian matches a SymPy derivation.
- Analytic rank bounds hold.
- I_meas → 0 as noise → ∞.
- The identifiability pattern matches item 3 above.
- Acceleration strictly adds information.
- The DCT-II basis is orthonormal, and the C0 precision count is 3K.
- The bend-noise sweep reproduces σ and 2σ exactly.
- Prior-scale behaviour: the unidentified quantity scales 3×; the identified quantity
  scales by less than 1.5×.
- The full verbose run is saved in `pytest_verification.log`.
