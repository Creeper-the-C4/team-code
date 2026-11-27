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
| C2a / C2b | C0 + flexible bend sensor along L5–L1: one channel / one channel per joint |

C3 (plantar-pressure insoles) is a kinetic extension and is analysed separately, not
ranked with orientation sensors.

## Information (`src/spine/spine_information.py`)

- **I_meas = J^T R^-1 J** is built from sensor rows only, after marginalising the pelvis
  world orientation (Schur complement). Only this matrix measures a sensor's contribution.
- **H_post = I_meas + H_prior** adds the range-of-motion prior. It shows what becomes
  solvable *with* anatomical assumptions and is **not** a sensor gain.
- A direction counts as **determined** when its sensor-only standard deviation is below 5°.
- Trajectory analysis (M3, one repetition, 41 frames) adds IMU free-acceleration rows
  (second time difference of the mount-point position) and projects joint trajectories
  onto the first 6 cosine modes (slow posture changes). Accelerations are dimensionless,
  `kappa = sigma_a * tau^2 / L`, until the dataset's acceleration unit is resolved.

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
5. **Acceleration gives weak observability.** Over one repetition (slow modes), adding
   C0's free acceleration raises the determined directions from 18 to 22–28 (kappa
   2.0–0.1). Adding an L1 IMU raises them to 36.

## Assumptions to check before citing

- **ROM prior.** Per-level ranges are the representative values of White & Panjabi,
  *Clinical Biomechanics of the Spine*, 2nd ed. (1990). They are entered from secondary
  knowledge and **must be checked against the original table**. FE total is mapped to
  4 sigma; one-side LB/AR to 2 sigma.
- **Noise.** Orientation noise is 2° per axis and bend noise 2° per channel (nominal
  values). Rank results do not depend on these; the posterior std values do.
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

- Analytic orientation and position Jacobians match central differences (< 1e-6).
- The SO(3) right Jacobian matches a SymPy derivation.
- Analytic rank bounds hold.
- I_meas → 0 as noise → ∞.
- The identifiability pattern matches item 3 above.
- Acceleration strictly adds information.
