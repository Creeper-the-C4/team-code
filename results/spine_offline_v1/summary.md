# spine_offline_v1 summary

> Model-based prediction (synthetic deadlift posture; not measured data)

Settings: posture phase 0.5, orientation noise 2.0 deg/axis, bend noise 2.0 deg.

## Sensor-only rank and posterior std of lumbar flexion

Three distinct statistics: numerical rank of I_meas (structural directions); measurement-only 5-deg-precision eigen-directions (eigenvalues of I_meas with 1/sqrt(lambda) < 5 deg); posterior std (I_meas plus the ROM-derived nominal prior).

| model | config | joint DoF | rank I_meas | 5-deg-precision dirs | logdet+ I_meas | rank H_post | L1-S1 post std (deg) | L1-L5 post std | L5/S1 post std | identifiable from sensors alone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M1 | C0 | 6 | 3 | 3 | 18.04 | 6 | - | - | 4.25 | none |
| M1 | C1 | 6 | 3 | 3 | 18.90 | 6 | - | - | 4.25 | none |
| M1 | C1p | 6 | 3 | 3 | 18.90 | 6 | - | - | 4.25 | none |
| M1 | C1pp | 6 | 6 | 6 | 36.95 | 6 | - | - | 2.34 | L5/S1 |
| M1 | C2a | - | n/a | - | - | - | - | - | - | span does not align with model joints |
| M1 | C2b | - | n/a | - | - | - | - | - | - | span does not align with model joints |
| M2 | C0 | 9 | 3 | 3 | 20.13 | 9 | 6.35 | 4.72 | 4.25 | none |
| M2 | C1 | 9 | 6 | 6 | 36.96 | 9 | 4.97 | 2.57 | 4.25 | L1-L5 |
| M2 | C1p | 9 | 3 | 3 | 20.99 | 9 | 6.30 | 4.65 | 4.25 | none |
| M2 | C1pp | 9 | 6 | 6 | 39.04 | 9 | 4.94 | 4.70 | 2.34 | L5/S1 |
| M2 | C2a | 9 | 4 | 4 | 26.14 | 9 | 4.63 | 1.84 | 4.25 | L1-L5 |
| M2 | C2b | 9 | 4 | 4 | 26.14 | 9 | 4.63 | 1.84 | 4.25 | L1-L5 |
| M3 | C0 | 21 | 3 | 3 | 23.43 | 21 | 6.35 | 4.72 | 4.25 | none |
| M3 | C1 | 21 | 6 | 6 | 43.20 | 21 | 4.97 | 2.57 | 4.25 | L1-L5 |
| M3 | C1p | 21 | 6 | 6 | 41.79 | 21 | 5.53 | 3.53 | 4.25 | none |
| M3 | C1pp | 21 | 6 | 6 | 42.34 | 21 | 4.94 | 4.70 | 2.34 | L5/S1 |
| M3 | C2a | 21 | 4 | 4 | 30.42 | 21 | 4.63 | 1.84 | 4.25 | L1-L5 |
| M3 | C2b | 21 | 7 | 7 | 49.17 | 21 | 5.23 | 3.04 | 4.25 | L1-L5 |

## Rank metadata (per model and configuration)

Bound: min(3n, 3 x (distinct IMU segments - 1) + bend channels); gauge = pelvis world orientation (marginalised).

| model | config | state dim | measurement dim | gauge dim | rank upper bound | numerical rank | rank tolerance |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | C0 | 9 | 6 | 3 | 3 | 3 | 4.1e-07 |
| M1 | C1 | 9 | 9 | 3 | 3 | 3 | 5.47e-07 |
| M1 | C1p | 9 | 9 | 3 | 3 | 3 | 5.47e-07 |
| M1 | C1pp | 9 | 9 | 3 | 6 | 6 | 8.21e-07 |
| M2 | C0 | 12 | 6 | 3 | 3 | 3 | 8.21e-07 |
| M2 | C1 | 12 | 9 | 3 | 6 | 6 | 8.21e-07 |
| M2 | C1p | 12 | 9 | 3 | 3 | 3 | 1.09e-06 |
| M2 | C1pp | 12 | 9 | 3 | 6 | 6 | 1.29e-06 |
| M2 | C2a | 12 | 7 | 3 | 4 | 4 | 1.4e-06 |
| M2 | C2b | 12 | 7 | 3 | 4 | 4 | 1.4e-06 |
| M3 | C0 | 24 | 6 | 3 | 3 | 3 | 2.46e-06 |
| M3 | C1 | 24 | 9 | 3 | 6 | 6 | 2.59e-06 |
| M3 | C1p | 24 | 9 | 3 | 6 | 6 | 2.9e-06 |
| M3 | C1pp | 24 | 9 | 3 | 6 | 6 | 3.44e-06 |
| M3 | C2a | 24 | 7 | 3 | 4 | 4 | 5.23e-06 |
| M3 | C2b | 24 | 10 | 3 | 7 | 7 | 3.06e-06 |

## Slow joint-trajectory information over one repetition (M3)

Joint trajectories restricted to the first 6 cosine modes (slow posture changes); kappa = sigma_a tau^2 / L.

Basis: DCT-II: b_k(t_n) = cos(pi k t_n), t_n = (n + 0.5) / N, n = 0..N-1, k = 0..K-1 (k = 0 is the constant mode); each column scaled to unit Euclidean norm over the N frames (orthonormal columns); even (Neumann, zero-slope) extension at both ends, implicit in DCT-II; time dimensionless repetition time [0, 1].

| case | dims | numerical rank | rank tolerance | 5-deg-precision dirs | largest eig | smallest positive eig |
|---|---:|---:|---:|---:|---:|---:|
| C0_orientation_only | 126 | 30 | 2.46e-06 | 18 | 2.46e+03 | 0.000131 |
| C1_orientation_only | 126 | 56 | 2.59e-06 | 36 | 2.59e+03 | 1.54e-05 |
| C0_plus_acc_kappa_0.1 | 126 | 47 | 7.98e-05 | 28 | 7.98e+04 | 9.82e-05 |
| C0_plus_acc_kappa_0.5 | 126 | 48 | 6.47e-06 | 25 | 6.47e+03 | 1.63e-05 |
| C0_plus_acc_kappa_2.0 | 126 | 49 | 2.92e-06 | 22 | 2.92e+03 | 5.08e-06 |

## Slow-mode count sensitivity (M3 trajectory)

Each cell: numerical rank / 5-deg-precision directions. With a posture-independent Jacobian the orientation-only rank would be 3K (C0) or 6K (C1); the numerical rank is higher because the Jacobian varies with posture over the repetition, and those extra directions are very weak (sensor-only std > 30 deg), so the 5-deg-precision count equals 3K / 6K exactly.

| case | K=4 | K=6 | K=8 | K=10 |
|---|---:|---:|---:|---:|
| C0_orientation_only | 24 / 12 | 30 / 18 | 36 / 24 | 42 / 30 |
| C1_orientation_only | 44 / 24 | 56 / 36 | 68 / 48 | 80 / 60 |
| C0_plus_acc_kappa_0.1 | 38 / 18 | 47 / 28 | 56 / 38 | 65 / 48 |
| C0_plus_acc_kappa_0.5 | 38 / 15 | 48 / 25 | 59 / 35 | 70 / 45 |
| C0_plus_acc_kappa_2.0 | 38 / 12 | 49 / 22 | 61 / 31 | 72 / 41 |

## Bend-sensor noise sensitivity (L1-L5 flexion, posterior std deg; measurement-only std in parentheses)

Channel noise values are nominal design values, not a literature-wide std.

| model | config | sigma_bend=0.5 | sigma_bend=1 | sigma_bend=2 | sigma_bend=3 | sigma_bend=5 |
|---|---|---:|---:|---:|---:|---:|
| M2 | C2a | 0.50 (0.50) | 0.98 (1.00) | 1.84 (2.00) | 2.53 (3.00) | 3.43 (5.00) |
| M2 | C2b | 0.50 (0.50) | 0.98 (1.00) | 1.84 (2.00) | 2.53 (3.00) | 3.43 (5.00) |
| M3 | C2a | 0.50 (0.50) | 0.98 (1.00) | 1.84 (2.00) | 2.53 (3.00) | 3.43 (5.00) |
| M3 | C2b | 0.98 (1.00) | 1.84 (2.00) | 3.04 (4.00) | 3.70 (6.00) | 4.26 (10.00) |

## ROM-derived nominal prior scale sensitivity (posterior std, deg)

| model | config | prior scale | L1-S1 | L1-L5 | L5/S1 |
|---|---|---:|---:|---:|---:|
| M2 | C0 | 0.5 | 3.41 | 2.66 | 2.12 |
| M2 | C0 | 1.0 | 6.35 | 4.72 | 4.25 |
| M2 | C0 | 1.5 | 9.35 | 6.84 | 6.38 |
| M2 | C1 | 0.5 | 3.01 | 2.13 | 2.12 |
| M2 | C1 | 1.0 | 4.97 | 2.57 | 4.25 |
| M2 | C1 | 1.5 | 6.92 | 2.70 | 6.38 |
| M2 | C1pp | 0.5 | 2.88 | 2.61 | 1.68 |
| M2 | C1pp | 1.0 | 4.94 | 4.70 | 2.34 |
| M2 | C1pp | 1.5 | 7.02 | 6.83 | 2.57 |
| M2 | C2a | 0.5 | 2.66 | 1.60 | 2.12 |
| M2 | C2a | 1.0 | 4.63 | 1.84 | 4.25 |
| M2 | C2a | 1.5 | 6.66 | 1.92 | 6.38 |
| M3 | C0 | 0.5 | 3.41 | 2.66 | 2.12 |
| M3 | C0 | 1.0 | 6.35 | 4.72 | 4.25 |
| M3 | C0 | 1.5 | 9.35 | 6.84 | 6.38 |
| M3 | C1 | 0.5 | 3.01 | 2.13 | 2.12 |
| M3 | C1 | 1.0 | 4.97 | 2.57 | 4.25 |
| M3 | C1 | 1.5 | 6.92 | 2.70 | 6.38 |
| M3 | C1pp | 0.5 | 2.88 | 2.61 | 1.68 |
| M3 | C1pp | 1.0 | 4.94 | 4.70 | 2.34 |
| M3 | C1pp | 1.5 | 7.02 | 6.83 | 2.57 |
| M3 | C2a | 0.5 | 2.66 | 1.60 | 2.12 |
| M3 | C2a | 1.0 | 4.63 | 1.84 | 4.25 |
| M3 | C2a | 1.5 | 6.66 | 1.92 | 6.38 |

## Jacobian check (float64, central difference)

| model | h=1e-05 | h=1e-06 | h=1e-07 |
|---|---:|---:|---:|
| M1 | 3.37e-12 | 2.68e-11 | 3.28e-10 |
| M2 | 4.06e-12 | 4.63e-11 | 4.41e-10 |
| M3 | 6.25e-12 | 7.93e-11 | 5.76e-10 |
