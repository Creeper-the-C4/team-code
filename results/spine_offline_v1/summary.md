# spine_offline_v1 summary

> Model-based prediction (synthetic deadlift posture; not measured data)

Settings: posture phase 0.5, orientation noise 2.0 deg/axis, bend noise 2.0 deg.

## Sensor-only rank and posterior std of lumbar flexion

"determined" = eigen-direction with sensor-only std < 5 deg.

| model | config | joint DoF | rank I_meas | determined dirs | logdet+ I_meas | rank H_post | L1-S1 post std (deg) | L1-L5 post std | L5/S1 post std | identifiable from sensors alone |
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

## Slow joint-trajectory information over one repetition (M3)

Joint trajectories restricted to the first 6 cosine modes (slow posture changes); kappa = sigma_a tau^2 / L.

| case | dims | numerical rank | determined dirs (std < 5 deg) | largest eig | smallest positive eig |
|---|---:|---:|---:|---:|---:|
| C0_orientation_only | 126 | 30 | 18 | 2.46e+03 | 0.000131 |
| C1_orientation_only | 126 | 56 | 36 | 2.59e+03 | 1.54e-05 |
| C0_plus_acc_kappa_0.1 | 126 | 47 | 28 | 7.98e+04 | 9.82e-05 |
| C0_plus_acc_kappa_0.5 | 126 | 48 | 25 | 6.47e+03 | 1.63e-05 |
| C0_plus_acc_kappa_2.0 | 126 | 49 | 22 | 2.92e+03 | 5.08e-06 |
