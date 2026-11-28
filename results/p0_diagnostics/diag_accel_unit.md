# diag_accel_unit

**Verdict: g**  (T1 -> g, T2 -> g)

Pre-registered rule: T1 free-acceleration signature with plateau in [-1.25,-0.75] => g, in [-12,-7.5] => m/s^2; near-zero flight with positive segment mean => specific_force; else filtered_or_unresolved. T2 stairs stride rise in [0.20,0.50] m and walking stride in [0.80,2.00] m.

T1 segment mean / range: {'n': 93, 'median': -0.0059964831889735126, 'mad': 0.0035940785146215074}; flat near-zero flight runs: 156

| test | n | median | MAD |
|---|---:|---:|---:|
| T1_flight_plateau | 6585 | -0.9816925128300984 | 0.02459467450777697 |
| T2_stairs_up_rise_raw | 9181 | 0.033832661125608815 | 0.015973228892567283 |
| T2_stairs_down_rise_raw | 7654 | -0.028583576374108403 | 0.015569123647814589 |
| T2_walk_stride_raw | 60186 | 0.12058078096937459 | 0.017090387506617732 |

T2 implied metres: {'g': {'stairs_rise': 0.3317850662274517, 'walk_stride': 1.1824935156933172}, 'mps2': {'stairs_rise': 0.033832661125608815, 'walk_stride': 0.12058078096937459}}

T3 knee lever arms (raw units; metres = raw x scale): r1 {'n': 472, 'median': 0.006850750252471057, 'mad': 0.0018133841387660736}, r2 {'n': 472, 'median': 0.006958084090587453, 'mad': 0.0007230388755939945}
