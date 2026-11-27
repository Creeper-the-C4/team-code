"""Synthetic fixture with the independent_v1 cache layout, for testing diagnostics only.

Writes <root>/subjects/PXX/{signals.npy, labels.npy, trials.parquet,
windows_2s.parquet, windows_4s.parquet, complete.json} and <root>/splits.json.
Signals come from simple world-frame kinematics with a known ground truth:
free acceleration stored in a chosen unit (g or m/s^2), world-frame gyro in
rad/s, xyzw quaternions with random sign flips, an exact knee hinge, ballistic
jumps, and feet that are stationary in stance. Never use it for results.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import FS, G, SENSORS

TRUNK = ('LowerBack', 'UpperBack', 'Head', 'RightShoulder', 'LeftShoulder', 'RightUpperArm',
         'LeftUpperArm', 'RightForeArm', 'LeftForeArm', 'RightWrist', 'LeftWrist')
TRUNK_OFFSET = {'LowerBack': (0, 0.05, -0.1), 'UpperBack': (0, 0.45, -0.1),
                'Head': (0, 0.7, -0.05), 'RightShoulder': (-0.15, 0.5, -0.05),
                'LeftShoulder': (0.15, 0.5, -0.05), 'RightUpperArm': (-0.2, 0.35, 0),
                'LeftUpperArm': (0.2, 0.35, 0), 'RightForeArm': (-0.22, 0.1, 0.02),
                'LeftForeArm': (0.22, 0.1, 0.02), 'RightWrist': (-0.22, -0.02, 0.05),
                'LeftWrist': (0.22, -0.02, 0.05)}
PLAN = [(3, 20), (4, 24), (5, 24), (6, 24), (7, 20), (8, 20), (11, 16), (2, 16), (12, 24),
        (9, 16), (10, 16), (1, 16)]
CADENCE = {4: 1.6, 5: 1.8, 6: 2.0, 7: 1.5, 8: 1.6, 10: 2.8}   # steps per second
STEP_HEIGHT = 0.17


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(np.shape(a) + (3, 3))
    out[..., 0, 0] = 1
    out[..., 1, 1], out[..., 1, 2] = c, -s
    out[..., 2, 1], out[..., 2, 2] = s, c
    return out


def rot_to_quat(r):
    """Rotation matrices (..., 3, 3) to xyzw quaternions."""
    m = r
    w = 0.5 * np.sqrt(np.maximum(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], 1e-12))
    x = (m[..., 2, 1] - m[..., 1, 2]) / (4 * w)
    y = (m[..., 0, 2] - m[..., 2, 0]) / (4 * w)
    z = (m[..., 1, 0] - m[..., 0, 1]) / (4 * w)
    return np.stack([x, y, z, w], axis=-1)


def min_jerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def feet_positions(t, f_stride, stride_len, rise_per_stride, phase):
    """Foot placement: stationary in stance (60 %), minimum-jerk swing with a lift."""
    u = t * f_stride + phase
    k = np.floor(u)
    s = (u - k - 0.6) / 0.4
    prog = k + min_jerk(s)
    z = prog * stride_len
    y = 0.05 + prog * rise_per_stride + 0.08 * np.sin(np.pi * np.clip(s, 0, 1))
    swing_pitch = 0.3 * np.sin(np.pi * np.clip(s, 0, 1))
    return z, y, swing_pitch


def simulate(label, dur, rng, legs, cadence_factor):
    n = int(dur * FS)
    t = np.arange(n) / FS
    pelvis = np.zeros((n, 3))
    pelvis[:, 1] = 1.0
    trunk_pitch = np.zeros(n)
    thigh = np.zeros(n)          # thigh pitch about X (sagittal)
    knee = np.zeros(n)           # knee flexion, exact hinge about the thigh X axis
    feet = {}
    l_th, l_sh = legs
    if label in CADENCE:
        f_step = CADENCE[label] * cadence_factor
        f_stride = f_step / 2
        stride = 1.4 * (f_step / 1.8) if label in (4, 5, 6, 10) else 0.6
        rise = {7: 2 * STEP_HEIGHT, 8: -2 * STEP_HEIGHT}.get(label, 0.0)
        w = 2 * np.pi * f_stride
        pelvis[:, 2] = stride * f_stride * t
        pelvis[:, 1] += 0.02 * np.sin(2 * w * t) + rise * f_stride * t
        thigh = -0.35 * np.sin(w * t)
        knee = 0.5 * (1 - np.cos(w * t + 0.6))
        for side, ph in (('Right', 0.0), ('Left', 0.5)):
            z, y, p = feet_positions(t, f_stride, stride, rise, ph)
            feet[side] = (z, y, p)
    elif label == 11:            # ballistic hops: flight 0.32 s of a 0.65 s period
        period, tf = 0.65, 0.32
        ts = period - tf
        a_s = G * tf / ts
        u = np.mod(t, period)
        acc = np.where(u < ts, a_s, -G)
        vel = np.cumsum(acc) / FS
        vel -= vel.mean()
        pelvis[:, 1] += np.cumsum(vel) / FS
    elif label == 12:            # ergometer rowing: seat slides, legs flex
        w = 2 * np.pi * 0.45
        pelvis[:, 1] = 0.45
        pelvis[:, 2] = 0.3 * np.sin(w * t)
        thigh = -1.2 + 0.5 * np.sin(w * t)
        knee = 1.2 + 0.9 * np.sin(w * t)
        trunk_pitch = 0.3 * np.sin(w * t)
    elif label == 9:
        w = 2 * np.pi * 1.2
        pelvis[:, 1] = 0.9
        thigh = -0.9 + 0.35 * np.sin(w * t)
        knee = 1.2 + 0.5 * np.sin(w * t + 0.8)
    elif label == 2:
        pelvis[:, 1] = 0.45
        thigh[:], knee[:] = -np.pi / 2, np.pi / 2
    elif label == 1:
        pelvis[:, 1] = 0.2
        trunk_pitch[:] = -np.pi / 2
        thigh[:] = -np.pi / 2
    else:                        # standing / others: small sway
        pelvis[:, 2] = 0.01 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))
        if label == 0:
            thigh = 0.2 * np.sin(2 * np.pi * 0.5 * t)
    pos, rot = {}, {}
    r_trunk = rot_x(trunk_pitch)
    for name in TRUNK:
        pos[name] = pelvis + np.einsum('nij,j->ni', r_trunk, TRUNK_OFFSET[name])
        rot[name] = r_trunk
    for side, sx in (('Right', -0.1), ('Left', 0.1)):
        ph = np.pi if side == 'Left' and label in CADENCE else 0.0
        th = thigh if ph == 0 else -thigh
        kn = knee if ph == 0 else 0.5 * (1 - np.cos(2 * np.pi * CADENCE.get(label, 1) *
                                                     cadence_factor / 2 * t + 0.6 + np.pi))
        if label not in CADENCE:
            th, kn = thigh, knee
        r_th = rot_x(th)
        r_sh = r_th @ rot_x(kn)
        hip = pelvis + np.array([sx, -0.05, 0])
        knee_pos = hip + np.einsum('nij,j->ni', r_th, (0, -l_th, 0))
        pos[f'{side}Thigh'] = hip + np.einsum('nij,j->ni', r_th, (0, -0.12, 0.06))
        pos[f'{side}Shank'] = knee_pos + np.einsum('nij,j->ni', r_sh, (0, -0.08, 0.05))
        rot[f'{side}Thigh'], rot[f'{side}Shank'] = r_th, r_sh
        if side in feet:
            z, y, p = feet[side]
            pos[f'{side}Foot'] = np.stack([np.full(n, sx), y, z], axis=1)
            rot[f'{side}Foot'] = rot_x(p)
        else:
            pos[f'{side}Foot'] = knee_pos + np.einsum('nij,j->ni', r_sh, (0, -l_sh, 0.08))
            rot[f'{side}Foot'] = r_sh
    return pos, rot


def to_signals(pos, rot, rng, acc_scale, smooth_frames, gyro_noise, wobble):
    n = next(iter(pos.values())).shape[0]
    out = np.zeros((n, len(SENSORS), 10), dtype=np.float32)
    for i, name in enumerate(SENSORS):
        a = np.gradient(np.gradient(pos[name], axis=0), axis=0) * FS**2
        if smooth_frames > 1:
            k = np.ones(smooth_frames) / smooth_frames
            a = np.stack([np.convolve(a[:, j], k, mode='same') for j in range(3)], axis=1)
        a += rng.normal(0, 0.05, a.shape)
        r = rot[name]
        dr = np.gradient(r, axis=0) * FS
        w_hat = np.einsum('nij,nkj->nik', dr, r)              # R_dot R^T
        omega = np.stack([w_hat[:, 2, 1], w_hat[:, 0, 2], w_hat[:, 1, 0]], axis=1)
        omega += rng.normal(0, gyro_noise, omega.shape)
        if wobble and 'Shank' in name:     # soft-tissue wobble seen by the gyro only
            omega += wobble * np.stack([np.sin(2 * np.pi * 7.3 * np.arange(n) / FS),
                                        np.cos(2 * np.pi * 5.1 * np.arange(n) / FS),
                                        np.zeros(n)], axis=1)
        q = rot_to_quat(r)
        flip = rng.random(n) < 0.03
        q[flip] *= -1
        out[:, i, 0:3] = a / acc_scale
        out[:, i, 3:6] = omega
        out[:, i, 6:10] = q
    return out


def window_table(subject, trials, labels, width):
    rows = []
    stride = width // 2
    for tr in trials:
        for s in range(tr['source_start'], tr['source_stop_exclusive'] - width + 1, stride):
            lab = labels[s:s + width]
            counts = np.bincount(lab, minlength=13)
            maj = int(np.argmax(counts))
            rows.append({'subject': subject, 'trial': tr['trial'], 'source_start': s,
                         'source_stop_exclusive': s + width, 'width_frames': width,
                         'stride_frames': stride, 'majority_label': np.int8(maj),
                         'center_label': np.int8(lab[width // 2]),
                         'purity': counts[maj] / width, 'mixed_labels': bool((counts > 0).sum() > 1),
                         'label_changes': int(np.sum(lab[1:] != lab[:-1])),
                         'invalid_signal_frames': 0, 'tie_for_majority': False,
                         **{f'label_count_{k}': np.int32(counts[k]) for k in range(13)}})
    return pd.DataFrame(rows)


def build(root, subjects=(1, 2, 3, 4), acc_unit='g', smooth_frames=0, seed=0,
          gyro_noise=0.02, wobble=0.0):
    root = Path(root)
    acc_scale = G if acc_unit == 'g' else 1.0
    rng = np.random.default_rng(seed)
    for subject in subjects:
        legs = (rng.uniform(0.42, 0.5), rng.uniform(0.38, 0.45))
        cadence = rng.uniform(0.92, 1.08)
        sig, lab, trials, start = [], [], [], 0
        for k, (label, dur) in enumerate(PLAN):
            pos, rot = simulate(label, dur, rng, legs, cadence)
            s = to_signals(pos, rot, rng, acc_scale, smooth_frames, gyro_noise, wobble)
            labels = np.full(len(s), label, dtype=np.int8)
            if k % 3 == 1:
                labels[:int(2 * FS)] = 0          # leading transition labelled Others
            counts = np.bincount(labels, minlength=13)
            trials.append({'subject': subject, 'trial': k, 'source_start': start,
                           'source_end_inclusive': start + len(s) - 1,
                           'source_stop_exclusive': start + len(s), 'frames': len(s),
                           'duration_s': len(s) / FS, 'majority_label': int(np.argmax(counts)),
                           'purity': counts.max() / len(s),
                           **{f'label_count_{c}': int(counts[c]) for c in range(13)}})
            sig.append(s)
            lab.append(labels)
            start += len(s)
        d = root / 'subjects' / f'P{subject:02d}'
        d.mkdir(parents=True, exist_ok=True)
        labels = np.concatenate(lab)
        np.save(d / 'signals.npy', np.concatenate(sig))
        np.save(d / 'labels.npy', labels)
        pd.DataFrame(trials).to_parquet(d / 'trials.parquet')
        for secs in (2, 4):
            window_table(subject, trials, labels, int(secs * FS)).to_parquet(
                d / f'windows_{secs}s.parquet')
        (d / 'complete.json').write_text(json.dumps({'subject': subject, 'synthetic': True}))
    subs = list(subjects)
    folds = [{'fold': i, 'test': [s], 'validation': [subs[(i + 1) % len(subs)]],
              'train': [x for x in subs if x not in (s, subs[(i + 1) % len(subs)])]}
             for i, s in enumerate(subs)]
    (root / 'splits.json').write_text(json.dumps({'purpose': 'synthetic', 'folds': folds}))
    meta = {'synthetic': True, 'acc_unit': acc_unit, 'smooth_frames': smooth_frames,
            'seed': seed, 'gyro_noise': gyro_noise, 'wobble': wobble,
            'step_height_m': STEP_HEIGHT, 'cadence_steps_per_s': CADENCE}
    (root / 'fixture.json').write_text(json.dumps(meta, indent=1))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--acc-unit', choices=['g', 'mps2'], default='g')
    ap.add_argument('--smooth-frames', type=int, default=0)
    ap.add_argument('--wobble', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    build(args.root, acc_unit=args.acc_unit, smooth_frames=args.smooth_frames,
          seed=args.seed, wobble=args.wobble)


if __name__ == '__main__':
    main()
