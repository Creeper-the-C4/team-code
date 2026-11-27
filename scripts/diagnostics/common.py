"""Shared helpers for the P0 diagnostics (data access goes through the shared loader)."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'preprocessing'))
from independent_data_access import SENSORS, SubjectData  # noqa: E402

FS = 60.0
ACTIVITY = {0: 'Others', 1: 'lying', 2: 'sitting', 3: 'standing', 4: 'slow_walking',
            5: 'moderate_walking', 6: 'brisk_walking', 7: 'ascending_stairs',
            8: 'descending_stairs', 9: 'cycling', 10: 'running', 11: 'jumping', 12: 'rowing'}
G = 9.80665
ACC, GYR, QUAT = slice(0, 3), slice(3, 6), slice(6, 10)
BATCHES = {'P01-P24': range(1, 25), 'P25': range(25, 26), 'P26-P30': range(26, 31)}


def batch_of(subject):
    return next(k for k, r in BATCHES.items() if subject in r)


def available_subjects(root):
    out = []
    for p in sorted((Path(root) / 'subjects').glob('P*')):
        if (p / 'complete.json').exists():
            out.append(int(p.name[1:]))
    return out


def pure_trials(data, label, min_purity=0.95, min_frames=240):
    t = data.trials
    sel = t[(t.majority_label == label) & (t.purity >= min_purity) & (t.frames >= min_frames)]
    return [int(x) for x in sel.trial]


def label_segments(data, label, sensors, min_frames=240, max_trials=None, prepare_q=True):
    """Yield (trial, values) for contiguous runs of `label` inside official trials."""
    t = data.trials
    rows = t[t[f'label_count_{label}'] >= min_frames]
    if max_trials is not None:
        rows = rows.head(max_trials)
    for tr in rows.trial:
        values, labels = data.read_trial(int(tr), sensors, prepare_q=prepare_q)
        for s, e in label_runs(labels, label):
            if e - s >= min_frames:
                yield int(tr), values[s:e]


def label_runs(labels, label):
    """[start, stop) runs where labels == label."""
    m = np.concatenate([[False], np.asarray(labels) == label, [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# quaternions are stored x, y, z, w (Hamilton), sensor-to-world
def quat_to_rot(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_conj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def world_rate_from_quat(q, fs=FS):
    """World-frame angular velocity 2 vec(q_dot q*) by central difference (sign-continuous q)."""
    q = np.asarray(q, dtype=float)
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) * fs / 2.0
    dq[0], dq[-1] = (q[1] - q[0]) * fs, (q[-1] - q[-2]) * fs
    return 2.0 * quat_mul(dq, quat_conj(q))[..., :3]


def robust(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0, 'median': None, 'mad': None}
    med = float(np.median(v))
    return {'n': int(v.size), 'median': med, 'mad': float(np.median(np.abs(v - med)))}


def autocorr_period(x, fs=FS, min_s=0.25, max_s=4.0):
    """Dominant period (s) from the autocorrelation peak with parabolic interpolation."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    if n < 4 or np.allclose(x, 0):
        return np.nan, 0.0
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    lo, hi = int(min_s * fs), min(int(max_s * fs), n - 2)
    if hi <= lo + 1:
        return np.nan, 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    if k <= lo or k >= hi - 1:
        return np.nan, float(ac[k])
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return (k + shift) / fs, float(b)


def write_report(out_dir, name, report, markdown):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(report, indent=1, default=float),
                                      encoding='utf-8')
    (out / f'{name}.md').write_text(markdown, encoding='utf-8')
    return out / f'{name}.json'
