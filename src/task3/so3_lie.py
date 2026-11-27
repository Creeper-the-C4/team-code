"""Vectorised SO(3) operators: Exp/Log, left/right Jacobians and inverses, quaternions.

All functions accept a leading batch shape. Small-angle branches use Taylor
expansions, so no 0/0 appears at theta -> 0. Log handles angles up to pi.
Quaternions follow the dataset order x, y, z, w (Hamilton, sensor-to-world).
"""
import numpy as np

SMALL = 1e-4


def hat(v):
    v = np.asarray(v, dtype=float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def vee(m):
    return np.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], axis=-1)


def _theta(v):
    return np.linalg.norm(v, axis=-1)[..., None, None]


def exp_so3(phi):
    phi = np.asarray(phi, dtype=float)
    t = _theta(phi)
    k = hat(phi)
    k2 = k @ k
    small = t < SMALL
    ts = np.where(small, 1.0, t)
    a = np.where(small, 1 - t**2 / 6, np.sin(ts) / ts)
    b = np.where(small, 0.5 - t**2 / 24, (1 - np.cos(ts)) / ts**2)
    return np.eye(3) + a * k + b * k2


def log_so3(r):
    r = np.asarray(r, dtype=float)
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    t = np.arccos(c)
    skew = vee(r - np.swapaxes(r, -1, -2)) / 2               # sin(t) * axis
    small = t < SMALL
    near_pi = t > np.pi - 1e-3
    ts = np.where(small | near_pi, 1.0, t)
    out = np.where(small[..., None], skew * (1 + t[..., None] ** 2 / 6),
                   skew * (ts / np.sin(ts))[..., None])
    if np.any(near_pi):
        idx = np.nonzero(near_pi)
        for i in zip(*idx):
            rr = r[i]
            ci = np.cos(t[i])
            # symmetric part = cos(t) I + (1 - cos t) a a^T, exact for any angle
            b = ((rr + rr.T) / 2 - ci * np.eye(3)) / (1 - ci)
            j = int(np.argmax(np.diag(b)))
            axis = b[:, j] / np.sqrt(max(b[j, j], 1e-300))
            if axis @ vee(rr - rr.T) < 0:
                axis = -axis
            out[i] = axis * t[i]
    return out


def jr(phi):
    """Right Jacobian: Exp(phi + d) ~= Exp(phi) Exp(jr(phi) d)."""
    phi = np.asarray(phi, dtype=float)
    t = _theta(phi)
    k = hat(phi)
    small = t < SMALL
    ts = np.where(small, 1.0, t)
    a = np.where(small, 0.5 - t**2 / 24, (1 - np.cos(ts)) / ts**2)
    b = np.where(small, 1.0 / 6 - t**2 / 120, (ts - np.sin(ts)) / ts**3)
    return np.eye(3) - a * k + b * (k @ k)


def jr_inv(phi):
    """Inverse right Jacobian: Log(Exp(phi) Exp(d)) ~= phi + jr_inv(phi) d."""
    phi = np.asarray(phi, dtype=float)
    t = _theta(phi)
    k = hat(phi)
    small = t < SMALL
    ts = np.where(small, 1.0, t)
    c = np.where(small, 1.0 / 12 + t**2 / 720,
                 1 / ts**2 - (1 + np.cos(ts)) / (2 * ts * np.sin(ts)))
    return np.eye(3) + 0.5 * k + c * (k @ k)


def jl(phi):
    return jr(-np.asarray(phi, dtype=float))


def jl_inv(phi):
    """Inverse left Jacobian: Log(Exp(d) Exp(phi)) ~= phi + jl_inv(phi) d."""
    return jr_inv(-np.asarray(phi, dtype=float))


def quat_xyzw_to_rot(q):
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


def rot_to_quat_xyzw(r):
    """Shepperd's method, w >= 0."""
    r = np.asarray(r, dtype=float)
    tr = np.trace(r, axis1=-2, axis2=-1)
    q = np.empty(r.shape[:-2] + (4,))
    d = np.stack([r[..., 0, 0], r[..., 1, 1], r[..., 2, 2], tr], axis=-1)
    k = np.argmax(d, axis=-1)
    for case in range(4):
        m = k == case
        if not np.any(m):
            continue
        rr = r[m]
        if case == 3:
            s = np.sqrt(1 + tr[m]) * 2
            q[m] = np.stack([(rr[:, 2, 1] - rr[:, 1, 2]) / s, (rr[:, 0, 2] - rr[:, 2, 0]) / s,
                             (rr[:, 1, 0] - rr[:, 0, 1]) / s, s / 4], axis=-1)
        else:
            i, j, l = case, (case + 1) % 3, (case + 2) % 3
            s = np.sqrt(1 + rr[:, i, i] - rr[:, j, j] - rr[:, l, l]) * 2
            v = np.empty((rr.shape[0], 4))
            v[:, i] = s / 4
            v[:, j] = (rr[:, j, i] + rr[:, i, j]) / s
            v[:, l] = (rr[:, l, i] + rr[:, i, l]) / s
            v[:, 3] = (rr[:, l, j] - rr[:, j, l]) / s
            q[m] = v
    q *= np.where(q[..., 3:4] < 0, -1.0, 1.0)
    return q


def hemisphere_continuity(q):
    """Flip signs along axis 0 so consecutive quaternions have a non-negative dot product."""
    q = np.array(q, dtype=float, copy=True)
    if len(q) > 1:
        dot = np.sum(q[1:] * q[:-1], axis=-1)
        sign = np.cumprod(np.where(dot < 0, -1.0, 1.0), axis=0)
        q[1:] *= sign[..., None]
    return q


def geodesic_deg(ra, rb):
    """Rotation angle of ra^T rb in degrees."""
    m = np.swapaxes(ra, -1, -2) @ rb
    c = np.clip((np.trace(m, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def chordal_mean(rs):
    """Projection of the arithmetic mean of rotation matrices onto SO(3)."""
    u, _, vt = np.linalg.svd(np.mean(rs, axis=0))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt
