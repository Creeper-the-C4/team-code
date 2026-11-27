"""Minimal SO(3) utilities (rotation vectors, right Jacobian)."""
import numpy as np

_EPS = 1e-12


def hat(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def exp(v):
    v = np.asarray(v, dtype=float)
    t = np.linalg.norm(v)
    k = hat(v)
    if t < 1e-8:
        return np.eye(3) + k + 0.5 * k @ k
    return np.eye(3) + np.sin(t) / t * k + (1.0 - np.cos(t)) / t**2 * k @ k


def log(r):
    c = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    t = np.arccos(c)
    if t < 1e-8:
        return vee(r - r.T) / 2.0
    if np.pi - t < 1e-6:
        raise ValueError('log map near pi is not needed for spine-sized rotations')
    return t / (2.0 * np.sin(t)) * vee(r - r.T)


def right_jacobian(v):
    """J_r(v) with Exp(v + d) ~= Exp(v) Exp(J_r(v) d) for small d."""
    v = np.asarray(v, dtype=float)
    t = np.linalg.norm(v)
    k = hat(v)
    if t < 1e-6:
        return np.eye(3) - 0.5 * k + k @ k / 6.0
    return (np.eye(3) - (1.0 - np.cos(t)) / t**2 * k
            + (t - np.sin(t)) / t**3 * k @ k)
