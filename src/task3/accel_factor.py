"""Local robust link-acceleration consistency factor for a sandwiched withheld segment (T3-M3).

Only used for a withheld segment w whose parent p and child c are both observed.
With bone vectors b (proximal -> distal joint, sensor frame) from the standing
calibration and sensors assumed at mid-segment, the joint accelerations are

  a(J_pw) = a_p + G_p R_p (+b_p / 2)        (parent sensor -> its distal joint)
  a(J_wc) = a_c + G_c R_c (-b_c / 2)        (child sensor -> its proximal joint)
  G(w, alpha) = [alpha]_x + [w]_x^2

and rigid-body kinematics of w gives a(J_wc) - a(J_pw) = G_w R_w b_w. The residual is

  e_a = da_obs - G_w R_w b_w,   da_obs = a(J_wc) - a(J_pw)

in m/s^2. Accelerations are the dataset's gravity-compensated free linear
acceleration in g (P0 verdict), converted with ACCEL_G_TO_SI; gravity is never
removed again. Parent/child rates and angular accelerations come from their
WNOJ-smoothed states, not from raw 60 Hz gyro differences.

Robustness: frames whose |da_obs| or parent/child |alpha| exceed impact
thresholds are masked out; the remaining frames use IRLS Huber weights on
the whitened residual norm.

Perturbations of w: R <- Exp(d) R (left), w <- w + d, alpha <- alpha + d.
"""
from dataclasses import dataclass

import numpy as np

from .so3_lie import hat

ACCEL_G_TO_SI = 9.80665
ACCEL_SCHEMA = {
    'ACCEL_G_TO_SI': ACCEL_G_TO_SI,
    'accel_semantics': 'gravity_compensated_linear_acceleration',
    'accel_raw_unit': 'g',
    'accel_si_unit': 'm/s^2',
    'gravity_removed': True,
}
IMPACT_ACC = 30.0        # m/s^2: |da_obs| above this is treated as an impact frame
IMPACT_ALPHA = 150.0     # rad/s^2: parent/child angular acceleration above this is masked


def gamma(w, alpha):
    """G = [alpha]_x + [w]_x^2 for batches (T, 3)."""
    wx = hat(w)
    return hat(alpha) + wx @ wx


@dataclass
class AccelFactor:
    target: np.ndarray       # (T, 3) da_obs in m/s^2
    bone: np.ndarray         # (3,) bone vector of w in its sensor frame (m)
    sigma: float             # m/s^2
    huber_k: float = 1.345
    mask: np.ndarray = None  # (T,) frames used


def residual(r, w, alpha, f):
    v = np.einsum('nij,j->ni', r, f.bone)
    pred = np.einsum('nij,nj->ni', gamma(w, alpha), v)
    return f.target - pred


def jacobian(r, w, alpha, f):
    """d e_a / d(phi, w, alpha): (T, 3, 9). e_a = target - G R b."""
    v = np.einsum('nij,j->ni', r, f.bone)
    vx = hat(v)
    wx = hat(w)
    g = hat(alpha) + wx @ wx
    jac = np.zeros((len(r), 3, 9))
    jac[:, :, 0:3] = g @ vx                                   # -(-G [v]x)
    jac[:, :, 3:6] = hat(np.cross(w, v)) + wx @ vx           # -(-[w x v]x - [w]x[v]x)
    jac[:, :, 6:9] = vx                                       # -(-[v]x)
    return jac


def irls_weights(e, f):
    """Huber IRLS weights on the whitened residual norm (masked frames get 0)."""
    s = np.linalg.norm(e, axis=1) / f.sigma
    wts = np.where(s <= f.huber_k, 1.0, f.huber_k / np.maximum(s, 1e-12))
    if f.mask is not None:
        wts = np.where(f.mask, wts, 0.0)
    return np.where(np.isfinite(s), wts, 0.0)


def link_target(acc_p, acc_c, rp, wp, ap, rc, wc, ac, bone_p, bone_c):
    """da_obs (T, 3) in m/s^2 and the impact mask. acc_* are raw (g) world-frame."""
    ja = (acc_p * ACCEL_G_TO_SI + np.einsum('nij,nj->ni', gamma(wp, ap),
                                            np.einsum('nij,j->ni', rp, 0.5 * bone_p)))
    jb = (acc_c * ACCEL_G_TO_SI + np.einsum('nij,nj->ni', gamma(wc, ac),
                                            np.einsum('nij,j->ni', rc, -0.5 * bone_c)))
    da = jb - ja
    ok = (np.all(np.isfinite(da), axis=1)
          & (np.linalg.norm(da, axis=1) < IMPACT_ACC)
          & (np.linalg.norm(ap, axis=1) < IMPACT_ALPHA)
          & (np.linalg.norm(ac, axis=1) < IMPACT_ALPHA))
    return np.where(ok[:, None], da, 0.0), ok
