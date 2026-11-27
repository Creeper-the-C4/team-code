"""Sensor configurations C0..C2b and their tangent-space observation Jacobians.

Per-frame state x = [phi_P (3), theta_0..theta_{n-1} (3n), p_P (3)]:
pelvis world-orientation right perturbation, joint rotation vectors, pelvis
(L5/S1) world position. phi_P and p_P are nuisance parameters.

Measurement models
- IMU orientation on segment i: e_R = Log(R_meas^T R_pred,i(x))^vee with
  R_meas = R_pred,i(x_bar); its Jacobian is the segment orientation Jacobian
  G_i (analytic, from the SO(3) right Jacobian). Pelvis orientation uses the
  right retraction R Exp(d); joints use an additive rotation-vector chart.
- Single-channel flexible bend sensor over a span: y = sum of the flexion (FE)
  components of the joints inside the span (integrated sagittal curvature;
  exact for planar bending, first order otherwise).
- IMU free acceleration: second time difference of the mount-point world
  position, kept dimensionless (lengths in spine-span units, time in
  repetition units) until the dataset acceleration unit is resolved.

Configurations (sensors added to the existing LowerBack(L5) + UpperBack pair):
C0 none, C1 +L1 IMU, C1p +T12 IMU, C1pp +sacrum IMU, C2a +one bend channel
L5->L1, C2b +one bend channel per joint between L5 and L1. C3 (plantar
pressure) is a kinetic extension and is not part of this kinematic analysis.
"""
from dataclasses import dataclass
import numpy as np

from . import so3
from .spine_kinematics import (FE, LUMBAR_LEVELS, mount_point, segment_origins,
                               segment_rotations)

CONFIGS = {
    'C0': [],
    'C1': [('imu', 'L1')],
    'C1p': [('imu', 'T12')],
    'C1pp': [('imu', 'SACRUM')],
    'C2a': [('bend', ('L5', 'L1'), 'single')],
    'C2b': [('bend', ('L5', 'L1'), 'per_joint')],
}
CONFIG_LABELS = {
    'C0': "C0  L5 + upper back (existing)",
    'C1': "C1  + L1 IMU",
    'C1p': "C1'  + T12 IMU",
    'C1pp': "C1''  + sacrum IMU",
    'C2a': "C2a  + bend L5-L1 (1 ch)",
    'C2b': "C2b  + bend L5-L1 (per joint)",
}
BASE_IMUS = ('L5', 'UB')
SPAN_LEVELS = {('L5', 'L1'): set(LUMBAR_LEVELS)}


@dataclass
class Sensor:
    kind: str            # 'imu' or 'bend'
    segment: int = -1    # imu: segment index
    joints: tuple = ()   # bend: joint indices whose FE components are summed
    label: str = ''


class NotRepresentable(Exception):
    """The configuration cannot be expressed in this spine model."""


def config_sensors(model, config):
    sensors = [Sensor('imu', model.segment_of(v), label=v) for v in BASE_IMUS]
    for spec in CONFIGS[config]:
        if spec[0] == 'imu':
            sensors.append(Sensor('imu', model.segment_of(spec[1]), label=spec[1]))
            continue
        _, (lo, hi), mode = spec
        a, b = model.segment_of(lo), model.segment_of(hi)
        joints = tuple(range(a, b))
        covered = set().union(*(model.joint_levels[j] for j in joints)) if joints else set()
        if not joints or covered != SPAN_LEVELS[(lo, hi)]:
            raise NotRepresentable(f'{config}: span {lo}-{hi} does not align with '
                                   f'{model.name} joints')
        if mode == 'single':
            sensors.append(Sensor('bend', joints=joints, label=f'bend {lo}-{hi}'))
        else:
            sensors.extend(Sensor('bend', joints=(j,), label=f'bend J{j}') for j in joints)
    return sensors


def state_dim(model):
    return 3 + 3 * model.n_joints + 3


def orientation_jacobians(model, theta):
    """G_i (3 x state_dim) for every segment: Rbar_i^T R_i(x+dx) ~= Exp(G_i dx)."""
    n = model.n_joints
    exps = [so3.exp(theta[j]) for j in range(n)]
    jr = [so3.right_jacobian(theta[j]) for j in range(n)]
    out = []
    for i in range(len(model.segments)):
        g = np.zeros((3, state_dim(model)))
        tail = np.eye(3)                     # product of Exp(theta_l) for j < l < i
        for j in range(i - 1, -1, -1):
            g[:, 3 + 3 * j:6 + 3 * j] = tail.T @ jr[j]
            tail = exps[j] @ tail
        g[:, 0:3] = tail.T                   # tail is now prod_{l<i} Exp(theta_l)
        out.append(g)
    return out


def numeric_orientation_jacobian(model, phi_p, theta, segment, h):
    """float64 central difference of e_R for one segment (pelvis + joint columns).

    Pelvis columns use the SO(3) right retraction R_P Exp(+-h e_k); joint
    columns perturb the rotation-vector chart theta +- h e_k.
    """
    base = segment_rotations(model, phi_p, theta)[segment]
    n = 3 + 3 * model.n_joints
    jac = np.zeros((3, n))
    for k in range(n):
        res = []
        for sgn in (1.0, -1.0):
            if k < 3:
                d = np.zeros(3)
                d[k] = sgn * h
                rots = [so3.exp(phi_p) @ so3.exp(d)]
                for j in range(model.n_joints):
                    rots.append(rots[-1] @ so3.exp(theta[j]))
                r = rots[segment]
            else:
                th = np.array(theta, dtype=np.float64, copy=True)
                th.flat[k - 3] += sgn * h
                r = segment_rotations(model, phi_p, th)[segment]
            res.append(so3.log(base.T @ r))
        jac[:, k] = (res[0] - res[1]) / (2 * h)
    return jac


def jacobian_relative_errors(model, phi_p, theta, steps=(1e-5, 1e-6, 1e-7)):
    """max over segments of ||J_num - J_ana||_F / ||J_ana||_F for each step size."""
    gs = orientation_jacobians(model, theta)
    out = {}
    for h in steps:
        errs = []
        for seg in range(len(model.segments)):
            ana = gs[seg][:, :-3]
            num = numeric_orientation_jacobian(model, phi_p, theta, seg, h)
            errs.append(np.linalg.norm(num - ana) / np.linalg.norm(ana))
        out[h] = float(max(errs))
    return out


def position_jacobian(model, rots, gs, segment):
    """Jacobian of the sensor mount-point world position w.r.t. the frame state."""
    up = np.array([0.0, 1.0, 0.0])
    jac = np.zeros((3, state_dim(model)))
    for l in range(segment):
        jac -= rots[l] @ so3.hat(model.heights[l] * up) @ gs[l]
    jac -= rots[segment] @ so3.hat(mount_point(model, segment)) @ gs[segment]
    jac[:, -3:] += np.eye(3)
    return jac


def sensor_position(model, phi_p, theta, p_p, segment):
    rots = segment_rotations(model, phi_p, theta)
    origins = segment_origins(model, rots, p_p)
    return origins[segment] + rots[segment] @ mount_point(model, segment)


def frame_rows(model, sensors, phi_p, theta, sigma_ori, sigma_bend):
    """Whitened orientation/bend rows (rows x state_dim) for one frame."""
    gs = orientation_jacobians(model, theta)
    rows = []
    for s in sensors:
        if s.kind == 'imu':
            rows.append(gs[s.segment] / sigma_ori)
        else:
            r = np.zeros((1, state_dim(model)))
            for j in s.joints:
                r[0, 3 + 3 * j + FE] = 1.0
            rows.append(r / sigma_bend)
    return np.vstack(rows)


def frame_position_jacobians(model, sensors, phi_p, theta):
    rots = segment_rotations(model, phi_p, theta)
    gs = orientation_jacobians(model, theta)
    return [position_jacobian(model, rots, gs, s.segment) for s in sensors if s.kind == 'imu']
