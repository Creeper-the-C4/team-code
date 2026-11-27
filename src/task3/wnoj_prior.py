"""White-noise-on-jerk (WNOJ, 3rd-order integrated Wiener process) prior on SO(3).

Per segment the state at frame k is (R_k, w_k, a_k): world-frame rotation,
world-frame angular velocity (the dataset gyro is world-frame rad/s) and
angular acceleration. Between frames k and k+1 the local variable is
xi(t) = Log(R(t) R_k^T), with xi''' = white noise of power q (rad^2/s^5),
following the local-GP construction of Barfoot, Tong & Sarkka (2014),
Anderson & Barfoot (2015) and Tang, Yoon & Barfoot (2019).

Local state gamma = (xi, xi_dot, xi_ddot):
  gamma_k(t_k)     = (0, w_k, a_k)
  gamma_k(t_{k+1}) ~= (xi, w_{k+1} - xi x w_{k+1} / 2, a_{k+1} - xi x a_{k+1} / 2)
with xi = Log(R_{k+1} R_k^T). For R(t) = Exp(xi(t)) R_k and world angular
velocity dot(R) R^T = [w]_x, we have w = J_l(xi) xi_dot, hence
xi_dot = J_l^{-1}(xi) w where J_l^{-1}(xi) = J_r^{-1}(-xi) = I - hat(xi)/2 + O(||xi||^2),
so xi_dot ~= w - xi x w / 2 (and likewise for acceleration, dropping the
quadratic Coriolis-like term). This is a first-order local approximation,
exact as the frame interval goes to zero (60 Hz here). The analytic Jacobians
below are exact derivatives of this residual and are checked against both
finite differences and direct ODE integration of dot(R) = [w(t)]_x R.

Residual e_k = gamma_k(t_{k+1}) - Phi(dt) gamma_k(t_k), whitened by
L^{-1} with Q(dt) = L L^T.

Perturbations: R <- Exp(d) R (left/world), w <- w + d, a <- a + d.
"""
import numpy as np

from .so3_lie import hat, jl_inv, jr_inv, log_so3


def phi_matrix(dt):
    i = np.eye(3)
    return np.block([[i, dt * i, 0.5 * dt**2 * i], [0 * i, i, dt * i], [0 * i, 0 * i, i]])


def q_matrix(dt, qc):
    m = np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                  [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                  [dt**3 / 6, dt**2 / 2, dt]])
    return np.kron(m, qc * np.eye(3))


def whitener(dt, qc):
    """W = L^{-1} with Q = L L^T, so W e has identity covariance."""
    return np.linalg.inv(np.linalg.cholesky(q_matrix(dt, qc)))


def residual(r0, w0, a0, r1, w1, a1, dt):
    """Unwhitened residual e (K, 9) for batches of consecutive frames."""
    xi = log_so3(r1 @ np.swapaxes(r0, -1, -2))
    e1 = xi - dt * w0 - 0.5 * dt**2 * a0
    e2 = (w1 - 0.5 * np.cross(xi, w1)) - w0 - dt * a0
    e3 = (a1 - 0.5 * np.cross(xi, a1)) - a0
    return np.concatenate([e1, e2, e3], axis=-1), xi


def residual_jacobians(r0, w0, a0, r1, w1, a1, dt):
    """e (K, 9), J0 (K, 9, 9) w.r.t. frame k, J1 (K, 9, 9) w.r.t. frame k+1.

    Column order per frame: (d_phi, d_w, d_a).
    """
    e, xi = residual(r0, w0, a0, r1, w1, a1, dt)
    k = e.shape[0]
    eye = np.broadcast_to(np.eye(3), (k, 3, 3))
    dxi_d1 = jl_inv(xi)                       # Log(Exp(d) X) = xi + Jl^-1 d
    dxi_d0 = -jr_inv(xi)                      # Log(X Exp(-d)) = xi - Jr^-1 d
    de2_dxi = 0.5 * hat(w1)                   # -xi x w1 / 2 = (w1 x xi) / 2
    de3_dxi = 0.5 * hat(a1)
    half_xi = 0.5 * hat(xi)
    j0 = np.zeros((k, 9, 9))
    j1 = np.zeros((k, 9, 9))
    j0[:, 0:3, 0:3] = dxi_d0
    j0[:, 0:3, 3:6] = -dt * eye
    j0[:, 0:3, 6:9] = -0.5 * dt**2 * eye
    j0[:, 3:6, 0:3] = de2_dxi @ dxi_d0
    j0[:, 3:6, 3:6] = -eye
    j0[:, 3:6, 6:9] = -dt * eye
    j0[:, 6:9, 0:3] = de3_dxi @ dxi_d0
    j0[:, 6:9, 6:9] = -eye
    j1[:, 0:3, 0:3] = dxi_d1
    j1[:, 3:6, 0:3] = de2_dxi @ dxi_d1
    j1[:, 3:6, 3:6] = eye - half_xi
    j1[:, 6:9, 0:3] = de3_dxi @ dxi_d1
    j1[:, 6:9, 6:9] = eye - half_xi
    return e, j0, j1


def orientation_residual(r_target, r):
    """Frozen convention e_R = Log(R_target^T R)^vee and its left-perturbation Jacobian."""
    e = log_so3(np.swapaxes(r_target, -1, -2) @ r)
    # Log(Y Exp(v)) = e + Jr^-1(e) v with Exp(d) R = R Exp(R^T d), so v = R^T d
    jac = jr_inv(e) @ np.swapaxes(r, -1, -2)
    return e, jac
