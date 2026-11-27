"""Serial-chain spine models M1/M2/M3 from the pelvis (sacrum) to the upper back.

Frames and axes follow the dataset convention (X left, Y up, Z forward). Each
segment frame has its local Y axis along the spine, local X lateral, local Z
anterior, so a joint rotation vector theta = (FE, AR, LB) holds
flexion/extension about X, axial rotation about Y and lateral bending about Z.

Chain: segment 0 is the pelvis; joint j connects segment j to segment j+1 and
acts as R_{j+1} = R_j Exp(theta_j). The pelvis frame origin sits at the L5/S1
joint.

Prior ranges of motion are the representative intervertebral values tabulated
by White & Panjabi, *Clinical Biomechanics of the Spine*, 2nd ed. (1990)
[values to be checked against the original table before citing in the paper].
FE is the total flexion+extension range; LB and AR are one-side ranges. The
ROM-derived nominal prior scale treats FE_total as 4 sigma and the one-side
LB/AR values as 2 sigma (sigma_FE = ROM/4, sigma_one-side = ROM/2); a 0.5x /
1.0x / 1.5x scale sweep is reported. Lumped joints add the variances of the
levels they represent (independent levels).
"""
from dataclasses import dataclass, field
import numpy as np

from . import so3

# level: (FE total, LB one side, AR one side) in degrees
ROM_DEG = {
    'L5/S1': (17.0, 3.0, 1.0),
    'L4/L5': (16.0, 6.0, 2.0),
    'L3/L4': (15.0, 8.0, 2.0),
    'L2/L3': (14.0, 6.0, 2.0),
    'L1/L2': (12.0, 6.0, 2.0),
    'T12/L1': (12.0, 8.0, 2.0),
    'T11/T12': (12.0, 9.0, 2.0),
    'T10/T11': (9.0, 7.0, 2.0),
    'T9/T10': (6.0, 6.0, 4.0),
    'T8/T9': (6.0, 6.0, 6.0),
    'T7/T8': (6.0, 6.0, 7.0),
    'T6/T7': (5.0, 6.0, 7.0),
}
LUMBAR_LEVELS = ('L4/L5', 'L3/L4', 'L2/L3', 'L1/L2')
THORACIC_LEVELS = ('T11/T12', 'T10/T11', 'T9/T10', 'T8/T9', 'T7/T8', 'T6/T7')
ALL_LEVELS = ('L5/S1',) + LUMBAR_LEVELS + ('T12/L1',) + THORACIC_LEVELS

FE, AR, LB = 0, 1, 2  # components of a joint rotation vector (local X, Y, Z)


def level_prior_var(level):
    fe, lb, ar = np.deg2rad(ROM_DEG[level])
    var = np.zeros(3)
    var[FE] = (fe / 4.0) ** 2
    var[AR] = (ar / 2.0) ** 2
    var[LB] = (lb / 2.0) ** 2
    return var


@dataclass
class Segment:
    name: str
    vertebrae: tuple          # anatomical labels carried rigidly by this segment
    n_levels: int             # motion levels spanned along its length (for height)


@dataclass
class SpineModel:
    name: str
    description: str
    segments: list
    joint_levels: list        # joint_levels[j] = levels lumped into joint j
    level_height: float = 1.0 / len(ALL_LEVELS)   # equal-height levels, span = 1
    sensor_posterior_offset: float = 0.12          # skin offset, in span units
    heights: np.ndarray = field(init=False)

    def __post_init__(self):
        assert len(self.joint_levels) == len(self.segments) - 1
        self.heights = np.array([s.n_levels * self.level_height for s in self.segments])

    @property
    def n_joints(self):
        return len(self.joint_levels)

    def segment_of(self, vertebra):
        for i, s in enumerate(self.segments):
            if vertebra in s.vertebrae:
                return i
        raise KeyError(vertebra)

    def prior_cov(self):
        """Block-diagonal prior covariance of the stacked joint vector (3n)."""
        var = np.concatenate([sum(level_prior_var(l) for l in levels)
                              for levels in self.joint_levels])
        return np.diag(var)

    def joint_sigma_deg(self):
        return np.rad2deg(np.sqrt(np.diag(self.prior_cov()))).reshape(-1, 3)

    def qoi_vector(self, levels, component=FE):
        """Linear functional c with c.theta = summed component over `levels`.

        Returns None when a joint lumps levels both inside and outside the set,
        i.e. the quantity is not representable in this model.
        """
        levels = set(levels)
        c = np.zeros(3 * self.n_joints)
        for j, jl in enumerate(self.joint_levels):
            inside = levels.intersection(jl)
            if inside and len(inside) != len(jl):
                return None
            if inside:
                c[3 * j + component] = 1.0
        return c if c.any() else None


def build_m1():
    segs = [Segment('pelvis', ('S1', 'SACRUM'), 0),
            Segment('L5', ('L5',), 1),
            Segment('trunk', ('L4', 'L3', 'L2', 'L1', 'T12', 'UB'), len(ALL_LEVELS) - 1)]
    joints = [('L5/S1',), LUMBAR_LEVELS + ('T12/L1',) + THORACIC_LEVELS]
    return SpineModel('M1', 'lumped: pelvis | L5 | rigid L4..upper back', segs, joints)


def build_m2():
    segs = [Segment('pelvis', ('S1', 'SACRUM'), 0),
            Segment('L5', ('L5',), 1),
            Segment('lumbar', ('L4', 'L3', 'L2', 'L1'), len(LUMBAR_LEVELS)),
            Segment('thorax', ('T12', 'UB'), 1 + len(THORACIC_LEVELS))]
    joints = [('L5/S1',), LUMBAR_LEVELS, ('T12/L1',) + THORACIC_LEVELS]
    return SpineModel('M2', 'two-segment: lumbar block L4-L1 | rigid thorax T12..upper back',
                      segs, joints)


def build_m3():
    segs = [Segment('pelvis', ('S1', 'SACRUM'), 0),
            Segment('L5', ('L5',), 1),
            Segment('L4', ('L4',), 1),
            Segment('L3', ('L3',), 1),
            Segment('L2', ('L2',), 1),
            Segment('L1', ('L1',), 1),
            Segment('T12', ('T12',), 1),
            Segment('upper_thorax', ('UB',), len(THORACIC_LEVELS))]
    joints = [('L5/S1',), ('L4/L5',), ('L3/L4',), ('L2/L3',), ('L1/L2',), ('T12/L1',),
              THORACIC_LEVELS]
    return SpineModel('M3', 'per-level lumbar L5..L1 + T12 | lumped thoracic T11/T12..T6/T7',
                      segs, joints)


MODELS = {'M1': build_m1, 'M2': build_m2, 'M3': build_m3}


def segment_rotations(model, phi_p, theta):
    """World rotation of every segment for pelvis rotation phi_p and joints theta (n,3)."""
    rots = [so3.exp(phi_p)]
    for j in range(model.n_joints):
        rots.append(rots[-1] @ so3.exp(theta[j]))
    return rots


def segment_origins(model, rots, p_pelvis):
    """World position of each segment's proximal joint (pelvis origin = L5/S1)."""
    origins = [np.asarray(p_pelvis, dtype=float)]
    up = np.array([0.0, 1.0, 0.0])
    for i in range(1, len(model.segments)):
        origins.append(origins[-1] + rots[i - 1] @ (model.heights[i - 1] * up))
    return origins


def mount_point(model, segment):
    """Sensor mount point in segment coordinates: mid-height, posterior skin."""
    h = model.heights[segment]
    y = 0.5 * h if h > 0 else -0.5 * model.level_height   # sacrum sits below L5/S1
    return np.array([0.0, y, -model.sensor_posterior_offset])


def deadlift_pose(model, s, lumbar_flexion_deg=25.0, thoracic_flexion_deg=10.0,
                  hip_hinge_deg=50.0):
    """Synthetic deadlift posture at phase s in [0, 1] (0 upright, 1 bottom).

    Lumbar flexion (L5/S1..L1/L2) and thoracic flexion are distributed over
    levels in proportion to their FE range; the pelvis tilts forward by the
    hip hinge. Model-based input, not a measurement.
    """
    lumbar = ('L5/S1',) + LUMBAR_LEVELS
    thoracic = ('T12/L1',) + THORACIC_LEVELS

    def share(levels, total):
        fe = np.array([ROM_DEG[l][0] for l in levels])
        return dict(zip(levels, np.deg2rad(total) * fe / fe.sum()))

    per_level = {**share(lumbar, lumbar_flexion_deg), **share(thoracic, thoracic_flexion_deg)}
    theta = np.zeros((model.n_joints, 3))
    for j, levels in enumerate(model.joint_levels):
        theta[j, FE] = s * sum(per_level[l] for l in levels)
    phi_p = np.array([s * np.deg2rad(hip_hinge_deg), 0.0, 0.0])
    return phi_p, theta
