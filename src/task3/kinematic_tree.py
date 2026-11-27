"""17-node kinematic tree on the dataset sensors, and forward kinematics for evaluation.

Each node is one sensor/segment. Bone vectors are expressed in the sensor
frame by a static standing (N-pose) calibration: b_s = R_s,stand^T d_world * L,
where d_world is the bone direction while standing (Y up, X left, Z forward)
and L comes from body_dimensions_m.csv. The sensor-to-segment alignment is
otherwise unknown. FK positions are used only to express orientation errors
as end-point displacements (V3); they are not measured positions.
"""
import numpy as np

PARENT = {
    'LowerBack': None, 'UpperBack': 'LowerBack', 'Head': 'UpperBack',
    'RightShoulder': 'UpperBack', 'RightUpperArm': 'RightShoulder',
    'RightForeArm': 'RightUpperArm', 'RightWrist': 'RightForeArm',
    'LeftShoulder': 'UpperBack', 'LeftUpperArm': 'LeftShoulder',
    'LeftForeArm': 'LeftUpperArm', 'LeftWrist': 'LeftForeArm',
    'RightThigh': 'LowerBack', 'RightShank': 'RightThigh', 'RightFoot': 'RightShank',
    'LeftThigh': 'LowerBack', 'LeftShank': 'LeftThigh', 'LeftFoot': 'LeftShank',
}
CHILDREN = {s: [c for c, p in PARENT.items() if p == s] for s in PARENT}
END_EFFECTORS = ('Head', 'RightWrist', 'LeftWrist', 'RightFoot', 'LeftFoot')

# standing bone direction (world) and body-dimension column(s) giving its length
BONES = {
    'LowerBack': ((0, 1, 0), ('torso_length_m',), 0.5),
    'UpperBack': ((0, 1, 0), ('torso_length_m',), 0.5),
    'Head': ((0, 1, 0), ('neck_length_m', 'head_height_m'), 1.0),
    'RightShoulder': ((-1, 0, 0), ('shoulder_width_m',), 0.5),
    'LeftShoulder': ((1, 0, 0), ('shoulder_width_m',), 0.5),
    'RightUpperArm': ((0, -1, 0), ('upper_arm_length_m',), 1.0),
    'LeftUpperArm': ((0, -1, 0), ('upper_arm_length_m',), 1.0),
    'RightForeArm': ((0, -1, 0), ('forearm_length_m',), 1.0),
    'LeftForeArm': ((0, -1, 0), ('forearm_length_m',), 1.0),
    'RightWrist': ((0, -1, 0), None, 0.08),
    'LeftWrist': ((0, -1, 0), None, 0.08),
    'RightThigh': ((0, -1, 0), ('thigh_length_m',), 1.0),
    'LeftThigh': ((0, -1, 0), ('thigh_length_m',), 1.0),
    'RightShank': ((0, -1, 0), ('shank_length_m',), 1.0),
    'LeftShank': ((0, -1, 0), ('shank_length_m',), 1.0),
    'RightFoot': ((0, 0, 1), ('foot_length_m',), 0.7),
    'LeftFoot': ((0, 0, 1), ('foot_length_m',), 0.7),
}
HIP_SIDE = {'RightThigh': -1.0, 'LeftThigh': 1.0}     # along world X (X points left)


def tree_order(nodes=None):
    """Parents before children."""
    order, todo = [], list(PARENT if nodes is None else nodes)
    while todo:
        for n in list(todo):
            if PARENT[n] is None or PARENT[n] not in todo:
                order.append(n)
                todo.remove(n)
    return order


def bone_lengths(dims):
    """dims: mapping column -> metres (one participant)."""
    out = {}
    for s, (_, cols, scale) in BONES.items():
        out[s] = scale if cols is None else scale * sum(float(dims[c]) for c in cols)
    out['hip_half_width'] = 0.5 * float(dims['bi_iliac_breadth_m'])
    return out


def calibrate_bones(stand_rot, lengths):
    """Bone and hip-offset vectors in each sensor frame from mean standing orientations."""
    out = {}
    for s, (d, _, _) in BONES.items():
        r = stand_rot.get(s, np.eye(3))
        out[s] = r.T @ (np.asarray(d, float) * lengths[s])
    r_lb = stand_rot.get('LowerBack', np.eye(3))
    for s, side in HIP_SIDE.items():
        out[f'hip:{s}'] = r_lb.T @ np.array([side * lengths['hip_half_width'], 0.0, 0.0])
    return out


def forward_kinematics(rots, bones):
    """Distal end-point of every bone, (T, 3) per segment; root at the pelvis (LowerBack).

    rots: sensor -> (T, 3, 3) world rotations (all 17 needed).
    """
    t = next(iter(rots.values())).shape[0]
    start, end = {}, {}
    for s in tree_order():
        p = PARENT[s]
        if p is None:
            start[s] = np.zeros((t, 3))
        elif s in HIP_SIDE:
            start[s] = start[p] + np.einsum('nij,j->ni', rots[p], bones[f'hip:{s}'])
        else:
            start[s] = end[p]
        end[s] = start[s] + np.einsum('nij,j->ni', rots[s], bones[s])
    return end
