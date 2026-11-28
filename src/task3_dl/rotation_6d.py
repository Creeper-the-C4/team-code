"""Column-based 6D rotations from Zhou et al., CVPR 2019 (arXiv:1812.07035)."""

import torch
from torch.nn import functional as F


def matrix_to_rot6d(matrix):
    """Concatenate the first and second columns, in that order."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rot6d_to_matrix(columns):
    """Construct a right-handed orthonormal frame, including for degenerate outputs."""
    if columns.dtype in (torch.float16, torch.bfloat16):
        columns = columns.float()
    first, second = columns.split(3, dim=-1)
    eps = 8 * torch.finfo(columns.dtype).eps
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    x_axis = first.new_tensor([1., 0., 0.])
    u = torch.where(first_norm > eps, first / first_norm.clamp_min(eps), x_axis)
    # Cross products implement Gram-Schmidt without subtracting nearly equal vectors.
    normal = torch.linalg.cross(u, F.normalize(second, dim=-1, eps=eps))
    fallback = F.one_hot(u.abs().argmin(dim=-1), 3).to(u.dtype)
    length = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    normal = torch.where(length > eps, normal, torch.linalg.cross(u, fallback))
    w = F.normalize(normal, dim=-1, eps=eps)
    v = F.normalize(torch.linalg.cross(w, u), dim=-1, eps=eps)
    w = torch.linalg.cross(u, v)
    return torch.stack((u, v, w), dim=-1)


def geodesic_loss(predicted, target, eps=1e-6, reduction="mean"):
    """Geodesic angle in radians. Clamp only the training loss for bounded gradients."""
    cosine = ((predicted * target).sum(dim=(-1, -2)) - 1) * 0.5
    angles = cosine.clamp(-1 + eps, 1 - eps).acos()
    if reduction == "none":
        return angles
    if reduction == "mean":
        return angles.mean()
    raise ValueError("Expected reduction 'none' or 'mean'")
