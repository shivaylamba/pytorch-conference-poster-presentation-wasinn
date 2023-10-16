import torch
from torch import nn


def ray_aabb_intersection(rays_o, rays_d, scale):
    xyz_max = torch.tensor([scale, scale, scale], device=rays_o.device)
    xyz_min = -xyz_max
    half_size = (xyz_max - xyz_min) / 2
    center = torch.tensor([0.0, 0.0, 0.0], device=rays_o.device)

    inv_d = 1.0 / rays_d
    t_min = (center - half_size - rays_o) * inv_d
    t_max = (center + half_size - rays_o) * inv_d

    _t1 = torch.min(t_min, t_max)
    _t2 = torch.max(t_min, t_max)
    t1 = _t1.max(dim=1).values
    t2 = _t2.min(dim=1).values

    hits_t = torch.empty(rays_o.size(0), 2, device=rays_o.device, dtype=rays_o.dtype)

    hits_t[:, 0] = torch.max(t1, torch.tensor([NEAR_DISTANCE], device=rays_o.device))
    hits_t[:, 1] = t2
    hits_t[t2 <= 0.0] = torch.tensor([-1.0, -1.0], device=rays_o.device)

    return hits_t
