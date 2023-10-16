import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--")
    parser.add_argument("--res_w", type=int, default=300)
    parser.add_argument("--res_h", type=int, default=600)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--aot", action="store_true", default=False)
    return parser.parse_args()


args = parse_arguments()

block_dim = 128

sigma_sm_preload = int((16 * 16 + 16 * 16) / block_dim)
rgb_sm_preload = int((16 * 32 + 16 * 16) / block_dim)
data_type = torch.float32
np_type = np.float32

# Replace ti.types with torch types
tf_vec3 = torch.Tensor
tf_vec8 = torch.Tensor
tf_vec16 = torch.Tensor
tf_vec32 = torch.Tensor
tf_vec1 = torch.Tensor
tf_vec2 = torch.Tensor
tf_mat1x3 = torch.Tensor
tf_index_temp = torch.IntTensor

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01

SQRT3 = 1.7320508075688772
SQRT3_MAX_SAMPLES = SQRT3 / 1024
SQRT3_2 = 1.7320508075688772 * 2

res_w = args.res_w
res_h = args.res_h
scale = 0.5
cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
grid_size = 128
base_res = 32
log2_T = 21
res = [res_w, res_h]
level = 4
exp_step_factor = 0
NGP_res = res
NGP_N_rays = res[0] * res[1]
NGP_grid_size = grid_size
NGP_exp_step_factor = exp_step_factor
NGP_scale = scale

NGP_center = tf_vec3([0.0, 0.0, 0.0])
NGP_xyz_min = -tf_vec3([scale, scale, scale])
NGP_xyz_max = tf_vec3([scale, scale, scale])
NGP_half_size = (NGP_xyz_max - NGP_xyz_min) / 2

# hash table variables
NGP_min_samples = 1 if exp_step_factor == 0 else 4
NGP_per_level_scales = 1.3195079565048218
NGP_base_res = base_res
NGP_level = level
NGP_offsets = [0 for _ in range(16)]


#################################
# Initialization & Util Kernels #
#################################
# <----------------- hash table util code ----------------->
def calc_dt(t, exp_step_factor, grid_size, scale):
    return torch.clamp(
        t * exp_step_factor, SQRT3_MAX_SAMPLES, SQRT3_2 * scale / grid_size
    )


def __expand_bits(v):
    v = (v * torch.tensor(0x00010001, dtype=torch.uint32)) & torch.tensor(
        0xFF0000FF, dtype=torch.uint32
    )
    v = (v * torch.tensor(0x00000101, dtype=torch.uint32)) & torch.tensor(
        0x0F00F00F, dtype=torch.uint32
    )
    v = (v * torch.tensor(0x00000011, dtype=torch.uint32)) & torch.tensor(
        0xC30C30C3, dtype=torch.uint32
    )
    v = (v * torch.tensor(0x00000005, dtype=torch.uint32)) & torch.tensor(
        0x49249249, dtype=torch.uint32
    )
    return v


def __morton3D(xyz):
    xyz = __expand_bits(xyz)
    return xyz[0] | (xyz[1] << 1) | (xyz[2] << 2)


def fast_hash(pos_grid_local):
    result = torch.tensor(0, dtype=torch.uint32)
    primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.uint32)
    for i in range(3):
        result ^= pos_grid_local[i].to(torch.uint32) * primes[i]
    return result


def under_hash(pos_grid_local, resolution):
    result = torch.tensor(0, dtype=torch.uint32)
    stride = torch.tensor(1, dtype=torch.uint32)
    for i in range(3):
        result += pos_grid_local[i].to(torch.uint32) * stride
        stride *= resolution
    return result


def grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size):
    hash_result = torch.tensor(0, dtype=torch.uint32)
    if indicator == 1:
        hash_result = under_hash(pos_grid_local, resolution)
    else:
        hash_result = fast_hash(pos_grid_local)

    return hash_result % map_size


# <----------------- hash table util code ----------------->


@torch.jit.script
def random_in_unit_disk():
    theta = 2.0 * np.pi * torch.rand(1)
    return torch.tensor([torch.sin(theta), torch.cos(theta)])


@torch.jit.script
def random_normal():
    x = torch.rand(1) * 2.0 - 1.0
    y = torch.rand(1) * 2.0 - 1.0
    return torch.tensor([x, y])


@torch.jit.script
def dir_encode_func(dir_):
    input_val = torch.zeros(16)
    d = (dir_ / dir_.norm() + 1) / 2
    x = d[0]
    y = d[1]
    z = d[2]
    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z

    temp = 0.28209479177387814
    input_val[0] = temp
    input_val[1] = -0.48860251190291987 * y
    input_val[2] = 0.48860251190291987 * z
    input_val[3] = -0.48860251190291987 * x
    input_val[4] = 1.0925484305920792 * xy
    input_val[5] = -1.0925484305920792 * yz
    input_val[6] = 0.94617469575755997 * z2 - 0.31539156525251999
    input_val[7] = -1.0925484305920792 * xz
    input_val[8] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2
    input_val[9] = 0.59004358992664352 * y * (-3.0 * x2 + y2)
    input_val[10] = 2.8906114426405538 * xy * z
    input_val[11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2)
    input_val[12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0)
    input_val[13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2)
    input_val[14] = 1.4453057213202769 * z * (x2 - y2)
    input_val[15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2)

    return input_val


@torch.jit.script
def rotate_scale(
    NGP_pose: Tensor, angle_x: float, angle_y: float, angle_z: float, radius: float
):
    # first move camera to radius
    res = torch.eye(4)
    res[2, 3] -= radius

    # rotate
    rot = torch.eye(4)
    rot[:3, :3] = NGP_pose[None][:3, :3]

    rot = (
        torch.matmul(
            torch.matmul(
                torch.tensor(
                    [
                        [1, 0, 0],
                        [0, torch.cos(angle_x), -torch.sin(angle_x)],
                        [0, torch.sin(angle_x), torch.cos(angle_x)],
                    ]
                ),
                torch.matmul(
                    torch.tensor(
                        [
                            [torch.cos(angle_y), 0, torch.sin(angle_y)],
                            [0, 1, 0],
                            [-torch.sin(angle_y), 0, torch.cos(angle_y)],
                        ]
                    ),
                    torch.tensor(
                        [
                            [torch.cos(angle_z), -torch.sin(angle_z), 0],
                            [torch.sin(angle_z), torch.cos(angle_z), 0],
                            [0, 0, 1],
                        ]
                    ),
                ),
            )
        )
        @ rot
    )

    res = torch.matmul(rot, res)
    # translate
    res[:3, 3] -= NGP_center

    NGP_pose[None] = res[:3, :4]


@torch.jit.script
def reset(
    counter: Tensor, NGP_alive_indices: Tensor, NGP_opacity: Tensor, NGP_rgb: Tensor
):
    for i in range(NGP_opacity.shape[0]):
        NGP_opacity[i] = 0.0
    for i in range(NGP_rgb.shape[0]):
        NGP_rgb[i] = torch.tensor([0.0, 0.0, 0.0])
    counter[0] = NGP_N_rays
    for i, j in torch.ndindex(NGP_N_rays, 2):
        NGP_alive_indices[i * 2 + j] = i


@torch.jit.script
def init_current_index(NGP_current_index: Tensor):
    NGP_current_index[None] = 0


@torch.jit.script
def fill_ndarray(arr: Tensor, val: float):
    for i in range(arr.shape[0]):
        arr[i] = torch.tensor([val, val])


@torch.jit.script
def rearange_index(
    NGP_model_launch: Tensor,
    NGP_padd_block_network: Tensor,
    NGP_temp_hit: Tensor,
    NGP_run_model_ind: Tensor,
    B: int,
):
    NGP_model_launch[None] = 0

    for i in range(B):
        if NGP_run_model_ind[i]:
            index = torch.atomic_add(NGP_model_launch[None], 1)
            NGP_temp_hit[index] = i

    NGP_model_launch[None] += 1
    NGP_padd_block_network[None] = (
        (NGP_model_launch[None] + block_dim - 1) // block_dim
    ) * block_dim


@torch.jit.script
def re_order(
    counter: Tensor, NGP_alive_indices: Tensor, NGP_current_index: Tensor, B: int
):
    counter[0] = 0
    c_index = NGP_current_index[None]
    n_index = (c_index + 1) % 2
    NGP_current_index[None] = n_index

    for i in range(B):
        alive_temp = NGP_alive_indices[i * 2 + c_index]
        if alive_temp >= 0:
            index = torch.atomic_add(counter[0], 1)
            NGP_alive_indices[index * 2 + n_index] = alive_temp


@torch.jit.script
def _ray_aabb_intersec(ray_o, ray_d):
    inv_d = 1.0 / ray_d

    t_min = (NGP_center - NGP_half_size - ray_o) * inv_d
    t_max = (NGP_center + NGP_half_size - ray_o) * inv_d

    _t1 = torch.min(t_min, t_max)
    _t2 = torch.max(t_min, t_max)
    t1 = _t1.max()
    t2 = _t2.min()

    return torch.tensor([t1, t2])


@torch.jit.script
def ray_intersect(
    counter: Tensor,
    NGP_pose: Tensor,
    NGP_directions: Tensor,
    NGP_hits_t: Tensor,
    NGP_rays_o: Tensor,
    NGP_rays_d: Tensor,
):
    for i in range(counter[0]):
        c2w = NGP_pose[None]
        mat_result = torch.matmul(NGP_directions[i], c2w[:, :3].transpose(0, 1))
        ray_d = torch.tensor([mat_result[0, 0], mat_result[0, 1], mat_result[0, 2]])
        ray_o = c2w[:, 3]

        t1t2 = _ray_aabb_intersec(ray_o, ray_d)

        if t1t2[1] > 0.0:
            NGP_hits_t[i][0] = torch.max(t1t2[0], NEAR_DISTANCE)
            NGP_hits_t[i][1] = t1t2[1]

        NGP_rays_o[i] = ray_o
        NGP_rays_d[i] = ray_d


@torch.jit.script
def raymarching_test_kernel(
    counter: Tensor,
    NGP_density_bitfield: Tensor,
    NGP_hits_t: Tensor,
    NGP_alive_indices: Tensor,
    NGP_rays_o: Tensor,
    NGP_rays_d: Tensor,
    NGP_current_index: Tensor,
    NGP_xyzs: Tensor,
    NGP_dirs: Tensor,
    NGP_deltas: Tensor,
    NGP_ts: Tensor,
    NGP_run_model_ind: Tensor,
    NGP_N_eff_samples: Tensor,
    N_samples: int,
):
    for n in range(counter[0]):
        c_index = NGP_current_index.unsqueeze(0)
        r = NGP_alive_indices[n * 2 + c_index]
        grid_size3 = NGP_grid_size**3
        grid_size_inv = 1.0 / NGP_grid_size

        ray_o = NGP_rays_o[r]
        ray_d = NGP_rays_d[r]
        t1t2 = NGP_hits_t[r]

        d_inv = 1.0 / ray_d

        t = t1t2[0]
        t2 = t1t2[1]

        s = 0

        start_idx = n * N_samples

        while (0 <= t) & (t < t2) & (s < N_samples):
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, NGP_exp_step_factor, NGP_grid_size, NGP_scale)

            mip_bound = 0.5
            mip_bound_inv = 1 / mip_bound

            nxyz = torch.clamp(
                0.5 * (xyz * mip_bound_inv + 1) * NGP_grid_size,
                0.0,
                NGP_grid_size - 1.0,
            )

            idx = __morton3D(torch.tensor(nxyz, dtype=torch.uint32))
            occ = torch.uint32(
                NGP_density_bitfield[torch.uint32(idx // 32)]
                & (torch.u32(1) << torch.u32(idx % 32))
            )

            if occ:
                sn = start_idx + s
                for p in range(3):
                    NGP_xyzs[sn][p] = xyz[p]
                    NGP_dirs[sn][p] = ray_d[p]
                NGP_run_model_ind[sn] = 1
                NGP_ts[sn] = t
                NGP_deltas[sn] = dt
                t += dt
                NGP_hits_t[r][0] = t
                s += 1

            else:
                txyz = (
                    ((nxyz + 0.5 + 0.5 * torch.sign(ray_d)) * grid_size_inv * 2 - 1)
                    * mip_bound
                    - xyz
                ) * d_inv

                t_target = t + torch.max(torch.tensor(0), txyz.min())
                t += calc_dt(t, NGP_exp_step_factor, NGP_grid_size, NGP_scale)
                while t < t_target:
                    t += calc_dt(t, NGP_exp_step_factor, NGP_grid_size, NGP_scale)

        NGP_N_eff_samples[n] = s
        if s == 0:
            NGP_alive_indices[n * 2 + c_index] = -1


@torch.jit.script
def hash_encode(
    NGP_hash_embedding: Tensor,
    NGP_model_launch: Tensor,
    NGP_xyzs: Tensor,
    NGP_dirs: Tensor,
    NGP_deltas: Tensor,
    NGP_xyzs_embedding: Tensor,
    NGP_temp_hit: Tensor,
):
    for sn in range(NGP_model_launch[0]):
        for level in range(NGP_level):
            xyz = NGP_xyzs[NGP_temp_hit[sn]] + 0.5
            offset = NGP_offsets[level] * 4

            init_val0 = torch.tensor(0.0)
            init_val1 = torch.tensor(1.0)
            local_feature_0 = init_val0[0]
            local_feature_1 = init_val0[0]
            local_feature_2 = init_val0[0]
            local_feature_3 = init_val0[0]

            scale = NGP_base_res * torch.exp(level * NGP_per_level_scales) - 1.0
            resolution = torch.ceil(scale).to(torch.uint32) + 1

            pos = xyz * scale + 0.5
            pos_grid_uint = torch.floor(pos).to(torch.uint32)
            pos -= pos_grid_uint

            for idx in range(8):
                w = init_val1[0]
                pos_grid_local = torch.zeros(3, dtype=torch.uint32)

                for d in range(3):
                    if (idx & (1 << d)) == 0:
                        pos_grid_local[d] = pos_grid_uint[d]
                        w *= data_type(1 - pos[d])
                    else:
                        pos_grid_local[d] = pos_grid_uint[d] + 1
                        w *= data_type(pos[d])

                index = 0
                stride = 1
                for c_ in range(3):
                    index += pos_grid_local[c_] * stride
                    stride *= resolution

                local_feature_0 += data_type(w * NGP_hash_embedding[offset + index * 4])
                local_feature_1 += data_type(
                    w * NGP_hash_embedding[offset + index * 4 + 1]
                )
                local_feature_2 += data_type(
                    w * NGP_hash_embedding[offset + index * 4 + 2]
                )
                local_feature_3 += data_type(
                    w * NGP_hash_embedding[offset + index * 4 + 3]
                )

            NGP_xyzs_embedding[sn, level * 4] = local_feature_0
            NGP_xyzs_embedding[sn, level * 4 + 1] = local_feature_1
            NGP_xyzs_embedding[sn, level * 4 + 2] = local_feature_2
            NGP_xyzs_embedding[sn, level * 4 + 3] = local_feature_3


@torch.jit.script
def sigma_rgb_layer(
    NGP_sigma_weights: torch.Tensor,
    NGP_rgb_weights: torch.Tensor,
    NGP_model_launch: torch.Tensor,
    NGP_padd_block_network: torch.Tensor,
    NGP_xyzs_embedding: torch.Tensor,
    NGP_dirs: torch.Tensor,
    NGP_out_1: torch.Tensor,
    NGP_out_3: torch.Tensor,
    NGP_temp_hit: torch.Tensor,
):
    block_dim = 16
    for sn in range(NGP_padd_block_network.item()):
        ray_id = NGP_temp_hit[sn].item()
        tid = sn % block_dim
        did_launch_num = NGP_model_launch.item()
        init_val = torch.zeros(1)
        sigma_weight = torch.zeros((16 * 16 + 16 * 16,), dtype=NGP_sigma_weights.dtype)
        rgb_weight = torch.zeros((16 * 32 + 16 * 16,), dtype=NGP_rgb_weights.dtype)

        for i in range(sigma_sm_preload):
            k = tid * sigma_sm_preload + i
            sigma_weight[k] = NGP_sigma_weights[k]

        for i in range(rgb_sm_preload):
            k = tid * rgb_sm_preload + i
            rgb_weight[k] = NGP_rgb_weights[k]

        if sn < did_launch_num:
            s0 = init_val[0]
            s1 = init_val[0]
            s2 = init_val[0]

            dir_ = NGP_dirs[ray_id]
            rgb_input_val = dir_encode_func(dir_)
            sigma_output_val = torch.zeros((16,), dtype=NGP_xyzs_embedding.dtype)

            for i in range(16):
                temp = init_val[0]
                for j in range(16):
                    temp += NGP_xyzs_embedding[sn, j] * sigma_weight[i * 16 + j]

                for j in range(16):
                    sigma_output_val[j] += (
                        torch.max(
                            torch.tensor([0.0], dtype=NGP_xyzs_embedding.dtype), temp
                        )
                        * sigma_weight[16 * 16 + j * 16 + i]
                    )

            for i in range(16):
                rgb_input_val[16 + i] = sigma_output_val[i]

            for i in range(16):
                temp = init_val[0]
                for j in range(32):
                    temp += rgb_input_val[j] * rgb_weight[i * 32 + j]

                s0 += (
                    torch.max(torch.tensor([0.0], dtype=NGP_xyzs_embedding.dtype), temp)
                    * rgb_weight[16 * 32 + i]
                )
                s1 += (
                    torch.max(torch.tensor([0.0], dtype=NGP_xyzs_embedding.dtype), temp)
                    * rgb_weight[16 * 32 + 16 + i]
                )
                s2 += (
                    torch.max(torch.tensor([0.0], dtype=NGP_xyzs_embedding.dtype), temp)
                    * rgb_weight[16 * 32 + 32 + i]
                )

            NGP_out_1[NGP_temp_hit[sn]] = torch.exp(sigma_output_val[0])
            NGP_out_3[NGP_temp_hit[sn], 0] = 1 / (1 + torch.exp(-s0))
            NGP_out_3[NGP_temp_hit[sn], 1] = 1 / (1 + torch.exp(-s1))
            NGP_out_3[NGP_temp_hit[sn], 2] = 1 / (1 + torch.exp(-s2))


def composite_test(
    counter: torch.Tensor,
    NGP_alive_indices: torch.Tensor,
    NGP_rgb: torch.Tensor,
    NGP_opacity: torch.Tensor,
    NGP_current_index: torch.Tensor,
    NGP_deltas: torch.Tensor,
    NGP_ts: torch.Tensor,
    NGP_out_3: torch.Tensor,
    NGP_out_1: torch.Tensor,
    NGP_N_eff_samples: torch.Tensor,
    max_samples: int,
    T_threshold: torch.Tensor,
):
    for n in range(counter.item()):
        N_samples = NGP_N_eff_samples[n]
        if N_samples != 0:
            c_index = NGP_current_index.item()
            r = NGP_alive_indices[n * 2 + c_index]

            T = 1.0 - NGP_opacity[r]

            start_idx = n * max_samples

            rgb_temp = torch.zeros(3)
            depth_temp = torch.zeros(1)
            opacity_temp = torch.zeros(1)
            out_3_temp = torch.zeros(3)

            for s in range(N_samples):
                sn = start_idx + s
                a = 1.0 - torch.exp(-NGP_out_1[sn] * NGP_deltas[sn])
                w = a * T

                for i in range(3):
                    out_3_temp[i] = NGP_out_3[sn, i]

                rgb_temp += w * out_3_temp
                depth_temp[0] += w * NGP_ts[sn]
                opacity_temp[0] += w

                T *= 1.0 - a

                if T <= T_threshold:
                    NGP_alive_indices[n * 2 + c_index] = -1
                    break

            NGP_rgb[r] = NGP_rgb[r] + rgb_temp
            NGP_opacity[r] = NGP_opacity[r] + opacity_temp[0]


@torch.jit.script
def get_rays(pose: Tensor, directions: Tensor, rays_o: Tensor, rays_d: Tensor):
    for i in range(directions.shape[0]):
        c2w = pose.unsqueeze(0)
        mat_result = torch.matmul(directions[i], c2w[:, :3].transpose(0, 1))
        ray_d = mat_result[0, :3]
        ray_o = c2w[:, 3]

        rays_o[i] = ray_o
        rays_d[i] = ray_d


########################
# Model Initialization #
########################
def initialize():
    offset = 0
    NGP_max_params = 2**log2_T
    for i in range(NGP_level):
        resolution = (
            int(np.ceil(NGP_base_res * np.exp(i * NGP_per_level_scales) - 1.0)) + 1
        )
        print(f"level: {i}, res: {resolution}")
        params_in_level = resolution**3
        params_in_level = (
            int(resolution**3)
            if params_in_level % 8 == 0
            else int((params_in_level + 8 - 1) / 8) * 8
        )
        params_in_level = min(NGP_max_params, params_in_level)
        NGP_offsets[i] = offset
        offset += params_in_level


def load_deployment_model(model_path):
    def NGP_get_direction(res_w, res_h, camera_angle_x):
        w, h = int(res_w), int(res_h)
        fx = 0.5 * w / np.tan(0.5 * camera_angle_x)
        fy = 0.5 * h / np.tan(0.5 * camera_angle_x)
        cx, cy = 0.5 * w, 0.5 * h

        x, y = np.meshgrid(
            np.arange(w, dtype=np.float32) + 0.5,
            np.arange(h, dtype=np.float32) + 0.5,
            indexing="xy",
        )

        directions = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], -1)

        return directions.reshape(-1, 3)

    if model_path is None:
        print(
            "Please specify your pretrain deployment model with --model_path={path_to_model}/deployment.npy"
        )
        assert False

    print(f"Loading model from {model_path}")
    model = np.load(model_path, allow_pickle=True).item()

    global NGP_per_level_scales
    NGP_per_level_scales = model["model.per_level_scale"]

    # Initialize directions
    camera_angle_x = 0.5
    directions = NGP_get_direction(NGP_res[0], NGP_res[1], camera_angle_x)
    directions = directions[:, None, :].astype(np_type)
    model["model.directions"] = directions

    return model
