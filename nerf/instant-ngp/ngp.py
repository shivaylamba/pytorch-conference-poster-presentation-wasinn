import argparse
import os
import shutil
from typing import Tuple

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from modules.intersection import ray_aabb_intersect

import numpy as np
import torch
from matplotlib import pyplot as plt
from kernels import (
    args,
    np_type,
    data_type,
    rotate_scale,
    reset,
    ray_intersect,
    raymarching_test_kernel,
    rearange_index,
    hash_encode,
    sigma_rgb_layer,
    composite_test,
    re_order,
    fill_ndarray,
    init_current_index,
    rotate_scale,
    initialize,
    load_deployment_model,
    cascades,
    grid_size,
    scale,
    NGP_res,
    NGP_N_rays,
    NGP_min_samples,
    get_rays,
)


#########################
# Compile for AOT files #
#########################
def save_aot_weights(aot_folder, np_arr, name):
    # Binary Header: int32(dtype) int32(num_elements)
    # Binary Contents: flat binary buffer

    # dtype: 0(float32), 1(float16), 2(int32), 3(int16), 4(uint32), 5(uint16)
    if np_arr.dtype == np.float32:
        dtype = 0
    elif np_arr.dtype == np.float16:
        dtype = 1
    elif np_arr.dtype == np.int32:
        dtype = 2
    elif np_arr.dtype == np.int16:
        dtype = 3
    elif np_arr.dtype == np.uint32:
        dtype = 4
    elif np_arr.dtype == np.uint16:
        dtype = 5
    else:
        print("Unrecognized dtype: ", np_arr.dtype)
        assert False
    num_elements = np_arr.size

    byte_arr = np_arr.flatten().tobytes()
    header = np.array([dtype, num_elements]).astype("int32").tobytes()

    filename = aot_folder + "/" + name + ".bin"
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, "wb+") as f:
        f.write(header)
        f.write(byte_arr)


##########################################
# Inference on Host Machine (DEBUG ONLY) #
##########################################
def update_model_weights(model):
    NGP_hash_embedding.from_numpy(model["model.hash_encoder.params"].astype(np_type))
    NGP_sigma_weights.from_numpy(model["model.xyz_encoder.params"].astype(np_type))
    NGP_rgb_weights.from_numpy(model["model.rgb_net.params"].astype(np_type))
    NGP_density_bitfield.from_numpy(model["model.density_bitfield"].view("uint32"))

    pose = model["poses"][20].astype(np_type).reshape(3, 4)
    NGP_pose.from_numpy(pose)

    NGP_directions.from_numpy(model["model.directions"])


def run_inference(
    max_samples, T_threshold, dist_to_focus=0.8, len_dis=0.0
) -> Tuple[float, int, int]:
    samples = 0
    # rotate_scale(NGP_pose, 0.5, 0.5, 0.0, 2.5)
    reset(NGP_counter, NGP_alive_indices, NGP_opacity, NGP_rgb)

    get_rays(NGP_pose, NGP_directions, NGP_rays_o, NGP_rays_d)
    ray_aabb_intersect(NGP_hits_t, NGP_rays_o, NGP_rays_d, scale)

    while samples < max_samples:
        N_alive = NGP_counter[0]
        if N_alive == 0:
            break

        # how many more samples the number of samples add for each ray
        N_samples = max(min(NGP_N_rays // N_alive, 64), NGP_min_samples)
        samples += N_samples
        launch_model_total = N_alive * N_samples

        raymarching_test_kernel(
            NGP_counter,
            NGP_density_bitfield,
            NGP_hits_t,
            NGP_alive_indices,
            NGP_rays_o,
            NGP_rays_d,
            NGP_current_index,
            NGP_xyzs,
            NGP_dirs,
            NGP_deltas,
            NGP_ts,
            NGP_run_model_ind,
            NGP_N_eff_samples,
            N_samples,
        )
        rearange_index(
            NGP_model_launch,
            NGP_padd_block_network,
            NGP_temp_hit,
            NGP_run_model_ind,
            launch_model_total,
        )
        hash_encode(
            NGP_hash_embedding,
            NGP_model_launch,
            NGP_xyzs,
            NGP_dirs,
            NGP_deltas,
            NGP_xyzs_embedding,
            NGP_temp_hit,
        )
        sigma_rgb_layer(
            NGP_sigma_weights,
            NGP_rgb_weights,
            NGP_model_launch,
            NGP_padd_block_network,
            NGP_xyzs_embedding,
            NGP_dirs,
            NGP_out_1,
            NGP_out_3,
            NGP_temp_hit,
        )

        composite_test(
            NGP_counter,
            NGP_alive_indices,
            NGP_rgb,
            NGP_opacity,
            NGP_current_index,
            NGP_deltas,
            NGP_ts,
            NGP_out_3,
            NGP_out_1,
            NGP_N_eff_samples,
            N_samples,
            T_threshold,
        )
        re_order(NGP_counter, NGP_alive_indices, NGP_current_index, N_alive)

    return samples, N_alive, N_samples


def inference_local(n=1):
    for _ in range(n):
        samples, N_alive, N_samples = run_inference(max_samples=100, T_threshold=1e-2)

    torch.cuda.synchronize()

    # Show inferenced image
    rgb_np = NGP_rgb.to_numpy().reshape(NGP_res[1], NGP_res[0], 3)
    plt.imshow((rgb_np * 255).astype(np.uint8))
    plt.show()


if __name__ == "__main__":
    model = load_deployment_model(args.model_path)
    initialize()

    ##################################
    #     THIS IS FOR DEBUG ONLY     #
    # Run inference on local machine #
    ##################################
    # Others
    NGP_hits_t = torch.zeros((NGP_N_rays, 2), dtype=data_type)

    fill_ndarray(NGP_hits_t, -1.0)

    NGP_rays_o = torch.zeros((NGP_N_rays, 3), dtype=data_type)
    NGP_rays_d = torch.zeros((NGP_N_rays, 3), dtype=data_type)
    # use the pre-compute direction and scene pose
    NGP_directions = torch.zeros((NGP_N_rays, 3), dtype=data_type)
    NGP_pose = torch.zeros((3, 4), dtype=data_type)

    # density_bitfield is used for point sampling
    NGP_density_bitfield = torch.zeros(
        (cascades * grid_size**3 // 32,), dtype=torch.uint32
    )

    # count the number of rays that still alive
    NGP_counter = torch.zeros((1,), dtype=torch.int32)
    NGP_counter[0] = NGP_N_rays
    # current alive buffer index
    NGP_current_index = torch.zeros((), dtype=torch.int32)
    # NGP_current_index[None] = 0
    init_current_index(NGP_current_index)

    # how many samples that need to run the model
    NGP_model_launch = torch.zeros((), dtype=torch.int32)

    # buffer for the alive rays
    NGP_alive_indices = torch.zeros((2 * NGP_N_rays,), dtype=torch.int32)

    # padd the thread to the factor of block size (thread per block)
    NGP_padd_block_network = torch.zeros((), dtype=torch.int32)

    # model parameters
    sigma_layer1_base = 16 * 16
    layer1_base = 32 * 16
    NGP_hash_embedding = torch.zeros((17956864,), dtype=data_type)
    NGP_sigma_weights = torch.zeros((sigma_layer1_base + 16 * 16,), dtype=data_type)
    NGP_rgb_weights = torch.zeros((layer1_base + 16 * 16,), dtype=data_type)

    # buffers that used for points sampling
    NGP_max_samples_per_rays = 1
    NGP_max_samples_shape = NGP_N_rays * NGP_max_samples_per_rays

    NGP_xyzs = torch.zeros((NGP_max_samples_shape, 3), dtype=data_type)
    NGP_dirs = torch.zeros((NGP_max_samples_shape, 3), dtype=data_type)
    NGP_deltas = torch.zeros((NGP_max_samples_shape,), dtype=data_type)
    NGP_ts = torch.zeros((NGP_max_samples_shape,), dtype=data_type)

    # buffers that store the info of sampled points
    NGP_run_model_ind = torch.zeros((NGP_max_samples_shape,), dtype=torch.int32)
    NGP_N_eff_samples = torch.zeros((NGP_N_rays,), dtype=torch.int32)

    # intermediate buffers for network
    NGP_xyzs_embedding = torch.zeros((NGP_max_samples_shape, 32), dtype=data_type)
    NGP_final_embedding = torch.zeros((NGP_max_samples_shape, 16), dtype=data_type)
    NGP_out_3 = torch.zeros((NGP_max_samples_shape, 3), dtype=data_type)
    NGP_out_1 = torch.zeros((NGP_max_samples_shape,), dtype=data_type)
    NGP_temp_hit = torch.zeros((NGP_max_samples_shape,), dtype=torch.int32)

    # results buffers
    NGP_opacity = torch.zeros((NGP_N_rays,), dtype=data_type)
    NGP_rgb = torch.zeros((NGP_N_rays, 3), dtype=data_type)

    update_model_weights(model)
    inference_local()
