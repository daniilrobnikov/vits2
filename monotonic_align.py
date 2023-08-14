import torch
import numba
from numba import cuda
import numpy as np


# ! ----------------------------- CUDA monotonic_align.py -----------------------------


@cuda.jit("void(int32[:,:,:], float32[:,:,:], int32[:], int32[:])")
def maximum_path_cuda_jit(paths, values, t_ys, t_xs):
    max_neg_val = -1e9
    i = cuda.grid(1)
    if i >= paths.shape[0]:  # exit if the thread is out of the index range
        return

    path = paths[i]
    value = values[i]
    t_y = t_ys[i]
    t_x = t_xs[i]

    v_prev = v_cur = 0.0
    index = t_x - 1

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = value[y - 1, x] if x != y else max_neg_val
            v_prev = value[y - 1, x - 1] if x != 0 else (0.0 if y == 0 else max_neg_val)
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1

    # Wait for other threads in this block
    cuda.syncthreads()


def maximum_path_cuda(neg_cent: torch.Tensor, mask: torch.Tensor):
    """
    Monotonic alignment search algorithm
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype

    neg_cent_device = cuda.as_cuda_array(neg_cent)
    path_device = cuda.device_array(neg_cent.shape, dtype=np.int32)
    t_t_max_device = cuda.as_cuda_array(mask.sum(1, dtype=torch.int32)[:, 0])
    t_s_max_device = cuda.as_cuda_array(mask.sum(2, dtype=torch.int32)[:, 0])

    threadsperblock = 32
    blockspergrid = (path_device.shape[0] + (threadsperblock - 1)) // threadsperblock

    maximum_path_cuda_jit[blockspergrid, threadsperblock](path_device, neg_cent_device, t_t_max_device, t_s_max_device)

    # Convert device array back to tensor
    path = torch.as_tensor(path_device.copy_to_host(), device=device, dtype=dtype)

    return path


# ! ----------------------------- CPU monotonic_align.py -----------------------------


def maximum_path(neg_cent, mask):
    """numba optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


@numba.jit(numba.void(numba.int32[:, :, ::1], numba.float32[:, :, ::1], numba.int32[::1], numba.int32[::1]), nopython=True, nogil=True)
def maximum_path_jit(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(int(b)):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        v_prev = v_cur = 0.0
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index = index - 1
