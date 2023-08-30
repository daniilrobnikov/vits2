import math
import torch
import torch.nn.functional as F
import numba
import numpy as np
from numba import cuda

from utils.model import sequence_mask, convert_pad_shape


# * Ready and Tested
def search_path(z_p, m_p, logs_p, x_mask, y_mask, mas_noise_scale=0.01):
    with torch.no_grad():
        o_scale = torch.exp(-2 * logs_p)  # [b, d, t]
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t]
        logp2 = torch.matmul(-0.5 * (z_p**2).mT, o_scale)  # [b, t', d] x [b, d, t] = [b, t', t]
        logp3 = torch.matmul(z_p.mT, (m_p * o_scale))  # [b, t', d] x [b, d, t] = [b, t', t]
        logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1], keepdim=True)  # [b, 1, t]
        logp = logp1 + logp2 + logp3 + logp4  # [b, t', t]

        if mas_noise_scale > 0.0:
            epsilon = torch.std(logp) * torch.randn_like(logp) * mas_noise_scale
            logp = logp + epsilon

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  # [b, 1, t] * [b, t', 1] = [b, t', t]
        attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t', t] maximum_path_cuda
    return attn


def generate_path(duration: torch.Tensor, mask: torch.Tensor):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).mT * mask
    return path


# ! ----------------------------- CUDA monotonic_align.py -----------------------------


# TODO test for the optimal blockspergrid and threadsperblock values
def maximum_path_cuda(neg_cent: torch.Tensor, mask: torch.Tensor):
    """CUDA optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype

    neg_cent_device = cuda.as_cuda_array(neg_cent)
    path_device = cuda.device_array(neg_cent.shape, dtype=np.int32)
    t_t_max_device = cuda.as_cuda_array(mask.sum(1, dtype=torch.int32)[:, 0])
    t_s_max_device = cuda.as_cuda_array(mask.sum(2, dtype=torch.int32)[:, 0])

    blockspergrid = neg_cent.shape[0]
    threadsperblock = max(neg_cent.shape[1], neg_cent.shape[2])

    maximum_path_cuda_jit[blockspergrid, threadsperblock](path_device, neg_cent_device, t_t_max_device, t_s_max_device)

    # Convert device array back to tensor
    path = torch.as_tensor(path_device.copy_to_host(), device=device, dtype=dtype)
    return path


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
    cuda.syncthreads()


# ! ------------------------------- CPU monotonic_align.py -------------------------------


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor):
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
