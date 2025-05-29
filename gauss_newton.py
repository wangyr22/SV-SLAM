# 25/2/?. wyr
from time import perf_counter
from tqdm import tqdm
from einops import rearrange

import numpy as np
import torch
from lietorch import Sim3

def transform_to_frame(pts3d, transform_matrix):
    return pts3d @ transform_matrix[:3, :3].T + transform_matrix[None, :3, -1]

def fill_lower_triangle(m):
    n1, n2 = m.shape
    assert n1 == n2
    
    grid = np.mgrid[:n1, :n2]
    is_lower = (grid[0] > grid[1])

    if isinstance(m, torch.Tensor):
        grid = torch.tensor(grid).to(m.device)

    m[is_lower] = m.T[is_lower]
    return m

def make_blocks(pp, qq, cc=None):
    """pp, qq: N, 3. cc: N,"""

    n = pp.shape[0]

    if isinstance(pp, np.ndarray):
        blocks_pp = np.tile(np.eye(7), [n, 1, 1])
        blocks_qq = np.tile(np.eye(7), [n, 1, 1])
        blocks_pq_cross = np.tile(np.eye(7), [n, 1, 1])
        pp_xx_qq = np.zeros(n, 3)
        neg_pxqx = np.zeros(n, 3, 3)
        eye_33 = np.tile(np.eye(3, 3), [n, 1, 1])
        vecs_pp = np.zeros(n, 7)
        vecs_qq = np.zeros(n, 7)
        pkg = np
    if isinstance(pp, torch.Tensor):
        device = pp.device
        blocks_pp = torch.tile(torch.eye(7), [n, 1, 1]).to(device)
        blocks_qq = torch.tile(torch.eye(7), [n, 1, 1]).to(device)
        blocks_pq_cross = torch.tile(torch.eye(7), [n, 1, 1]).to(device)
        pp_xx_qq = torch.zeros(n, 3).to(device)
        neg_pxqx = torch.zeros(n, 3, 3).to(device)
        eye_33 = torch.tile(torch.eye(3, 3), [n, 1, 1]).to(device)
        vecs_pp = torch.zeros(n, 7).to(device)
        vecs_qq = torch.zeros(n, 7).to(device)
        pkg = torch

    pp2 = pp ** 2
    pp2_sum = pp2.sum(-1)
    qq2 = qq ** 2
    qq2_sum = qq2.sum(-1)
    pq = pkg.matmul( pp[:, :, None], qq[:, None, :] )
    # Cross product
    pp_xx_qq[:, 0] = pq[:, 1, 2] - pq[:, 2, 1]
    pp_xx_qq[:, 1] = -pq[:, 0, 2] + pq[:, 2, 0]
    pp_xx_qq[:, 2] = pq[:, 0, 1] - pq[:, 1, 0]
    pp_dot_qq = pq[:, 0, 0] + pq[:, 1, 1] + pq[:, 2, 2]
    # -[p]x @ [q]x
    neg_pxqx = eye_33 * pp_dot_qq[..., None, None] - rearrange(pq, 'n h w -> n w h')

    for blocks_self, pts, pts2, pts2_sum in [(blocks_pp, pp, pp2, pp2_sum), (blocks_qq, qq, qq2, qq2_sum)]:
        blocks_self[:, :3, -1] = pts
        blocks_self[:, 1, 5] = pts[:, 0]
        blocks_self[:, 2, 4] = -pts[:, 0]
        blocks_self[:, 0, 5] = -pts[:, 1]
        blocks_self[:, 2, 3] = pts[:, 1]
        blocks_self[:, 0, 4] = pts[:, 2]
        blocks_self[:, 1, 3] = -pts[:, 2]

        blocks_self[:, 6, 6] = pts2_sum

        blocks_self[:, 3, 3] = pts2[:, 1] + pts2[:, 2]
        blocks_self[:, 4, 4] = pts2[:, 0] + pts2[:, 2]
        blocks_self[:, 5, 5] = pts2[:, 0] + pts2[:, 1]

        blocks_self[:, 3, 4] = -pts[:, 0] * pts[:, 1]
        blocks_self[:, 3, 5] = -pts[:, 0] * pts[:, 2]
        blocks_self[:, 4, 5] = -pts[:, 1] * pts[:, 2]

    blocks_pq_cross[:, 3:6, :3] = -blocks_pp[:, :3, 3:6]
    blocks_pq_cross[:, :3, 3:6] = blocks_qq[:, :3, 3:6]
    blocks_pq_cross[:, -1, :3] = pp
    blocks_pq_cross[:, :3, -1] = qq
    blocks_pq_cross[:, 3:6, 3:6] = neg_pxqx
    blocks_pq_cross[:, 3:6, -1] = pp_xx_qq
    blocks_pq_cross[:, -1, 3:6] = -pp_xx_qq
    blocks_pq_cross[:, 6, 6] = pp_dot_qq

    vecs_pp[:, :3] = pp - qq
    vecs_pp[:, 3:6] = -pp_xx_qq
    vecs_pp[:, -1] = pp2_sum - pp_dot_qq
    vecs_qq[:, :3] = -vecs_pp[:, :3]
    vecs_qq[:, 3:6] = pp_xx_qq
    vecs_qq[:, -1] = qq2_sum - pp_dot_qq

    if cc is not None:
        blocks_pp = blocks_pp * cc[..., None, None]
        blocks_qq = blocks_qq * cc[..., None, None]
        blocks_pq_cross = blocks_pq_cross * cc[..., None, None]
        vecs_pp = vecs_pp * cc[..., None]
        vecs_qq = vecs_qq * cc[..., None]

    # Returns ready-to-use matrices.
    return fill_lower_triangle(blocks_pp.sum(0)), fill_lower_triangle(blocks_qq.sum(0)), -blocks_pq_cross.sum(0), -vecs_pp.sum(0), -vecs_qq.sum(0)

@torch.no_grad()
def preprocess_conf(matches, q=0.5):
    for m in matches.values():
        pts3d_i, pts3d_j = m['pts3d_i'], m['pts3d_j']

        md_i = torch.quantile(pts3d_i[:, -1], q=q, dim=0, interpolation='nearest')
        md_j = torch.quantile(pts3d_j[:, -1], q=q, dim=0, interpolation='nearest')

        conf_scale_i = torch.clip((md_i / pts3d_i[:, -1]), min=None, max=1)
        conf_scale_j = torch.clip((md_j / pts3d_j[:, -1]), min=None, max=1)

        conf_scale = torch.min(conf_scale_i, conf_scale_j)

        m['conf'] = m['conf'] * conf_scale
    
    return matches

@torch.no_grad()
def solve_sim3(poses, matches, device, iters,
               use_conf=True, stopping_criterion=0.005, use_pts_ratio=0.1,
               clip_val=2.5, verbose=True):
    
    tik = perf_counter()
    num_frames = poses.shape[0]

    # Keep frame 0 fixed.
    num_var_frames = num_frames - 1
    dof = 7
    
    for it in range(iters):
        H = torch.zeros([dof * num_var_frames, dof * num_var_frames], device=device)
        B = torch.zeros([dof * num_var_frames], device=device)

        obj = 0
        pts_nums = 0

        for m in tqdm(matches.values(), desc='Matches'):
            i, j = m['i'], m['j']
            pts3d_i, pts3d_j = m['pts3d_i'], m['pts3d_j']
            
            # Use transformed points
            pts3d_i = transform_to_frame(pts3d_i, poses[i])
            pts3d_j = transform_to_frame(pts3d_j, poses[j])
            
            conf = m['conf']
            if not use_conf:
                conf = torch.ones_like(conf)

            if i == 0:
                i, j = j, i
                pts3d_i, pts3d_j = pts3d_j, pts3d_i

            num_pts = pts3d_i.shape[0]

            # Only takes a small subset of all matched points
            nums = int(num_pts * use_pts_ratio)
            pts_nums += nums
            _, ks = torch.topk(conf, nums)

            obj_m = 0

            i_idx = (i - 1) * dof
            j_idx = (j - 1) * dof

            ppi = pts3d_i[ks, :]
            ppj = pts3d_j[ks, :]
            cc = conf[ks].squeeze(-1)
            bi, bj, bij, vi, vj = make_blocks(ppi, ppj, cc)
            H[i_idx: i_idx+dof, i_idx: i_idx+dof] += bi
            B[i_idx: i_idx+dof] += vi
            if j != 0:
                H[j_idx: j_idx+dof, j_idx: j_idx+dof] += bj
                H[i_idx: i_idx+dof, j_idx: j_idx+dof] += bij
                H[j_idx: j_idx+dof, i_idx: i_idx+dof] += bij.T
                B[j_idx: j_idx+dof] += vj

            obj_m = (ppi - ppj).square().sum().item()

            obj += obj_m
            # if nums > 0:
            #     obj_m = np.sqrt(obj_m / nums)
            #     print(f"Step {it} {i} <-> {j} obj: {obj_m}")
            # else:
            #     print(f"Step {it} {i} <-> {j} nums = 0.")

        # Normalize to average per-point l2 distance. Human-readable
        obj = np.sqrt( obj / num_pts )

        if np.isnan(obj):
            breakpoint()

        # Torch doc: 
        # It is always preferred to use solve() when possible,
        # as it is faster and more numerically stable than computing the inverse explicitly.
        # H_inv = torch.linalg.inv(H.double()).float()
        # step = H_inv @ B
        try:
            step = torch.linalg.solve(H.double(), B.double()).float()
        except:
            breakpoint()

        # Clipping
        step = torch.clip(step, min=-clip_val, max=clip_val)

        # Average per-pose norm
        norm = torch.norm( step / num_var_frames )
        # print(f"max: {step.max()}")
        # print(f"min: {step.min()}")
        # print(f"25%: {torch.quantile(step, 0.25, interpolation='nearest')}")
        # print(f"50%: {torch.quantile(step, 0.50, interpolation='nearest')}")
        # print(f"75%: {torch.quantile(step, 0.75, interpolation='nearest')}")

        # Update pose
        for i in range(1, num_frames):
            i_idx = (i - 1) * dof
            # delta_pose_i = exp_se3( step[i_idx: i_idx+dof] )
            delta_pose_i = Sim3.exp( step[i_idx: i_idx+dof] ).matrix()
            poses[i] = delta_pose_i @ poses[i]

        if verbose:
            print(f"Step {it} obj: {obj}")
            print(f"Step {it} step norm: {norm}")
        if norm < stopping_criterion:
            if verbose:
                print(f"Stopping criterion is met.")
            break
    
    tok = perf_counter()

    return poses, tok - tik
