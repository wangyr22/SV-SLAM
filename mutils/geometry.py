import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def transform_pts3d(pts3d, transform_matrix):
    """
    pts3d: ..., 3, torch tensor

    transform_matrix: 4, 4. numpy array or torch tensor

    return: transformed 3d points, same shape as pts3d.
    """
    if isinstance(transform_matrix, np.ndarray):
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32)
    transform_matrix = transform_matrix.to(pts3d.device)

    assert pts3d.shape[-1] == 3
    old_shape = pts3d.shape

    pts3d_transformed = pts3d.view(-1, 3)
    pts3d_transformed = torch.concatenate([pts3d_transformed, torch.ones_like(pts3d_transformed[..., [0]])], dim=-1)
    pts3d_transformed = pts3d_transformed @ transform_matrix.T
    pts3d_transformed = pts3d_transformed[..., :-1]
    pts3d_transformed = pts3d_transformed.view(old_shape)
    return pts3d_transformed

def torchvec_to_npmatrix(posevec):
    """posevec: 7,"""
    m = np.eye(4)
    m[:3, :3] = R.from_quat(posevec[3:].detach().cpu().numpy()).as_matrix()
    m[:3, -1] = posevec[:3].detach().cpu().numpy()
    return m

def npmatrix_to_torchvec(m, device):
    """m: 4, 4"""
    t = m[:3, -1]
    r = R.from_matrix(m[:3, :3]).as_quat()
    posevec = torch.tensor(
        np.concatenate([t, r], axis=0), device=device, dtype=torch.float32
    )
    return posevec

def matrix_to_degree(R):
    assert R.shape == (3, 3)
    return np.arccos(
        np.trace(R) * 0.5 - 0.5,
    ) * 180 / np.pi

def calculate_pose_error(pred, gt):
    """
    inputs: 4x4 npdarrays. Sim3 rotations will be normalized first.
    
    returns: r_error_degree, t_erre
    """
    rerr = gt[:3, :3].T @ pred[:3, :3]
    scale = np.linalg.det(rerr)
    rerr = rerr / np.power(scale, 1/3.)
    r_error_degree = matrix_to_degree(rerr)
    t_error = np.sqrt(
        np.sum( np.square( gt[:3, -1] - pred[:3, -1] ) )
        )
    return r_error_degree, t_error

@torch.no_grad()
def save_ply(dest, poses, matches):
    import open3d as o3d

    pts_raw = []
    color_raw = []
    for m in (matches.values()):
        i, j = m['i'], m['j']
        pts3d_i, pts3d_j = m['pts3d_i'], m['pts3d_j']
        colors_i, colors_j = m['colors_i'], m['colors_j']

        pts3d_i = pts3d_i @ poses[i, :3, :3].T + poses[None, i, :3, -1]
        pts3d_j = pts3d_j @ poses[j, :3, :3].T + poses[None, j, :3, -1]

        pts_raw.append(pts3d_i)
        color_raw.append(colors_i)
        pts_raw.append(pts3d_j)
        color_raw.append(colors_j)

    pts_raw = torch.concatenate(pts_raw, 0).cpu().numpy()
    color_raw = torch.concatenate(color_raw, 0).cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_raw)
    pcd.colors = o3d.utility.Vector3dVector(color_raw)
    
    print(f"Saving pointcloud to '{dest}'")
    o3d.io.write_point_cloud(dest, pcd)