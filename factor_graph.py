import sys
sys.path.append("./thirdparty/mast3r")

from tqdm import tqdm
import os
from os.path import join as pjoin
from time import perf_counter
from tqdm import trange

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.utils.image import load_images as _d_load_images
from dust3r.inference import inference

from gauss_newton import solve_sim3, preprocess_conf
from mutils.geometry import transform_pts3d, calculate_pose_error

def load_image(path, device):
    view, = _d_load_images([path], size=512, verbose=False)
    out = dict(
        img=view['img'].to(device),
        true_shape=torch.tensor(view['true_shape']),
        instance=view['instance']
    )
    return out

def masked_fast_NN(desc_cf, desc_kf, mask_cf, mask_kf, device):
    """
    mask: (h, w)
    """
    hc, wc, dimc = desc_cf.shape
    hk, wk, dimk = desc_kf.shape
    
    if mask_cf is None:
        mask_cf = np.ones([hc, wc], dtype=bool)
    if mask_kf is None:
        mask_kf = np.ones([hk, wk], dtype=bool)

    original_grid_c = np.mgrid[:hc, :wc].transpose(1, 2, 0)[..., [1,0]]
    original_grid_k = np.mgrid[:hk, :wk].transpose(1, 2, 0)[..., [1,0]]

    # 1, n, dim
    desc_cf = desc_cf[None, mask_cf, :]
    desc_kf = desc_kf[None, mask_kf, :]
    original_grid_c = original_grid_c[None, mask_cf, :]
    original_grid_k = original_grid_k[None, mask_kf, :]

    # n', 2
    _matches_cf, _matches_kf = fast_reciprocal_NNs(desc_cf, desc_kf, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # Get true pixel indices
    matches_cf = original_grid_c[_matches_cf[:, 1], _matches_cf[:, 0], :]
    matches_kf = original_grid_k[_matches_kf[:, 1], _matches_kf[:, 0], :]

    return matches_cf, matches_kf

class FactorGraph:
    def __init__(self, mast3r_ckpt,
                 device, K, imgsize, buffer=512, use_K=True,
                 merge_pointmap=True,
                 lc_thresh_degree=15.,
                 filter_by_3d=False,
                 thresh_inlier=0.03,
                 use_pts_ratio=0.1,
                 use_ds=True,
                 run_desc=''):
        self.ht, self.wd = imgsize
        self.device = device
        self.m3r = AsymmetricMASt3R.from_pretrained(mast3r_ckpt).to(device)

        self.merge_pointmap = merge_pointmap

        self.total_frames = 0
        self.t = []
        self.views = []

        self.buffer = buffer
        self.poses = torch.tile(torch.eye(4), [buffer, 1, 1]).to(device)
        self._identity_pose = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32,  device=device)
        self.pts3d = torch.zeros([buffer, self.ht, self.wd, 3], device=device)
        self.pts3d_conf = torch.zeros([buffer, self.ht, self.wd], device=device)
        
        grid_size = 16
        dh, dw = (self.ht - 7) // (grid_size - 1), (self.wd - 7) // (grid_size - 1)
        self.supp_grid = torch.tensor(np.mgrid[3: self.ht-3: dh, 3: self.wd-3: dw], device=device).reshape(2, -1).permute(1, 0)
        self.supp_pts_thresh = 25.9

        self.ii = []
        self.jj = []
        # TODO: optimize to save memory
        self.matches = []

        self.use_K = use_K
        self.K = K
        self.K_torch = torch.tensor(K, device=self.device, dtype=torch.float32)
        self.K_inv_torch = torch.linalg.inv(self.K_torch)

        self.pnp_thresh = 5
        self.pnp_iters = 100

        self.filter_by_3d = filter_by_3d
        self.thresh_inlier = thresh_inlier
        self.sqthresh_inlier = thresh_inlier ** 2
        self.use_pts_ratio = use_pts_ratio
        self.use_ds = use_ds

        self.lc_thresh_degree = lc_thresh_degree
        self.lc_thresh_m = 100.

        # Timers
        self.time_enc_tracking = 0.
        self.time_dec_tracking = 0.
        self.time_match_tracking = 0.
        self.time_pnp_tracking = 0.
        self.time_enc_loop = 0.
        self.time_dec_loop = 0.
        self.time_match_loop = 0.
        self.time_pnp_loop = 0.
        self.time_ba = 0.

        self.run_desc = run_desc

    def export_kitti_fmt(self, out_path):
        out = []
        for i in range(self.total_frames):
            # pose_i = SE3(self.poses[i]).matrix().detach().cpu().numpy()
            pose_i = self.poses[i].cpu().numpy()
            scale = np.linalg.det(pose_i[:3, :3])
            pose_i[:3, :3] = pose_i[:3, :3] / np.power(scale, 1/3.)
            out.append(
                ' '.join( [str(x) for x in pose_i.reshape(-1)[:12]] ) + '\n'
            )
        with open(out_path, 'w') as f:
            f.writelines(out)

    def update_pts3d(self, idx, new_pts3d, new_conf):
        """shape: [ht, wd, 3], [ht, wd]"""
        new_conf = new_conf.squeeze(0)[..., None]
        
        pts3d_sum = new_pts3d * new_conf + self.pts3d[idx] * self.pts3d_conf[idx, ..., None]
        conf_sum = new_conf + self.pts3d_conf[idx, ..., None]
        self.pts3d[idx] = pts3d_sum / conf_sum
        self.pts3d_conf[idx] = conf_sum.squeeze(-1)

    def structure_matches(self):
        """Save tensors on GPU."""

        num_pairs = len(self.ii)
        matches = {}
        for k in range(num_pairs):
            i = self.ii[k]
            j = self.jj[k]
            match_data = self.matches[k]

            matches_i = match_data['matches_i']
            if self.use_K:
                depths_i = self.pts3d[i, matches_i[:, 1], matches_i[:, 0], -1]
                pts3d_i = torch.concatenate([matches_i, torch.ones_like(matches_i[:, [0]])], -1) * depths_i[..., None]
                pts3d_i = pts3d_i @ self.K_inv_torch.T
            else:
                pts3d_i = self.pts3d[i, matches_i[:, 1], matches_i[:, 0], :]
            colors_i = self.views[i]['img'].squeeze(0).permute(1, 2, 0)[matches_i[:, 1], matches_i[:, 0], :] * 0.5 + 0.5

            matches_j = match_data['matches_j']
            if self.use_K:
                depths_j = self.pts3d[j, matches_j[:, 1], matches_j[:, 0], -1]
                pts3d_j = torch.concatenate([matches_j, torch.ones_like(matches_j[:, [0]])], -1) * depths_j[..., None]
                pts3d_j = pts3d_j @ self.K_inv_torch.T
            else:
                pts3d_j = self.pts3d[j, matches_j[:, 1], matches_j[:, 0], :]
            colors_j = self.views[j]['img'].squeeze(0).permute(1, 2, 0)[matches_j[:, 1], matches_j[:, 0], :] * 0.5 + 0.5

            matches[k] = {
                'i': i, 'pts3d_i': pts3d_i, 'colors_i': colors_i,
                'j': j, 'pts3d_j': pts3d_j, 'colors_j': colors_j,
                'conf': match_data['matches_conf'].clone()
            }

        if self.use_ds:
            matches = preprocess_conf(matches)

        # For running G-N optimization.
        return matches

    def run_ba(self, iters=5, stopping_criterion=0.005):
        print(f"Running global ba.")
        tik = perf_counter()
        matches = self.structure_matches()
        tok = perf_counter()
        time_ba_preprocess = tok - tik
        _, time_gn = solve_sim3(self.poses[:self.total_frames], matches, self.device,
                   iters=iters, use_conf=True, stopping_criterion=stopping_criterion, use_pts_ratio=self.use_pts_ratio)
        self.time_ba += time_gn + time_ba_preprocess

    def solve_pnp(self, view_cf, view_kf, pred_cf, pred_kf, mask_cf=None, mask_kf=None, initial_pose=None, data_desc=None):
        """
        initial_pose: 4x4 kf-to-cf ndarray
        """
        desc_cf, desc_kf = pred_cf['desc'].squeeze(0), pred_kf['desc'].squeeze(0)
        try:
            tik = perf_counter()
            matches_im_cf, matches_im_kf = masked_fast_NN(desc_cf, desc_kf, mask_cf=mask_cf, mask_kf=mask_kf, device=self.device)
            tok = perf_counter()
            time_match = tok - tik
        except:
            return None

        if data_desc is not None:
            tik = perf_counter()
            _matches_cf, _matches_kf = fast_reciprocal_NNs(desc_cf, desc_kf, subsample_or_initxy1=8, device='cuda:0', dist='dot', block_size=2**13)
            tok = perf_counter()
            time_full_match = tok - tik
            print(f"Masked matching: {time_match * 1e3:.3f} ms")
            print(f"Full matching: {time_full_match * 1e3:.3f} ms")
            conf_cf = pred_cf['desc_conf'][0, matches_im_cf[:, 1], matches_im_cf[:, 0]]
            conf_kf = pred_kf['desc_conf'][0, matches_im_kf[:, 1], matches_im_kf[:, 0]]
            matches_conf = torch.sqrt(conf_cf * conf_kf).cpu().numpy()
            _conf_cf = pred_cf['desc_conf'][0, _matches_cf[:, 1], _matches_cf[:, 0]]
            _conf_kf = pred_kf['desc_conf'][0, _matches_kf[:, 1], _matches_kf[:, 0]]
            _matches_conf = torch.sqrt(_conf_cf * _conf_kf).cpu().numpy()
            np.save(data_desc + '_masked_matches_im_cf.npy', matches_im_cf)
            np.save(data_desc + '_masked_matches_im_kf.npy', matches_im_kf)
            np.save(data_desc + '_vanilla_matches_im_cf.npy', _matches_cf)
            np.save(data_desc + '_vanilla_matches_im_kf.npy', _matches_kf)
            np.save(data_desc + '_masked_conf.npy', matches_conf)
            np.save(data_desc + '_vanilla_conf.npy', _matches_conf)

        # ignore small border around the edge
        H0, W0 = view_cf['true_shape'][0]
        valid_matches_im_cf = (matches_im_cf[:, 0] >= 3) & (matches_im_cf[:, 0] < int(W0) - 3) & (
            matches_im_cf[:, 1] >= 3) & (matches_im_cf[:, 1] < int(H0) - 3)
        H1, W1 = view_kf['true_shape'][0]
        valid_matches_im_kf = (matches_im_kf[:, 0] >= 3) & (matches_im_kf[:, 0] < int(W1) - 3) & (
            matches_im_kf[:, 1] >= 3) & (matches_im_kf[:, 1] < int(H1) - 3)
        
        if self.filter_by_3d:
            pts3d_kf = pred_kf['pts3d'][0, matches_im_kf[:, 1], matches_im_kf[:, 0]]
            pts3d_cf = pred_cf['pts3d_in_other_view'][0, matches_im_cf[:, 1], matches_im_cf[:, 0]]
            inlier_3dspace = (pts3d_kf - pts3d_cf).square().sum(-1, keepdim=False) <= self.sqthresh_inlier
            valid_matches = valid_matches_im_cf & valid_matches_im_kf & inlier_3dspace.cpu().numpy()
            # print(f'outlier rate: {(~inlier_3dspace).float().mean()}, cnt: {(~inlier_3dspace).sum()}')
        else:
            valid_matches = valid_matches_im_cf & valid_matches_im_kf
        matches_im_cf, matches_im_kf = matches_im_cf[valid_matches], matches_im_kf[valid_matches]
        pts3d_kf = pred_kf['pts3d'][0, matches_im_kf[:, 1], matches_im_kf[:, 0]]
        pts2d_cf = matches_im_cf
        pts3d_kf = pts3d_kf.detach().cpu().numpy()
        pts2d_cf = pts2d_cf.astype(np.float64)

        try:
            tik = perf_counter()
            if initial_pose is not None:
                rvec, _ = cv2.Rodrigues(initial_pose[:3, :3])
                tvec = initial_pose[:3, [-1]].copy()
                # tik = perf_counter()
                success, r_pose, t_pose, inliers = cv2.solvePnPRansac(
                   pts3d_kf, pts2d_cf, self.K, None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE,
                   reprojectionError=self.pnp_thresh, iterationsCount=self.pnp_iters,
                )
                # tok = perf_counter()
                # print(f"Iterative pnpransac time: {(tok - tik) * 1e3:.3f} ms")
            else:
                success, r_pose, t_pose, inliers = cv2.solvePnPRansac(
                    pts3d_kf, pts2d_cf, self.K, None, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=self.pnp_thresh, iterationsCount=self.pnp_iters
                )
            tok = perf_counter()
            time_pnp = tok - tik
        except:
            print(f"pnp failed.")
            return None
        if not success:
            print(f"pnp failed.")
            return None

        pose_pred = np.eye(4)
        pose_pred[:3, :3] = cv2.Rodrigues(r_pose)[0]
        pose_pred[:3, [-1]] = t_pose
        # cv2 estimates world-to-camera(kf-to-cf). We need cf-to-kf.

        if initial_pose is not None:
            # Calculate error and filter out possibly wrong matches.
            r_error, t_error = calculate_pose_error(pose_pred, initial_pose)
            if r_error > self.lc_thresh_degree or t_error > self.lc_thresh_m:
                print(f"pnp inconsistent.")
                return None

        pose_pred = np.linalg.inv(pose_pred)

        matches_im_cf = torch.tensor(matches_im_cf[inliers[:, 0]], device=self.device)
        matches_im_kf = torch.tensor(matches_im_kf[inliers[:, 0]], device=self.device)

        conf_cf = pred_cf['desc_conf'][0, matches_im_cf[:, 1], matches_im_cf[:, 0]]
        conf_kf = pred_kf['desc_conf'][0, matches_im_kf[:, 1], matches_im_kf[:, 0]]
        # shape: N,
        matches_conf = torch.sqrt(conf_cf * conf_kf)

        # Normalize matches_conf
        matches_conf = matches_conf / matches_conf.max()

        return pose_pred, matches_im_cf, matches_im_kf, matches_conf, time_match, time_pnp
    
    def build_match(self, i, j, mask=[None, None]):
        """
        mask: [2, h, w]. 1st ~ i, 2nd ~ j
        """
        view_kf = self.views[i]
        view_cf = self.views[j]

        # Infer
        with torch.no_grad():
            pred_kf, pred_cf, time_enc, time_dec = self.m3r(view_kf, view_cf)
        self.time_enc_loop += time_enc
        self.time_dec_loop += time_dec

        # Solve & get inlier correspondence
        # initial_pose = np.linalg.inv(torchvec_to_npmatrix(self.poses[j])) @ torchvec_to_npmatrix(self.poses[i])
        initial_pose = (torch.linalg.inv(self.poses[j]) @ self.poses[i]).cpu().numpy()

        ret = self.solve_pnp(view_cf, view_kf, pred_cf, pred_kf, mask_kf=mask[0], mask_cf=mask[1], initial_pose=initial_pose)

        if ret is None:
            # Failed
            print(f"Failed building matche between {i} and {j}.")
            return False

        cf2kf, matches_cf, matches_kf, matches_conf, time_match, time_pnp = ret
        self.time_match_loop += time_match
        self.time_pnp_loop += time_pnp

        # Update depth
        if self.merge_pointmap:
            self.update_pts3d(i, pred_kf['pts3d'], pred_kf['conf'])
            self.update_pts3d(j, 
                          transform_pts3d(pred_cf['pts3d_in_other_view'], np.linalg.inv(cf2kf)), pred_cf['conf'] )
        
        # Store correspondence
        self.ii.append(i)
        self.jj.append(j)
        self.matches.append(
            { 'matches_i': matches_kf, 'matches_j': matches_cf, 'matches_conf': matches_conf }
        )

        return True

    def supplement_match(self, i, dist_min=10, num_edges=1, max_attemps=5):
        """
        Supplementary loop detection.
        """
        print(f"Supplementing match for {i}({self.t[i]})")
        
        pose_i = self.poses[i].cpu().numpy()
        pts3d_i = self.pts3d[i, self.supp_grid[:, 0], self.supp_grid[:, 1], :]
        pts3d_i = transform_pts3d(pts3d_i, pose_i)

        nums_ok = 0
        nums_attempt = 0

        for j in range(i - dist_min):
            pose_j = self.poses[j].cpu().numpy()
            pts3d_i_in_j = transform_pts3d(pts3d_i, np.linalg.inv(pose_j))
            pts2d_i_in_j = pts3d_i_in_j @ self.K_torch.T
            pts2d_i_in_j = pts2d_i_in_j[:, :-1] / pts2d_i_in_j[:, [-1]]
        
            mask_x = (pts2d_i_in_j[:, 0] > 0) & (pts2d_i_in_j[:, 0] < self.wd)
            mask_y = (pts2d_i_in_j[:, 1] > 0) & (pts2d_i_in_j[:, 1] < self.ht)
        
            valid_pts = (mask_x & mask_y).float().sum()

            if valid_pts > self.supp_pts_thresh:
                nums_attempt += 1

                success = self.build_match(i, j)
                if success:
                    print(f"Succeeded in building loop edge: {i}({self.t[i]}) <-> {j}({self.t[j]}). valid_pts = {valid_pts}")
                    nums_ok += 1
                
                if nums_ok >= num_edges:
                    return True
                
                if nums_attempt >= max_attemps:
                    break

        if nums_ok > 0:
            return True
        else:
            return False

    def track_new_frame(self, view, mask=[None, None], t=None):
        """
        img(view) shall be loaded with load_image.

        mask: (2, H, W) if any. 1st is for cf.
        """

        current_frame_idx = self.total_frames
        
        if self.total_frames == 0:
            # 1st frame
            self.total_frames += 1
            self.t.append(t)
            self.views.append(view)
            return np.eye(4)
        
        # Infer with MASt3r
        view_kf = self.views[-1]
        with torch.no_grad():
            pred_kf, pred_cf, time_enc, time_dec = self.m3r(view_kf, view)
        self.time_enc_tracking += time_enc
        self.time_dec_tracking += time_dec

        # Solve pose with PnP
        ret = self.solve_pnp(view, view_kf, pred_cf, pred_kf, mask[0], mask[1])
        if ret is None:
            # Tracking has failed.
            raise RuntimeError("Tracking fails.")
        cf2kf, matches_cf, matches_kf, matches_conf, time_match, time_pnp = ret

        self.time_match_tracking += time_match
        self.time_pnp_tracking += time_pnp

        # Update per-frame depth
        if current_frame_idx == 1 or self.merge_pointmap:
            self.update_pts3d(current_frame_idx - 1, pred_kf['pts3d'], pred_kf['conf'])
        self.update_pts3d(current_frame_idx, 
                          transform_pts3d(pred_cf['pts3d_in_other_view'], np.linalg.inv(cf2kf)), pred_cf['conf'] )
        
        # Store correspondence
        self.ii.append(current_frame_idx - 1)
        self.jj.append(current_frame_idx)
        self.matches.append(
            { 'matches_i': matches_kf, 'matches_j': matches_cf, 'matches_conf': matches_conf }
        )

        # Update state variables
        pose_kf = self.poses[current_frame_idx - 1]
        pose_kf = pose_kf.cpu().numpy()
        pose_cf = pose_kf @ cf2kf
        pose_cf = torch.tensor(pose_cf).float().to(self.device)
        self.poses[current_frame_idx] = pose_cf
        
        if self.total_frames == 1:
            self.views[-1].update(pred_kf)
        view.update(pred_cf)
        self.total_frames += 1
        self.t.append(t)
        self.views.append(view)

        return cf2kf
    
    def save_all_depths(self, outdir):
        for idx in range(self.total_frames):
            depth = self.pts3d[idx].detach().cpu().numpy()
            t = self.t[idx]
            plt.imsave( pjoin(outdir, f"{idx}_{t}.jpg"), depth)

    @torch.no_grad()
    def export_pointcloud_associated(self, dest):
        import open3d as o3d

        pts_raw = []
        color_raw = []

        for i, j, matches in zip(self.ii, self.jj, self.matches):
            matches_i = matches['matches_i']
            matches_j = matches['matches_j']
            
            pose_i = self.poses[i]
            pose_j = self.poses[j]
            
            if self.use_K:
                depths_i = self.pts3d[i, matches_i[:, 1], matches_i[:, 0], -1]
                pts3d_i = torch.concatenate([matches_i, torch.ones_like(matches_i[:, [0]])], -1) * depths_i[..., None]
                pts3d_i = pts3d_i @ self.K_inv_torch.T
            else:
                pts3d_i = self.pts3d[i, matches_i[:, 1], matches_i[:, 0], :]
            # pts3d_i = pose_i.act(pts3d_i)
            pts3d_i = transform_pts3d(pts3d_i, pose_i)
            colors_i = self.views[i]['img'].squeeze(0).permute(1, 2, 0)[matches_i[:, 1], matches_i[:, 0], :] * 0.5 + 0.5

            if self.use_K:
                depths_j = self.pts3d[j, matches_j[:, 1], matches_j[:, 0], -1]
                pts3d_j = torch.concatenate([matches_j, torch.ones_like(matches_j[:, [0]])], -1) * depths_j[..., None]
                pts3d_j = pts3d_j @ self.K_inv_torch.T
            else:
                depths_j = self.pts3d[j, matches_j[:, 1], matches_j[:, 0], :]
            # pts3d_j = pose_j.act(pts3d_j)
            pts3d_j = transform_pts3d(pts3d_j, pose_j)
            colors_j = self.views[j]['img'].squeeze(0).permute(1, 2, 0)[matches_j[:, 1], matches_j[:, 0], :] * 0.5 + 0.5

            pts_raw.append(pts3d_i.detach().cpu().numpy())
            color_raw.append(colors_i.detach().cpu().numpy())

            pts_raw.append(pts3d_j.detach().cpu().numpy())
            color_raw.append(colors_j.detach().cpu().numpy())
        
        pts_raw = np.concatenate(pts_raw, 0)
        color_raw = np.concatenate(color_raw, 0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_raw)
        pcd.colors = o3d.utility.Vector3dVector(color_raw)

        print(f"Saving pointcloud to '{dest}'")
        o3d.io.write_point_cloud(dest, pcd)

    @torch.no_grad()
    def export_per_frame_pcd(self, outdir, desc):
        import open3d as o3d

        # Save pose
        poses = self.poses[:self.total_frames].cpu().numpy()
        np.save(pjoin(outdir, desc + '-pose.npy'), poses)

        # Save pointclouds
        for i in trange(self.total_frames, desc=f'pcd-{desc}'):
            c2w = self.poses[i].cpu().numpy()
            color = ((self.views[i]['img'].squeeze(0).permute(1, 2, 0) * 0.5 + 0.5) * 255.).cpu().numpy()
            color = np.asarray(color, order='C')
            depth = self.pts3d[i, ..., [-1]].cpu().numpy()

            color = o3d.geometry.Image(color.astype(np.uint8))
            depth = o3d.geometry.Image(depth.astype(np.float32))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1, depth_trunc=100.0, convert_rgb_to_intensity=False
                )
            
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            camera = o3d.camera.PinholeCameraIntrinsic(self.wd, self.ht, fx, fy, cx, cy)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                camera)
            
            pcd.points = o3d.utility.Vector3dVector( np.asarray(pcd.points) @ c2w[:3, :3].T + c2w[:3, -1][None, ...] )

            o3d.io.write_point_cloud(pjoin(outdir, desc + '-' + f"{i}_{self.t[i]}.ply"), pcd)

    def export_pcd_all(self, dest):
        import open3d as o3d

        pcds = []
        print("Exporting pointcloud...")
        for i in trange(self.total_frames, desc='pointcloud'):
            # TODO: does this accept Sim3 extrinsics?
            c2w = self.poses[i].cpu().numpy()
            w2c = np.linalg.inv(self.poses[i].cpu().numpy())
            color = ((self.views[i]['img'].squeeze(0).permute(1, 2, 0) * 0.5 + 0.5) * 255.).cpu().numpy()
            color = np.asarray(color, order='C')
            depth = self.pts3d[i, ..., [-1]].cpu().numpy()

            color = o3d.geometry.Image(color.astype(np.uint8))
            depth = o3d.geometry.Image(depth.astype(np.float32))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1, depth_trunc=100.0, convert_rgb_to_intensity=False
                )
            
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            camera = o3d.camera.PinholeCameraIntrinsic(self.wd, self.ht, fx, fy, cx, cy)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                camera)
            
            pcd.transform(c2w.astype(np.float64))
            pcds.append(pcd)

        pcds = sum(pcds[1:], pcds[0])
        o3d.io.write_point_cloud(dest, pcds)

    def export_for_reconstruction(self, outdir):
        # export all depth maps and camera parameters.
        os.makedirs(outdir, exist_ok=True)
        pose_all = self.poses[:self.total_frames].cpu().numpy()
        np.save( pjoin(outdir, f'pose_se3_all.npy'), pose_all )
        for i in trange(self.total_frames, desc='Exporting mesh data'):
            color = ((self.views[i]['img'].squeeze(0).permute(1, 2, 0) * 0.5 + 0.5) * 255.).cpu().numpy().round().astype(np.uint8)
            depth = self.pts3d[i, ..., [-1]].cpu().numpy()
            pose_i = self.poses[i].cpu().numpy()

            cv2.imwrite( pjoin(outdir, f'{i:06d}_rgb.png'), cv2.cvtColor(color, cv2.COLOR_RGB2BGR) )
            np.save( pjoin(outdir, f'{i:06d}_depth.npy'), depth )

        print(f'Done exporting!')

    def export_mesh_all(self, dest):
        import open3d as o3d

        print("Exporting mesh...")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
        
        for i in trange(self.total_frames, desc='mesh'):
            # TODO: does this accept Sim3 extrinsics?
            w2c = np.linalg.inv(self.poses[i].cpu().numpy())
            color = ((self.views[i]['img'].squeeze(0).permute(1, 2, 0) * 0.5 + 0.5) * 255.).cpu().numpy()
            color = np.asarray(color, order='C')
            depth = self.pts3d[i, ..., [-1]].cpu().numpy()

            color = o3d.geometry.Image(color.astype(np.uint8))
            depth = o3d.geometry.Image(depth.astype(np.float32))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1, depth_trunc=100.0, convert_rgb_to_intensity=False
                )
            
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            camera = o3d.camera.PinholeCameraIntrinsic(self.wd, self.ht, fx, fy, cx, cy)

            if False:
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd,
                    camera)
                o3d.io.write_point_cloud(f"/home/thu01/code/slam/ss4_03_redkitchen_demo_data/{i}_{self.t[i]}.ply", pcd)
                print(f"written {i}({self.t[i]})")
            
            # Please uncomment below once finished.
            volume.integrate(rgbd, camera, w2c)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(dest, mesh)


def debug():
    device = 'cuda:0'

    image_paths = [
        "/data1/data/zlq_datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031470.427641.png",
        "/data1/data/zlq_datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031471.227569.png"
    ]

    images = [load_image(image_path, device) for image_path in image_paths]
    view1, view2 = images

    mast3r_ckpt = "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    m3r = AsymmetricMASt3R.from_pretrained(mast3r_ckpt).to(device)

    pred1_r, pred2_r = m3r(view1, view2)

    # view1.update(pred1_r)
    view2.update(pred2_r)

    pred1_a, pred2_a = m3r(view1, view2)

    breakpoint()

if __name__ == "__main__":
    # pose_debug()
    pass