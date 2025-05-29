# 25/3/3. Renamed from 'pre_slam' to 'slam'.
import os
from os.path import join as pjoin
import json
import yaml
from tqdm import tqdm
from time import perf_counter
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from selector import FrameSelector
from factor_graph import FactorGraph
from factor_graph import load_image as load_image_f
from selector import load_image as load_image_s
from mutils.mask_utils import binarize, save_visualization
from mutils.slam_utils import export_kitti_pose
from slamdatasets import get_dataset

@torch.no_grad()
def rgb_slam(experiment_dir, selector_dir, data_config, config):
    device = "cuda:0"

    # Data
    dataset = get_dataset(data_config)
    num_frames = len(dataset)

    outdir = experiment_dir
    visdir = pjoin(outdir, 'vis')
    loop_visdir = pjoin(outdir, 'loop_vis')
    maskdir = pjoin(outdir, 'mask')
    loopdir = pjoin(outdir, 'loop')
    os.makedirs(outdir, exist_ok=True)
    if config['save']['save_mask']:
        os.makedirs(maskdir, exist_ok=True)
        if config['loop']['do_loop']:
            os.makedirs(loopdir, exist_ok=True)
    if config['save']['save_vis']:
        os.makedirs(visdir, exist_ok=True)
        if config['loop']['do_loop']:
            os.makedirs(loop_visdir, exist_ok=True)
    
    # Factor Graph
    mast3r_ckpt = "./thirdparty/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    graph = FactorGraph(mast3r_ckpt,
                        device=device, K=dataset.K, imgsize=dataset.imgsize,
                        use_K=True, merge_pointmap=config['general']['merge_pointmap'],
                        lc_thresh_degree=config['loop']['thresh_loop_degree'],
                        filter_by_3d=config['loop']['filter_by_3d'],
                        thresh_inlier=config['loop']['thresh_inlier'],
                        use_pts_ratio=config['ba']['use_pts_ratio'],
                        use_ds=config['ba']['use_ds'])
    mask_resizer = v2.Resize(dataset.imgsize, v2.InterpolationMode.NEAREST)

    # Selector
    model, _ = FrameSelector.load_model(selector_dir)
    model.to(device)
    model.eval()

    # ----- SLAM -----

    slam_begin = perf_counter()

    # Runtime statistics
    selector_sequential_all = []
    selector_loop_all = []
    time_loading = 0.

    # Load 1st image
    tik = perf_counter()
    img_kf = load_image_s(dataset[0]['image_path'], device, size=[192, 256])
    img_kf_f = load_image_f(dataset[0]['image_path'], device)
    tok = perf_counter()
    time_loading += tok - tik

    t_kf = 0

    imgs_selected = [img_kf]
    img_fps = [dataset[0]['image_path']]
    pose_gt = [dataset[0]['pose']]
    ts_selected = [t_kf]

    t_cache, img_cf_cache, out1_cache, out2_cache = None, None, None, None

    graph.track_new_frame(img_kf_f, t=0)

    for t in tqdm(range(1, num_frames), desc='SLAM'):
        # Load image
        tik = perf_counter()
        img_cf = load_image_s(dataset[t]['image_path'], device, size=[192, 256])
        tok = perf_counter()
        time_loading += tok - tik

        tik = perf_counter()
        ret = model(img_kf, img_cf)
        out1, out2 = ret[:2]
        tok = perf_counter()
        selector_sequential_all.append( (tok - tik) )

        out1 = binarize(out1, config['general']['thresh_binarize'])
        out2 = binarize(out2, config['general']['thresh_binarize'])

        min_ovelap = min( out1.mean(), out2.mean() )

        if min_ovelap < config['general']['thresh_newframe'] or (config['general']['fix_last'] and t == num_frames - 1):
            if min_ovelap < config['general']['thresh_fail']:
                print(f'failing. t={t}, min_overlap={min_ovelap}')
                assert t_cache is not None
                t, t_cache = t_cache, t
                img_cf, img_cf_cache = img_cf_cache, img_cf
                out1, out1_cache = out1_cache, out1
                out2, out2_cache = out2_cache, out2
            else:
                t_cache, img_cf_cache, out1_cache, out2_cache = None, None, None, None

            print(f"Selecting {t}")

            # Save visualization
            if config['save']['save_vis']:
                img1, img2 = img_kf, img_cf
                dest = pjoin(visdir, f'{int(t_kf):06d}_{int(t):06d}.jpg')
                save_visualization(img1, img2, out1, out2, dest)

            # Save mask
            if config['save']['save_mask']:
                out1, out2 = out1.squeeze(0).squeeze(0).cpu().numpy(), out2.squeeze(0).squeeze(0).cpu().numpy()
                mask = np.stack( [out1, out2], 0 )
                np.save(pjoin(maskdir, f'{t_kf}_{t}.npy'), mask)

            # Track
            tik = perf_counter()
            img_cf_f = load_image_f(dataset[t]['image_path'], device)
            tok = perf_counter()
            time_loading += tok - tik

            out1, out2 = mask_resizer(out1.squeeze(0)).squeeze(0).cpu().numpy().astype(bool), mask_resizer(out2.squeeze(0)).squeeze(0).cpu().numpy().astype(bool)
            cf2kf = graph.track_new_frame(img_cf_f, [out2, out1], t=t)

            i_cur = len(imgs_selected)

            if config['loop']['do_loop']:
                # Detect loop
                loop_info_history = []
                loop_info_local = []
                for i_ref, img_ref in enumerate(imgs_selected[:-1]):
                    # Calculate all covisible masks

                    tik = perf_counter()
                    out1, out2 = model(img_ref, img_cf)
                    tok = perf_counter()
                    selector_loop_all.append(tok - tik)
                    
                    out1, out2 = binarize(out1, config['general']['thresh_binarize']), binarize(out2, config['general']['thresh_binarize'])
                    covis_score = min( out1.mean().item(), out2.mean().item() )

                    if i_ref < i_cur - config['loop']['window_local']:
                        loop_info_history.append( [covis_score, i_ref, out1, out2] )
                    else:
                        loop_info_local.append( [covis_score, i_ref, out1, out2] )

                loop_info_history.sort(key=lambda x: x[0], reverse=True)
                loop_info_local.sort(key=lambda x: x[0], reverse=True)
                loop_info = loop_info_local[:config['loop']['nums_local']] + loop_info_history[:config['loop']['nums_history']]
                loop_info = [e for e in loop_info if e[0] >= config['loop']['thresh_loop']]

                # Add loop edges
                for score, i_ref, mask_ref, mask_cf in loop_info:

                    t_ref = ts_selected[i_ref]

                    if config['save']['save_vis']:
                        img_ref = imgs_selected[i_ref]
                        dest = pjoin(loop_visdir, f'{int(t_ref):06d}_{int(t):06d}.jpg')
                        save_visualization(imgs_selected[i_ref], img_cf, mask_ref, mask_cf, dest)
                    
                    mask_ref, mask_cf = mask_resizer(mask_ref.squeeze(0)).squeeze(0).cpu().numpy().astype(bool), mask_resizer(mask_cf.squeeze(0)).squeeze(0).cpu().numpy().astype(bool)
                    print(f"Building match: {i_ref}({t_ref}) <-> {i_cur}({t})")
                    graph.build_match(i_ref, i_cur, mask=[mask_ref, mask_cf])

            # Global BA
            if (i_cur % config['ba']['ba_every'] == 0) and i_cur != 0:
                graph.run_ba()

            # Update selected frame
            imgs_selected.append(img_cf)
            img_fps.append(dataset[t]['image_path'])
            pose_gt.append(dataset[t]['pose'])
            ts_selected.append(t)
            img_kf = img_cf
            t_kf = t
        else:
            t_cache = t
            img_cf_cache = img_cf
            out1_cache, out2_cache = out1, out2

    # Global BA at last
    graph.run_ba()
    slam_end = perf_counter()
    time_slam = slam_end - slam_begin

    print(f"Selected {len(ts_selected)} frames.")

    # Report runtime
    infer_time_avg = np.array(selector_sequential_all[3:]).mean()
    time_selector_tracking = sum(selector_sequential_all, 0.)
    time_selector_loop = sum(selector_loop_all, 0.)
    time_other = time_slam - (time_selector_tracking +
                               graph.time_enc_tracking +
                               graph.time_dec_tracking +
                               graph.time_match_tracking +
                               graph.time_pnp_tracking +
                               time_selector_loop +
                               graph.time_enc_loop +
                               graph.time_dec_loop +
                               graph.time_match_loop +
                               graph.time_pnp_loop +
                               graph.time_ba +
                               time_loading )
    print(f"time_selector_tracking: {time_selector_tracking:.3f} s")
    print(f"time_encoder_tracking: {graph.time_enc_tracking:.3f} s")
    print(f"time_decoder_tracking: {graph.time_dec_tracking:.3f} s")
    print(f"time_match_tracking: {graph.time_match_tracking:.3f} s")
    print(f"time_pnp_tracking: {graph.time_pnp_tracking:.3f} s")
    print(f"time_selector_loop: {time_selector_loop:.3f} s")
    print(f"time_encoder_loop: {graph.time_enc_loop:.3f} s")
    print(f"time_decoder_loop: {graph.time_dec_loop:.3f} s")
    print(f"time_match_loop: {graph.time_match_loop:.3f} s")
    print(f"time_pnp_loopt: {graph.time_pnp_loop:.3f} s")
    print(f"time_ba: {graph.time_ba:.3f} s")
    print(f"time_loading: {time_loading:.3f} s")
    print(f"total time: {time_slam:.3f} s")
    print(f"other time: {time_other:.3f} s")
    print(f"total time w/o loading: {time_slam - time_loading} s")

    print(f"Average inference time: {infer_time_avg.item() * 1e3:.3f} ms")

    with open(pjoin(outdir, 'selected_ts.txt'), 'w') as f:
        f.writelines( [f'{t}\n' for t in ts_selected] )
    with open(pjoin(outdir, 'selected_imgs.txt'), 'w') as f:
        f.writelines( [fp + '\n' for fp in img_fps] )
    export_kitti_pose(pose_gt, pjoin(outdir, 'gt_pose.txt'))
    graph.export_kitti_fmt(pjoin(outdir, f'svslam_pose.txt'))

if __name__ == "__main__":
    selector_dir = 'selector'

    parser = argparse.ArgumentParser()
    parser.add_argument('--expdir', type=str, help="e.g. 'demo'. Outputs will be written to './experiments/demo'.")
    parser.add_argument('--data_config', type=str, default='./config/data/7scenes.yaml', help='data config')
    parser.add_argument('--slam_config', type=str, default='./config/slam/7scenes.yaml', help='SLAM config')

    args = parser.parse_args()

    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.Loader)
    with open(args.slam_config, 'r') as f:
        slam_config = yaml.load(f, Loader=yaml.Loader)

    print(slam_config)

    experiment_dir = pjoin('experiments', args.expdir)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(pjoin(experiment_dir, 'data_config.yaml'), 'w') as f:
        yaml.dump(data_config, f)
    with open(pjoin(experiment_dir, 'slam_config.yaml'), 'w') as f:
        yaml.dump(slam_config, f)

    rgb_slam(experiment_dir, selector_dir, data_config, slam_config)