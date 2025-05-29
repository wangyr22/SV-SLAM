# 25/1/30. Datasets. wyr
import os
from os.path import join as pjoin
import json
import glob
from natsort import natsorted

import numpy as np

def get_dataset(config):
    if config['dataset'] == 'simple':
        return SimpleDataset(**config['dataset_config'])
    if config['dataset'] == '7scenes':
        return SScenesDataset(**config['dataset_config'])
    if config['dataset'] == 'tum':
        return TUMDataset(**config['dataset_config'])
    if config['dataset'] == 'scannet':
        return ScanNetDataset(**config['dataset_config'])
    if config['dataset'] == 'scannetpp':
        return ScanNetppDataset(**config['dataset_config'])

class SimpleDataset:
    def __init__(self, imgdir, fx, fy, cx, cy, num_frames=-1):
        self.image_dir = imgdir
        self.image_paths = natsorted(os.listdir(imgdir))
        self.num_frames = len(self.image_paths)
        if num_frames > 0:
            self.num_frames = min(self.num_frames, num_frames)

        self.get_size_ands_K(fx, fy, cx, cy)

    def get_size_and_K(self, fx, fy, cx, cy):
        from factor_graph import load_image
        import cv2

        img_path = pjoin(self.image_dir, self.image_paths[0])
        _, _, ht_tgt, wd_tgt = load_image( img_path, 'cpu' )['img']

        ht_src, wd_src, _ = cv2.imread(img_path)

        x_ratio = wd_tgt / wd_src
        y_ratio = ht_tgt / ht_src
        fx = fx * x_ratio
        cx = cx * x_ratio
        fy = fy * y_ratio
        cy = cy * y_ratio

        self._K = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1.]]
        )
        self.imgsize = [ht_tgt, wd_tgt]

    @property
    def K(self):
        return self._K
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, t):
        return {
            'image_path': pjoin(self.image_dir, self.image_paths[t])
        }

class TUMDataset:
    name = 'tum'
    imgsize = [384, 512]

    all_seqs = ['freiburg1_360', "freiburg1_desk", "freiburg1_desk2", 'freiburg1_floor', 'freiburg1_plant',
                 "freiburg1_room", 'freiburg1_rpy', 'freiburg1_teddy', 'freiburg1_xyz', "freiburg2_xyz", "freiburg3_long_office_household"]
    # TODO: make editable
    img_path_fmt = "{}/rgbd_dataset_{}/rgb/{}.png"
    depth_path_fmt = "{}/rgbd_dataset_{}/depth/{}.png"
    association_file_fmt = "config/tum_meta/associations/idx_{}.txt"

    camera_params = {
        'freiburg1': {
            'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3
        },
        'freiburg2': {
            'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7
        },
        'freiburg3': {
            'fx': 535.4, 'fy': 539.2, 'cx': 320.1, 'cy': 247.6
        }
    }


    def __init__(self, seq: int | str, data_root):
        if isinstance(seq, int):
            seq = TUMDataset.all_seqs[seq]
        else:
            assert seq in TUMDataset.all_seqs
        self.seq = seq
        self.data_root = data_root

        cam_params = TUMDataset.camera_params[seq.split('_')[0]]
        self.original_params = cam_params

        # intrinsics for images resized to 512x384
        self._K = self.get_scaled_intrinsics(cam_params)

        # gt pose
        c2w_all = np.loadtxt(f"./config/tum_meta/{seq}/valid_pose.txt")
        num_frames, _ = c2w_all.shape
        c2w_all = c2w_all.reshape(num_frames, 4, 4)
        self.pose_gt = c2w_all
        self.num_frames = num_frames

        # Load associations
        with open(TUMDataset.association_file_fmt.format(seq), 'r') as f:
            lines_association = f.readlines()
            lines_association = lines_association[2:]

        # Get all image paths & depth paths. TUM timestamps are a bit complex,
        # so we pre-process them altogether. Depths are used for a few tests.
        image_paths = []
        depth_paths = []
        for t in range(num_frames):
            # Get real tstamps: idx image depth pose
            assert t == int(lines_association[t].split()[0])
            t_real = lines_association[t].split()[1]
            t_real_depth = lines_association[t].split()[2]
            image_paths.append(TUMDataset.img_path_fmt.format(self.data_root, seq, t_real))
            depth_paths.append(TUMDataset.depth_path_fmt.format(self.data_root, seq, t_real_depth))

        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.png_depth_scale = 5000.

    # re-scaled. 640x480 --> 512x384
    def get_scaled_intrinsics(self, d):
        x_ratio = 512. / 640
        y_ratio = 384. / 480
        fx = d['fx'] * x_ratio
        cx = d['cx'] * x_ratio
        fy = d['fy'] * y_ratio
        cy = d['cy'] * y_ratio
        return np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1.]]
        )

    @property
    def K(self):
        return self._K
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, t):
        # return self.image_paths[t], self.pose_gt[t]
        return {
            'image_path': self.image_paths[t],
            'depth_path': self.depth_paths[t],
            'pose': self.pose_gt[t]
        }
    

class ReplicaDataset:
    name = 'replica'
    imgsize = [288, 512]

    all_seqs = ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2']
    img_path_fmt = "{}/{}/results/frame{:06d}.jpg"
    depth_path_fmt = "{}/{}/results/depth{:06d}.png"
    pose_file_fmt = "{}/{}/traj.txt"
    
    def __init__(self, seq: str | int, data_root):
        if isinstance(seq, int):
            seq = ReplicaDataset.all_seqs[seq]
        else:
            assert seq in ReplicaDataset.all_seqs
        self.seq = seq
        self.data_root = data_root

        c2w_all = np.loadtxt(ReplicaDataset.pose_file_fmt.format(self.data_root, seq))
        num_frames, _ = c2w_all.shape
        c2w_all = c2w_all.reshape(num_frames, 4, 4)
        self.pose_gt = c2w_all
        self.num_frames = num_frames

        # ht, wd, fx, fy, cx, cy
        self.original_params = {"fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5}

        # 1200x680 --> 512x288
        self._K = np.array(
            [[256.0, 0, 255.7866666666667],
             [0, 254.11764705882354, 143.78823529411764],
             [0, 0, 1.]]
             )

        self.png_depth_scale = 6553.5

    @property
    def K(self):
        return self._K

    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, t):
        return {
            'image_path': ReplicaDataset.img_path_fmt.format(self.data_root, self.seq, t),
            'depth_path': ReplicaDataset.depth_path_fmt.format(self.data_root, self.seq, t),
            'pose': self.pose_gt[t]
        }
    

class ScanNetDataset:
    name = 'scannet'
    imgsize = [368, 512]

    all_seqs = ["scene0000_00", "scene0025_02", "scene0059_00", "scene0062_00", "scene0103_00", "scene0106_00", "scene0126_00", "scene0169_00", "scene0181_00", "scene0207_00"]
    img_path_fmt = "{}/{}/color/{}.jpg"
    pose_path_fmt = "{}/{}/pose/{}.txt"
    
    def __init__(self, seq: str | int, data_root):
        if isinstance(seq, int):
            seq = ScanNetDataset.all_seqs[seq]
        else:
            assert seq in ReplicaDataset.all_seqs
        self.seq = seq
        self.data_root = data_root

        self._K = np.array(
            [[462.072530962963, 0, 255.32643713580245],
             [0, 443.6928490743802, 186.25325183471074],
             [0, 0, 1.]]
             )
        
        num_frames = len(
            glob.glob(ScanNetDataset.img_path_fmt.format(self.data_root, seq, '*'))
        )
        self.num_frames = num_frames
        
    @property
    def K(self):
        return self._K
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, t):
        pose = np.loadtxt(
            self.pose_path_fmt.format(self.data_root, self.seq, t)
        )
        return {
            'image_path': self.img_path_fmt.format(self.data_root, self.seq, t),
            'pose': pose
        }
    
class SScenesDataset:
    name = '7-scenes'
    imgsize = [384, 512]

    # Like NICER-SLAM, we use seq1 for every scene.
    all_seqs = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    img_path_fmt = "{}/{}/seq-01/frame-{:06d}.color.png"
    __img_path_fmt = "{}/{}/seq-01/frame-{}.color.png"
    pose_path_fmt = "{}/{}/seq-01/frame-{:06d}.pose.txt"

    # TODO
    # Data from NICER-SLAM; intrinsics are obtained by COLMAP.
    intrinsic_data = {
        "chess": {"fx": 535.30153598, "fy": 533.71239636, "cx": 316.85634818, "cy": 239.75744442},
        "fire": {"fx": 534.60449776, "fy": 539.02904297, "cx": 318.09034465, "cy": 248.36314533},
        "heads": {"fx": 533.48533767, "fy": 534.03326847, "cx": 315.07657519, "cy": 238.83690698},
        "office": {"fx": 534.924901, "fy": 549.31688003, "cx": 316.52655936, "cy": 256.39520434},
        "pumpkin": {"fx": 569.2724576, "fy": 544.82942106, "cx": 346.65669988, "cy": 221.8028837},
        "redkitchen": {"fx": 540.26264666, "fy": 545.1689031, "cx": 318.22221602, "cy": 246.72672228},
        "stairs": {"fx": 571.97464398, "fy": 570.18232961, "cx": 326.44024801, "cy": 238.53590499},
    }
    
    def __init__(self, seq: str | int, data_root):
        if isinstance(seq, int):
            seq = SScenesDataset.all_seqs[seq]
        else:
            assert seq in SScenesDataset.all_seqs
        self.seq = seq
        self.data_root = data_root

        self.original_params = SScenesDataset.intrinsic_data[seq]
        self._K = self.get_scaled_intrinsics( SScenesDataset.intrinsic_data[seq] )

        num_frames = len(
            glob.glob(SScenesDataset.__img_path_fmt.format(self.data_root, seq, '*'))
        )
        self.num_frames = num_frames

    # re-scaled. 640x480 --> 512x384
    def get_scaled_intrinsics(self, d):
        x_ratio = 512. / 640
        y_ratio = 384. / 480
        fx = d['fx'] * x_ratio
        cx = d['cx'] * x_ratio
        fy = d['fy'] * y_ratio
        cy = d['cy'] * y_ratio
        return np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1.]]
        )

    @property
    def K(self):
        return self._K
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, t):
        pose = np.loadtxt(
            self.pose_path_fmt.format(self.data_root, self.seq, t)
        )
        return {
            'image_path': self.img_path_fmt.format(self.data_root, self.seq, t),
            'pose': pose
        }

class ScanNetppDataset:
    name = 'scannetpp'
    # 1752x1168 --> 512x336
    imgsize = [336, 512]

    all_seqs = ["b20a261fdf", "8b5caf3398", "fb05e13ad1", "2e74812d00", "281bc17764"]
    img_path_fmt = "{}/{}/dslr/undistorted_images/{}"
    cam_metadata_fmt = "{}/{}/dslr/nerfstudio/transforms_undistorted.json"
    train_test_split_fmt = "{}/{}/dslr/train_test_lists.json"

    def __init__(self, seq: str | int, data_root):
        if isinstance(seq, int):
            seq = ScanNetppDataset.all_seqs[seq]
        else:
            assert seq in ScanNetppDataset.all_seqs
        self.seq = seq
        self.data_root = data_root

        # Use train-test split, as SplaTAM does.
        with open(ScanNetppDataset.train_test_split_fmt.format(self.data_root, seq), 'r') as f:
            self.train_test_split = json.load(f)

        # Following LoopSplat, we only use first 250 frames.
        self.img_names = self.train_test_split['train'][:250]
        self.num_frames = len(self.img_names)

        with open(ScanNetppDataset.cam_metadata_fmt.format(self.data_root, seq), 'r') as f:
            cam_metadata = json.load(f)

        self.original_params = {
            'fx': cam_metadata['fl_x'], 'fy': cam_metadata['fl_y'], 'cx': cam_metadata['cx'], 'cy': cam_metadata['cy']
        }
        self._K = self.get_scaled_intrinsics(cam_metadata)

        self.frames_metadata = cam_metadata['frames']
        self.name2idx = {
            frame['file_path']: idx for idx, frame in enumerate(self.frames_metadata)
        }

        self._P = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        )
        
    def get_scaled_intrinsics(self, d):
        # 1752x1168 --> 512x336
        x_ratio = 512. / 1752
        y_ratio = 336. / 1168
        fx = d['fl_x'] * x_ratio
        cx = d['cx'] * x_ratio
        fy = d['fl_y'] * y_ratio
        cy = d['cy'] * y_ratio
        return np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1.]]
        )

    @property
    def K(self):
        return self._K
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, t):
        img_name = self.img_names[t]
        frame_metadata = self.frames_metadata[ self.name2idx[img_name] ]
        assert frame_metadata['file_path'] == img_name
        c2w = np.array(frame_metadata["transform_matrix"])
        pose = self._P @ c2w @ self._P.T
        return {
            'image_path': self.img_path_fmt.format(self.data_root, self.seq, img_name),
            'pose': pose,
            'is_bad': frame_metadata['is_bad']
        }

if __name__ == "__main__":
    dataset = ScanNetDataset(0)
    print(len(dataset))
    print(dataset.K)

    for data in dataset:
        print(data)
        break