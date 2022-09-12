import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import pathlib
import cv2

from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from .base import BaseDataset


class NerfMpyDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        poses_arr = np.load(os.path.join(root_dir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])#.transpose([1, 2, 0])
        intr = poses[..., -1]
        bds = poses_arr[:, -2:].transpose([1, 0])
        self.downsample = 0.5

        self.read_intrinsics(intr)
        if kwargs.get('read_meta', True):
            self.read_meta(split, poses[..., :-1], bds, **kwargs)

    def read_intrinsics(self, intr):
        # Step 1: read and scale intrinsics (same for all images)
        #camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(intr[0, 0] * self.downsample)
        w = int(intr[0, 1] * self.downsample)
        self.img_wh = (w, h)

        focal_length = intr[0, 2] * self.downsample
        self.K = torch.FloatTensor([[focal_length, 0, intr[0, 1] / 2],
                                    [0, focal_length, intr[0, 0] / 2],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, poses_inp, bd_inp, **kwargs):
        img_paths = sorted(list(pathlib.Path(self.root_dir).glob('*.mp4')))
        assert poses_inp.shape[0] == len(img_paths)
        #imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        #img_names = [imdata[k].name for k in imdata]
        #perm = np.argsort(img_names)
        # folder = 'images'
        # # read successfully reconstructed images and ignore others
        # img_paths = [os.path.join(self.root_dir, folder, name)
        #              for name in sorted(img_names)]
        poses = poses_inp.copy()
        poses[..., 0] = poses_inp[..., 1]
        poses[...,  1] = poses_inp[..., 0]
        poses[...,  2] = -poses_inp[..., 2]

        self.poses = center_poses(poses, None)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        #self.pts3d /= scale

        self.rays = []
        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        # use  10th image as test set
        if split == 'train':
            img_paths = [x for i, x in enumerate(img_paths) if i != 0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i != 10])
        elif split == 'test':
            img_paths = [x for i, x in enumerate(img_paths) if i == 0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i == 10])

        print(f'Loading {len(img_paths)} {split} images ...')

        for img_path in tqdm(img_paths):
            buf = [] # buffer for ray attributes: rgb, etc
            cap = cv2.VideoCapture(str(img_path))
            flag, frame = cap.read()
            frame = frame.astype(np.float32)/255.0
            frame = cv2.resize(frame, self.img_wh)
            frame = rearrange(frame, 'h w c -> (h w) c')
            #img = read_image(img_path, self.img_wh, blend_a=False)
            frame = torch.FloatTensor(frame)
            buf += [frame]
            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)