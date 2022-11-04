import random
import torch
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ngp_config import  *
import pathlib
import imageio
import cv2
from timeit import default_timer as timer
from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from .base import BaseDataset

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    import imageio
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs

class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()
        #from nerf_load_llff import load_llff_data
        #images, poses, bds, render_poses, i_test = load_llff_data(root_dir, factor = None, bd_factor = 1.)
        #poses, bds = _load_data(root_dir, load_imgs = False) # Remove
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)
        assert w == IM_W
        assert h == IM_H

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        self.imgs_ids = []
        self.pix_ids = []
        self.img_ids = []
        self.frames_to_use = []
        self.rays = []

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        #vid_paths = sorted(list(pathlib.Path(VID_DIR).glob('*.mp4')))

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        if split == 'train':
            img_paths = [x for i, x in enumerate(img_paths) if i != TEST_VIEW_ID]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i != TEST_VIEW_ID])
        elif split == 'test':
            img_paths = [x for i, x in enumerate(img_paths) if i == TEST_VIEW_ID]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i == TEST_VIEW_ID])
        start = timer()
        print(f'Loading {len(img_paths)} {split} images ...')
        #frame_to_use = random.randint(MIN_FRAME, MAX_FRAME)
        #frame_to_use = 0
        #frames_to_use = []
        self.rays = torch.zeros((MAX_FRAME, len(img_paths), RAYS_CNT,  3), dtype = torch.uint8)
        if split == 'train':
            for frame_to_use in tqdm(range(MIN_FRAME, MAX_FRAME)):
                img_id = 0
                rays_per_camera = []
                pix_ids_per_camera = []

                for img_path in img_paths:
                    fname = os.path.basename(img_path).split('.')[0].zfill(2)
                    vid_name = 'cam' + fname + '.mp4'
                    vid_path = os.path.join(VID_DIR, vid_name)
                    aten_map = np.load(os.path.join(ATEN_FOLDER, 'cam' + fname + '.npy')).astype(np.float32)
                    aten_map /= aten_map.sum()
                    aten_map = aten_map.flatten()
                    t_dir = os.path.join(TEMP_DIR, str(frame_to_use))
                    t_path = os.path.join(t_dir, fname)  + '.jpg'
                    if not os.path.exists(t_dir):
                        os.mkdir(t_dir)

                    if not os.path.exists(t_path):
                        cap = cv2.VideoCapture(str(vid_path)) # 30.0 fps 300 frames count = 10 sec
                        cap.set(1, frame_to_use)
                        flag, frame = cap.read()
                        frame = cv2.resize(frame, self.img_wh)
                        cv2.imwrite(t_path, frame)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame = np.array(Image.open(t_path))

                    frame_to_use_formated = (frame_to_use - MIN_FRAME) / MAX_FRAME
                    frame = rearrange(frame, 'h w c -> (h w) c')
                    # #----------
                    # if random.random() < UNIFORM_SMPL_PROB:
                    #     pix_idxs = np.random.choice(RAYS_CNT, SAMPLE_RAYS_PER_FRAME)
                    # else:
                    #     pix_idxs = np.random.choice(RAYS_CNT, SAMPLE_RAYS_PER_FRAME, p=aten_map)
                    # rays = frame[pix_idxs]
                    # rays = rays.astype(np.float16) / 255.0
                    # #-------------------
                    self.rays[frame_to_use, img_id, ...] = torch.tensor(frame)
                    #rays = frame

                    #img = frame.astype(np.float16) / 255.0
                    #img = imageio.imread(img_path).astype(np.float32)/255.0

                    #img = read_image(img_path, self.img_wh, blend_a=False)
                    #rays = torch.FloatTensor(rays)
                    #img = torch.ShortTensor(img)
                    #img = torch.ByteTensor(img)
                    #buf += [img]

                    #rays_per_camera.append(rays)
                    #pix_ids_per_camera.append(pix_idxs)
                    #frames_to_use.extend([frame_to_use] * len(pix_idxs))

                    #self.rays += torch.tensor(rays_per_camera)
                    #rays_per_camera = rearrange(rays_per_camera, 'f r c -> (f r) c')
                    #self.rays.append(rays)
                    #self.pix_ids.append(pix_idxs)
                    #self.img_ids.append(img_id)
                    img_id += 1
            #self.frames_to_use.append(frame_to_use_formated)


            # self.rays = torch.tensor(self.rays).reshape((MAX_FRAME, len(img_paths), -1,  3)) # (frame_id, n_images, SAMPLE_RAYS_PER_FRAME, 3)
            # self.pix_ids = torch.tensor(self.pix_ids).reshape((MAX_FRAME, len(img_paths), -1)) # (frame_id, n_images, SAMPLE_RAYS_PER_FRAM)
            # self.img_ids = torch.tensor(self.img_ids)
        #self.frames_to_use = torch.stack(self.frames_to_use)

        # indexes = torch.randperm(self.pix_ids.shape[1])
        # self.rays = self.rays[:, indexes, :]
        # self.pix_ids = self.pix_ids[:, indexes]
        # self.img_ids = self.img_ids[:, indexes]
        # self.frames_to_use = self.frames_to_use[:, indexes]

        if split == 'test':
            rank = os.environ.get('LOCAL_RANK')
            if rank is not None:
                nodeId = int(rank)
            else:
                nodeId = 0
            frame_to_use = nodeId * 5
            frame_to_use_formated = (frame_to_use - MIN_FRAME) / MAX_FRAME
            for img_path in img_paths:
                fname = os.path.basename(img_path).split('.')[0].zfill(2)
                t_dir = os.path.join(TEMP_DIR, str(frame_to_use))
                t_path = os.path.join(t_dir, fname) + '.jpg'
                frame = np.array(Image.open(t_path))
                rays = frame
                rays = rays.astype(np.float16) / 255.0
                self.rays = torch.tensor(rays).reshape((1, 1, -1,  3)) # (frame_id, n_images, SAMPLE_RAYS_PER_FRAME, 3)
                #self.pix_ids = torch.tensor(self.pix_ids).reshape((MAX_FRAME, len(img_paths), -1)) # (frame_id, n_images, SAMPLE_RAYS_PER_FRAM)
                self.img_ids = 0
            self.frames_to_use = frame_to_use_formated

        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        end = timer()
        print('ELAPSED: ', end - start)  # Time in seconds, e.g. 5.38091952400282
