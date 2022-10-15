import random
from torch.utils.data import Dataset
import numpy as np
from ngp_config import *
import torch

class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.imgs_ids = []
        self.pix_ids = []
        self.img_ids = []
        self.frames_to_use = []
        self.rays = []
        self.sample_id = 0

    def read_intrinsics(self):
        raise NotImplementedError

    def read_meta(self, split, **kwargs):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return self.pix_ids.shape[1] // self.batch_size
        else:
            if BATCHED_EVAL:
                 return len(self.poses) * VAL_BATCHES_PER_IMG
            else:
                return len(self.poses)



    def __getitem__(self, idx):
        if self.split.startswith('train'):
            #if random.random() < FRAME_CHANGE_PROB:
            if self.sample_id >= BATCHES_FOR_RELOAD:
                self.read_meta(self.split)
                self.sample_id = 0
            else:
                self.sample_id += 1

            start_index = idx * self.batch_size
            #adjusted_batch_size = self.batch_size // self.poses
            # training pose is retrieved in train.py
            img_idxs = self.img_ids[:, start_index: (start_index + self.batch_size)]
            #  select pixels
            pix_idxs = self.pix_ids[:, start_index: (start_index + self.batch_size)]
            rays = self.rays[:, start_index: (start_index + self.batch_size)]
            frames = self.frames_to_use[:, start_index: (start_index + self.batch_size)]

            sample = {'img_idxs': torch.flatten(img_idxs), 'pix_idxs': torch.flatten(pix_idxs),
                      'rgb': rays.reshape((-1, 3)), 'frames' : torch.flatten(frames)}
            # if self.rays.shape[-1] == 4: # HDR-NeRF data
            #     sample['exposure'] = rays[:, 3:]


            #.sample_id += 1
        else:
            if not BATCHED_EVAL:
                sample = {'pose': self.poses[idx], 'img_idxs': idx, 'frames' : [self.frame_to_use] *  self.rays[idx].shape[0]}
                if len(self.rays)>0: # if ground truth available
                    rays = self.rays[idx]
                    sample['rgb'] = rays[:, :3]
                    if rays.shape[1] == 4: # HDR-NeRF data
                        sample['exposure'] = rays[0, 3] # same exposure for all rays
            else:
                img_id = idx // VAL_BATCHES_PER_IMG
                batch_id = idx % VAL_BATCHES_PER_IMG
                sample = {'pose': self.poses[img_id], 'img_idxs': img_id, 'frames' : [self.frame_to_use] *  self.rays[idx].shape[0]}
                if len(self.rays)>0: # if ground truth available
                    startRay = batch_id * VAL_BATCH_SIZE
                    endRay = min(startRay + VAL_BATCH_SIZE, RAYS_CNT)
                    rays = self.rays[img_id, startRay : endRay, :]
                    sample['rgb'] = rays[:, :3]
                    if rays.shape[1] == 4: # HDR-NeRF data
                        sample['exposure'] = rays[0, 3] # same exposure for all rays
                    pix_idxs = np.arange(startRay, endRay)
                    sample['pix_idxs'] = pix_idxs

        return sample