
import numpy as np
#np.random.seed(0)
from torch import nn
from opt import get_opts
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
#torch.manual_seed(0)
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
from ngp_config import *
from flip_loss import HDRFLIPLoss
from torchmetrics import MeanAbsoluteError
from pytorch_lightning.callbacks import LearningRateMonitor
# --lr 3e-5
# --optimize_ext
#--weight_path /home/ubuntu/repos/ngp_pl/ckpts/colmap/vid_train/epoch=29_slim.ckpt
# --root_dir /home/ubuntu/repos/vid_ds/flame/ --dataset_name colmap --exp_name flame   --num_gpus 1  --num_epochs 40 --downsample 0.5 --scale 2.0 --batch_size 6000  --lr 1e-6
#--root_dir /home/ubuntu/repos/instant-ngp-flame/ --dataset_name colmap --exp_name flame   --num_gpus 1  --num_epochs 40 --downsample 0.5 --scale 2.0 --batch_size 6000  --lr 1e-6
# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.rgb_pred = {}
        self.batch_ids = []
        self.rgb_gt = {}
        self.pix_ids = {}
        self.flip_loss = HDRFLIPLoss()
        #self.warmup_steps = 256
        self.warmup_steps = 999999999999999999999999999999999999999999
        #self.warmup_steps = 5
        self.update_interval = DENSITY_GRID_UPDATE_INTERVAL

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.l1 = MeanAbsoluteError().to(self.device)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        g = 2

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            if BATCHED_EVAL:
                directions = self.directions[batch['pix_idxs']]
            else:
                directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, batch['frames'], **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        #return opts, [net_sch]
        return opts

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=WORKERS_CNT,
                          #persistent_workers=WORKERS_CNT > 0,
                          batch_size=None,
                          #drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=0,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(batch['frames'],
                                           0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        #torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        logs = {}
        #print('666666666666666666666')
        #print(self.trainer.root_gpu)
        if self.trainer.root_gpu > - 999999999999:
            #print('GPU 0:')
            rgb_gt = batch['rgb'][0, ...]
            #batch['rgb'] = None
            #torch.cuda.empty_cache()
            results = self(batch, split='test')
            g =2
            if BATCHED_EVAL:
                img_id = batch_nb // VAL_BATCHES_PER_IMG
                batch_id = batch_nb % VAL_BATCHES_PER_IMG
                if not img_id in self.rgb_gt:
                    self.rgb_gt[img_id] = []

                self.rgb_gt[img_id].append(rgb_gt)

                if not img_id in self.rgb_pred:
                    self.rgb_pred[img_id] = []

                if not img_id in self.pix_ids:
                    self.pix_ids[img_id] = []

                self.rgb_pred[img_id].append(results['rgb'])
                self.batch_ids.append(batch_id)
                self.pix_ids[img_id].append(batch['pix_idxs'])
            else:
                # compute each metric per image
                self.val_psnr(results['rgb'], rgb_gt)
                logs['psnr'] = self.val_psnr.compute()
                self.val_psnr.reset()

                w, h = self.train_dataset.img_wh
                rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
                rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h).float()
                self.val_ssim(rgb_pred, rgb_gt)
                logs['ssim'] = self.val_ssim.compute()
                self.val_ssim.reset()
                logs['flip'] = self.flip_loss(rgb_pred, rgb_gt)
                logs['mae'] = self.l1(rgb_pred, rgb_gt)
                if self.hparams.eval_lpips:
                    self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                                   torch.clip(rgb_gt*2-1, -1, 1))
                    logs['lpips'] = self.val_lpips.compute()
                    self.val_lpips.reset()

                if not self.hparams.no_save_test: # save test image to disk
                    idx = batch['img_idxs']
                    rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                    rgb_pred = (rgb_pred*255).astype(np.uint8)
                    depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                    rank = os.environ.get('LOCAL_RANK')
                    if rank is not None:
                        nodeId = int(rank)
                    else:
                        nodeId = 0
                    nodeId = str(nodeId)
                    kn = "new_generated_img_" + str(nodeId)
                    self.logger.log_image(key=kn, images=[rgb_pred])
                    imageio.imsave(os.path.join(self.val_dir, str(idx) + nodeId + '.png'), rgb_pred)
                    self.logger.log_image(key="new_generated_depth_" + nodeId, images=[depth])
                    imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        if self.trainer.root_gpu > -19999999999999:
            if BATCHED_EVAL:
                import torchvision
                IMG_ID = 0
                rgb_gt = torch.cat(self.rgb_gt[IMG_ID])
                self.rgb_gt[IMG_ID].clear()
                self.rgb_gt = {}
                self.pix_ids = {}
                self.batch_ids.clear()
                #rgb_gt = all_gather_ddp_if_available(rgb_gt)
                rgb_pred = torch.cat(self.rgb_pred[IMG_ID])
                #rgb_pred = all_gather_ddp_if_available(rgb_pred)
                self.rgb_pred[IMG_ID].clear()
                self.rgb_pred = {}
                #pix_ids = all_gather_ddp_if_available(torch.cat(self.pix_ids[IMG_ID]))
                psnr = self.val_psnr(rgb_pred, rgb_gt)
                self.log('test/psnr', psnr, True)
                self.val_psnr.reset()
                w, h = self.train_dataset.img_wh
                rgb_pred = rearrange(rgb_pred, '(h w) c -> 1 c h w', h=h)
                assert rgb_pred.shape[-1] == IM_W
                rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
                mean_ssim = self.val_ssim(rgb_pred, rgb_gt)
                print('SSIM:::::::::::::: ', mean_ssim)
                self.log('test/ssim', mean_ssim)
                # lpipss =  self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                #                    torch.clip(rgb_gt*2-1, -1, 1))
                # self.log('test/lpips_vgg', lpipss)

                im = np.array(torchvision.transforms.ToPILImage()(rgb_pred[0, ...]))
                #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                imageio.imsave(os.path.join(self.val_dir, f'{0:03d}.png'), im)
                #self.logger.log_image(key="samples", images=[im])

            else:
                psnrs = torch.stack([x['psnr'] for x in outputs])
                mean_psnr = all_gather_ddp_if_available(psnrs).mean()
                self.log('test/psnr', mean_psnr, True)
                print('PSNR: ', mean_psnr)

                ssims = torch.stack([x['ssim'] for x in outputs])
                mean_ssim = all_gather_ddp_if_available(ssims).mean()
                print('SSIM:::::::::::::: ', mean_ssim)
                self.log('test/ssim', mean_ssim)

                flips = torch.stack([x['flip'] for x in outputs])
                mean_ssim = all_gather_ddp_if_available(flips).mean()
                self.log('test/flip', mean_ssim)

                maes = torch.stack([x['mae'] for x in outputs])
                mean_maes = all_gather_ddp_if_available(maes).mean()
                self.log('test/mae', mean_maes)

                if self.hparams.eval_lpips:
                    lpipss = torch.stack([x['lpips'] for x in outputs])
                    mean_lpips = all_gather_ddp_if_available(lpipss).mean()
                    self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()

    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=1,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1), lr_monitor]

    # logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
    #                            name=hparams.exp_name,
    #                            default_hp_metric=False)
    logger = WandbLogger(save_dir=f"logs/{hparams.dataset_name}",
                                project =  PROJECT_NAME,
                               name = hparams.exp_name)


    trainer = Trainer(max_epochs=hparams.num_epochs,
                      #replace_sampler_ddp=False,
                      #val_check_interval=10, # steps
                      accumulate_grad_batches=ACCUM_BATCHES,
                      check_val_every_n_epoch=1,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      #devices=1,
                      #devices=hparams.num_gpus,
                      devices=[int(hparams.num_gpus)],
                      #strategy='ddp_find_unused_parameters_false',
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)
    # trainer = Trainer(max_epochs=hparams.num_epochs,
    #                   val_check_interval=1, # steps
    #                   gpus=-1,
    #                   #check_val_every_n_epoch=100,
    #                   callbacks=callbacks,
    #                   logger=logger,
    #                   enable_model_summary=False,
    #                   accelerator='ddp',
    #                   devices=hparams.num_gpus,
    #                   num_sanity_val_steps=-1 if hparams.val_only else 0,
    #                   precision=16)
    #trainer.validate(system)
    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)