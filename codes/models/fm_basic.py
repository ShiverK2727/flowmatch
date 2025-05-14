# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from abc import abstractmethod

import torch
import math
from os import makedirs

import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
import wandb

from models.ema import EMACallback
from utils.logger import log_info


#################################################################################
#                          Core Flow Matching Model                             #
#################################################################################


class BaseFlowMatching(pl.LightningModule):
    """
    Flow Matching模型基类, 支持VAE和非VAE两种模式
    """

    def __init__(
            self,
            in_channels=4,
            class_dropout_prob=0.1,
            num_classes=1,
            learning_rate=0.0001,
            weight_decay=0.1,
            learn_sigma=True,
            latent_shape=None,
            training_type="shortcut",
            vae_model_path=None,  # 如果为None则不使用VAE
            image_save_path="log_images3",
            denoise_timesteps=[1, 2, 4, 8, 16, 32, 128],
            denoise_timesteps_target=128,
            bootstrap_every=8,
            bootstrap_dt_bias=0,
            bootstrap_cfg=False,
            cfg_scale=0.0,
            bootstrap_ema=False,
            eval_size=8,
            force_t=-1,
            force_dt=-1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.training_type = training_type
        self.eval_size = eval_size
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.vae_model_path = vae_model_path
        self.image_save_path = image_save_path

        self.denoise_timesteps_target = denoise_timesteps_target
        self.denoise_timesteps = denoise_timesteps
        self.bootstrap_every = bootstrap_every
        self.bootstrap_dt_bias = bootstrap_dt_bias
        self.bootstrap_cfg = bootstrap_cfg
        self.cfg_scale = cfg_scale
        self.bootstrap_ema = bootstrap_ema
        self.class_dropout_prob = class_dropout_prob

        self.force_t = force_t
        self.force_dt = force_dt

        self.latent_shape = latent_shape

        # 初始化VAE（如果指定了vae_model_path）
        self.vae = None
        if self.vae_model_path is not None:
            self.vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(self.device)
            self.vae = self.vae.eval()
            self.vae.requires_grad_(False)
            if self.latent_shape is not None:
                self.eps = torch.randn(self.latent_shape).to(self.device)

        self.fids = None
        self.validation_step_outputs = []

        makedirs(self.image_save_path, exist_ok=True)
        self._validate_config()
        log_info(f"FlowMatching initialized with config: {self.hparams}", print_message=True)

    def on_fit_start(self):
        self.fids = [FrechetInceptionDistance().to(self.device) for _ in range(len(self.denoise_timesteps))]

    def _validate_config(self):
        assert self.training_type in ["naive", "shortcut"], f"Invalid training_type: {self.training_type}"
        assert self.denoise_timesteps_target > 0, "denoise_timesteps_target must be positive"

    def _get_ema_callback(self):
        """获取 EMA callback 实例"""
        for callback in self.trainer.callbacks:
            if isinstance(callback, EMACallback):
                return callback
        return None

    def forward(self, x, t, dt, y, use_ema=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_ema: 是否使用 EMA 模型
        """
        if use_ema:
            ema_callback = self._get_ema_callback()
            if ema_callback is not None and ema_callback.ema is not None:
                with ema_callback.ema.average_parameters():
                    return self._forward_impl(x, t, dt, y)
        return self._forward_impl(x, t, dt, y)

    @abstractmethod
    def _forward_impl(self, x, t, dt, y):
        return 0

    def create_targets(self, images, labels):
        info = {}
        self.eval()

        current_batch_size = images.shape[0]

        # 1) =========== Sample dt. ============
        bootstrap_batch_size = current_batch_size // self.bootstrap_every
        log2_sections = int(math.log2(self.denoise_timesteps_target))

        if self.bootstrap_dt_bias == 0:
            dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections),
                                              bootstrap_batch_size // log2_sections)
            dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size - dt_base.shape[0], )])
            num_dt_cfg = bootstrap_batch_size // log2_sections
        else:
            dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections - 2),
                                              (bootstrap_batch_size // 2) // log2_sections)
            dt_base = torch.cat(
                [dt_base, torch.ones(bootstrap_batch_size // 4), torch.zeros(bootstrap_batch_size // 4)])
            dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size - dt_base.shape[0], )])
            num_dt_cfg = (bootstrap_batch_size // 2) // log2_sections

        force_dt_vec = torch.ones(bootstrap_batch_size) * self.force_dt
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(self.device)
        dt = 1 / (2 ** (dt_base))
        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2

        # 2) =========== Sample t. ============
        dt_sections = 2 ** dt_base
        t = torch.cat([
            torch.randint(low=0, high=int(val.item()), size=(1,)).float()
            for val in dt_sections
        ]).to(self.device)
        t = t / dt_sections
        force_t_vec = torch.ones(bootstrap_batch_size, dtype=torch.float32).to(self.device) * self.force_t
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(self.device)
        t_full = t[:, None, None, None]

        # 3) =========== Generate Bootstrap Targets ============
        x_1 = images[:bootstrap_batch_size]
        x_0 = torch.randn_like(x_1)
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        bst_labels = labels[:bootstrap_batch_size]

        with torch.no_grad():
            if not self.bootstrap_cfg:
                v_b1 = self.forward(x_t, t, dt_base_bootstrap, bst_labels, use_ema=self.bootstrap_ema)
                t2 = t + dt_bootstrap
                x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
                x_t2 = torch.clip(x_t2, -4, 4)
                v_b2 = self.forward(x_t2, t2, dt_base_bootstrap, bst_labels, use_ema=self.bootstrap_ema)
                v_target = (v_b1 + v_b2) / 2
            else:
                x_t_extra = torch.cat([x_t, x_t[:num_dt_cfg]], dim=0)
                t_extra = torch.cat([t, t[:num_dt_cfg]], dim=0)
                dt_base_extra = torch.cat([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], dim=0)
                labels_extra = torch.cat(
                    [bst_labels, torch.ones(num_dt_cfg, dtype=torch.int32).to(self.device) * self.num_classes], dim=0)

                v_b1_raw = self.forward(x_t_extra, t_extra, dt_base_extra, labels_extra, use_ema=self.bootstrap_ema)
                v_b_cond = v_b1_raw[:x_1.shape[0]]
                v_b_uncond = v_b1_raw[x_1.shape[0]:]
                v_cfg = v_b_uncond + self.cfg_scale * (v_b_cond[:num_dt_cfg] - v_b_uncond)
                v_b1 = torch.cat([v_cfg, v_b_cond[num_dt_cfg:]], dim=0)

                t2 = t + dt_bootstrap
                x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
                x_t2 = torch.clip(x_t2, -4, 4)
                x_t2_extra = torch.cat([x_t2, x_t2[:num_dt_cfg]], dim=0)
                t2_extra = torch.cat([t2, t2[:num_dt_cfg]], dim=0)

                v_b2_raw = self.forward(x_t2_extra, t2_extra, dt_base_extra, labels_extra, use_ema=self.bootstrap_ema)
                v_b2_cond = v_b2_raw[:x_1.shape[0]]
                v_b2_uncond = v_b2_raw[x_1.shape[0]:]
                v_b2_cfg = v_b2_uncond + self.cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
                v_b2 = torch.cat([v_b2_cfg, v_b2_cond[num_dt_cfg:]], dim=0)
                v_target = (v_b1 + v_b2) / 2

        v_target = torch.clip(v_target, -4, 4)
        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t
        bst_l = bst_labels

        # 4) =========== Generate Flow-Matching Targets ============
        labels_dropout = torch.bernoulli(torch.full(labels.shape, self.class_dropout_prob)).to(self.device)
        labels_dropped = torch.where(labels_dropout.bool(), self.num_classes, labels)
        info['dropped_ratio'] = torch.mean((labels_dropped == self.num_classes).float())

        t = torch.randint(low=0, high=self.denoise_timesteps_target, size=(images.shape[0],), dtype=torch.float32)
        t /= self.denoise_timesteps_target
        force_t_vec = torch.ones(images.shape[0]) * self.force_t
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(self.device)
        t_full = t[:, None, None, None]

        x_0 = torch.randn_like(images).to(self.device)
        x_1 = images
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - 1e-5) * x_0

        dt_flow = int(math.log2(self.denoise_timesteps_target))
        dt_base = (torch.ones(images.shape[0], dtype=torch.int32) * dt_flow).to(self.device)

        # 5) =========== Merge Flow+Bootstrap ============
        bst_size = current_batch_size // self.bootstrap_every
        bst_size_data = current_batch_size - bst_size

        x_t = torch.cat([bst_xt, x_t[:bst_size_data]], dim=0)
        t = torch.cat([bst_t, t[:bst_size_data]], dim=0)
        dt_base = torch.cat([bst_dt, dt_base[:bst_size_data]], dim=0)
        v_t = torch.cat([bst_v, v_t[:bst_size_data]], dim=0)
        labels_dropped = torch.cat([bst_l, labels_dropped[:bst_size_data]], dim=0)

        info['bootstrap_ratio'] = torch.mean((dt_base != dt_flow).float())
        info['v_magnitude_bootstrap'] = torch.sqrt(torch.mean(torch.square(bst_v)))
        info['v_magnitude_b1'] = torch.sqrt(torch.mean(torch.square(v_b1)))
        info['v_magnitude_b2'] = torch.sqrt(torch.mean(torch.square(v_b2)))

        return x_t, v_t, t, dt_base, labels_dropped, info

    def create_targets_naive(self, images, labels):
        info = {}
        self.eval()

        # 1) =========== Generate Flow-Matching Targets ============
        labels_dropout = torch.bernoulli(torch.full(labels.shape, self.class_dropout_prob)).to(self.device)
        labels_dropped = torch.where(labels_dropout.bool(), self.num_classes, labels)
        info['dropped_ratio'] = torch.mean((labels_dropped == self.num_classes).float())

        t = torch.randint(low=0, high=self.denoise_timesteps_target, size=(images.shape[0],), dtype=torch.float32)
        t /= self.denoise_timesteps_target
        force_t_vec = torch.ones(images.shape[0]) * -1
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(self.device)
        t_full = t[:, None, None, None]

        x_1 = images
        x_0 = torch.randn_like(images).to(self.device)
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - 1e-5) * x_0

        dt_flow = int(math.log2(self.denoise_timesteps_target))
        dt_base = (torch.ones(images.shape[0], dtype=torch.int32) * dt_flow).to(self.device)

        return x_t, v_t, t, dt_base, labels_dropped, info

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # 根据是否使用VAE处理输入
        if self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(images).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
        else:
            latents = images

        if self.training_type == "naive":
            x_t, v_t, t, dt_base, labels_dropped, info_t = self.create_targets_naive(latents, labels)
        elif self.training_type == "shortcut":
            x_t, v_t, t, dt_base, labels_dropped, info_t = self.create_targets(latents, labels)

        v_prime = self.forward(x_t, t, dt_base, labels)

        # 计算损失
        mse_v = torch.mean((v_prime - v_t) ** 2, dim=(1, 2, 3))
        loss = torch.mean(mse_v)

        # 收集指标
        info = {
            'loss': loss.item(),
            'v_magnitude_prime': torch.sqrt(torch.mean(torch.square(v_prime))).item(),
        }

        # 添加bootstrap相关信息
        if self.training_type == "shortcut":
            bootstrap_size = images.shape[0] // self.bootstrap_every
            info.update({
                'loss_flow': torch.mean(mse_v[bootstrap_size:]).item(),
                'loss_bootstrap': torch.mean(mse_v[:bootstrap_size]).item(),
                **info_t
            })

        # 记录指标
        self.log_dict({f'training/{k}': v for k, v in info.items()},
                      on_step=True, on_epoch=True, sync_dist=True)
        log_info(f"Training step {batch_idx} completed with info: {info}")

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels_real = batch

        # 根据是否使用VAE处理输入
        if self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(images).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
        else:
            latents = images

        # normalize to [0,255] range
        images = 255 * ((images - torch.min(images)) / (torch.max(images) - torch.min(images) + 1e-8))

        # sample noise
        eps_i = torch.randn_like(latents).to(self.device)

        # 计算验证损失
        x_t, v_t, t, dt_base, labels_dropped, _ = self.create_targets_naive(latents, labels_real)
        v_prime = self.forward(x_t, t, dt_base, labels_real)
        val_loss = torch.mean((v_prime - v_t) ** 2)

        # 确保验证损失被正确记录
        self.log('val_loss', val_loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 prog_bar=True,
                 logger=True)

        for i, denoise_timesteps in enumerate(self.denoise_timesteps):
            all_x = []
            delta_t = 1.0 / denoise_timesteps

            x = eps_i.to(self.device)

            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps
                t_vector = torch.full((eps_i.shape[0],), t).to(self.device)
                dt_base = torch.ones_like(t_vector).to(self.device) * math.log2(denoise_timesteps)

                with torch.no_grad():
                    v = self.forward(x, t_vector, dt_base, labels_real)

                x = x + v * delta_t

                # log 8 steps
                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == denoise_timesteps - 1:
                    if self.vae is not None:
                        with torch.no_grad():
                            decoded = self.vae.decode(x / self.vae.config.scaling_factor)[0]
                    else:
                        decoded = x
                    decoded = decoded.to("cpu")
                    all_x.append(decoded)

            if (len(all_x) == 9):
                all_x = all_x[1:]

            # estimate FID metric
            decoded_denormalized = 255 * (
                    (decoded - torch.min(decoded)) / (torch.max(decoded) - torch.min(decoded) + 1e-8))

            # generated images
            self.fids[i].update(images.to(torch.uint8).to(self.device), real=True)
            self.fids[i].update(decoded_denormalized.to(torch.uint8).to(self.device), real=False)

            # log only a single batch of generated images and only on first device
            if self.trainer.is_global_zero and batch_idx == 0:
                all_x = torch.stack(all_x)

                def process_img(img):
                    # normalize in range [0,1]
                    img = img * 0.5 + 0.5
                    img = torch.clip(img, 0, 1)
                    img = img.permute(1, 2, 0)
                    return img

                fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                for t in range(min(8, all_x.shape[0])):
                    for j in range(8):
                        axs[t, j].imshow(process_img(all_x[t, j]), vmin=0, vmax=1)

                fig.savefig(f"{self.image_save_path}/"
                            f"epoch:{self.trainer.current_epoch}_denoise_timesteps:{denoise_timesteps}.png")
                self.logger.experiment.log({f"denoise_timesteps:{denoise_timesteps}": [wandb.Image(fig)]})
                plt.close()
        return 0

    def on_validation_epoch_end(self):
        for i in range(len(self.fids)):
            denoise_timesteps_i = self.denoise_timesteps[i]

            # Compute FID for the current timestep
            fid_val_i = self.fids[i].compute()
            self.fids[i].reset()

            self.log(f"[FID] denoise_steps: {denoise_timesteps_i}", fid_val_i, on_epoch=True, on_step=False,
                     sync_dist=True)
            log_info(f"Validation end with FID: {fid_val_i}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    