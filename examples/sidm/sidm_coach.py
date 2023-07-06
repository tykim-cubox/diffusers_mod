import os
import math
# random
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import torchvision
import lightning.pytorch as pl

from packaging import version

from .utils import instantiate_from_config, exists, count_params, discard_kwargs, load_state_dict, get_model_size


import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, UNet2DImageConditionModel, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available



class Coach(pl.LightningModule):
    def __init__(self, model_config, loss_config, training_config, distiller_config):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        self.distiller_config = distiller_config

        self.load_model()
        self.setup_objectives()
        
        if training_config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.save_hyperparameters()

    def load_model(self):
        print("load UNet network...")
        # self.model = UNet2DImageConditionModel.from_config(self.model_config)
        self.model = instantiate_from_config(self.model_config.generator)
        
        if self.model_config.use_ema:
            self.ema_model = EMAModel(self.model.parameters(),
                                      decay=self.model_config.ema_max_decay,
                                      use_ema_warmup=True,
                                      inv_gamma=self.model_config.ema_inv_gamma,
                                      power=self.model_config.ema_power,
                                      model_cls=UNet2DModel,
                                      model_config=self.model.config,
                                )

        if self.model_config.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                self.model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.model_config.scheduler_cfg_path is None:
            self.noise_scheduler = DDPMScheduler(beta_end=0.012,
                                                 beta_schedule= "scaled_linear",
                                                beta_start= 0.00085,
                                                num_train_timesteps= 1000,
                                                set_alpha_to_one= False,
                                                skip_prk_steps= True,
                                                steps_offset= 1,
                                                trained_betas= None,
                                                clip_sample= False)
        else:
            self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_config.scheduler_cfg_path, subfolder="scheduler")

    def setup_objectives(self):
        # Loss를 기존것과 호환할 수 있도록 할 수 있다면
        # 결국 pred와 target 값을 비교하는 식이라서 l1을 사용할 수 있고
        # 이미지 space라면 lpips 등을 사용할 수 있게 해야하는데
        # super().setup_objectives()
        # **kwargs 
        self.loss = instantiate_from_config(self.loss_config)
        # self.loss = i2i_loss.I2ILoss(**self.loss_config)

    def configure_optimizers(self):
        lr = self.training_config.learning_rate
        params = list(self.model.parameters())

        if self.learn_logvar:
            params = params + [self.logvar]

        optimizer = torch.optim.AdamW(params, lr=lr,
                                  betas=(self.training_config.adam_beta1, self.self.training_config.adam_beta2),
                                  weight_decay=self.training_config.adam_weight_decay,
                                  eps=self.training_config.adam_epsilon)
        
        lr_scheduler = get_scheduler(
            self.training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.training_config.lr_warmup_steps,
            num_training_steps=(len(self.training_config.train_length) * self.training_config.num_epochs),)
        

        return [optimizer], [lr_scheduler]

    def forward(self, src_img):
        return self.model(src_img).sample
    
    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(arr)
        res = arr[timesteps].float().to(timesteps.device)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def training_step(self, batch, batch_idx):
        src = batch['src']
        tgt = batch['tgt']
        cond = batch['cond']

        # Sample noise that we'll add to the images
        noise = torch.randn(tgt.shape)
        bsz = tgt.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,)).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(tgt, noise, timesteps)

        # Predict the noise residual
        # enc_result = self.encoder(inp)
        model_output = self.model(noisy_images, cond, timesteps).sample

        loss = 0.0
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise

            if self.training_config.snr_gamma is None:
                loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = self.compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack([snr, self.training_config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                # We first calculate the original loss. Then we mean over the non-batch dimensions and
                # rebalance the sample-wise losses with their respective loss weights.
                # Finally, we take the mean of the rebalanced loss.
                loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # 추가 loss
            # 여기선 sample space로 돌려야함
            loss += self.loss(model_output.float(), target.float())
        elif self.noise_scheduler.config.prediction_type == "sample":
            alpha_t = self._extract_into_tensor(
                        self.noise_scheduler.alphas_cumprod, timesteps, (out.shape[0], 1, 1, 1)
                    )
            snr_weights = alpha_t / (1 - alpha_t)
            loss = snr_weights * F.mse_loss(model_output, tgt, reduction="none")
            # 추가 loss
            loss += self.loss(model_output.float(), tgt.float())
        else:
            raise ValueError(f"Unsupported prediction type: {self.noise_scheduler.config.prediction_type}")
        
        self.log(f'loss', loss)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    # controlnet
    def on_train_batch_end(self, *args, **kwargs):
        if self.model_config.use_ema:
            self.ema_model(self.model)

    def on_validation_start(self):
        if self.model_config.use_ema:
            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())

        self.pipeline = instantiate_from_config(self.model_config.generator)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)

        if self.model_config.enable_xformers_memory_efficient_attention: 
            self.pipeline.enable_xformers_memory_efficient_attention()

        if self.trainign_config.seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator().manual_seed(self.trainign_config.seed)


    def on_validation_end(self):
        del self.pipeline
        torch.cuda.empty_cache()
        if self.ema_model.use_ema:
            # Switch back to the original UNet parameters.
            self.ema_model.restore(self.model.parameters())


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        image = self.pipeline(batch['cond'], num_inference_steps=20, generator=self.generator).images[0]
        grid = torchvision.utils.make_grid(image) 
        # self.logger.experiment.add_image('generated_images', grid, 0) 
        self.logger.log_image("name", [grid, batch['tgt']])
        return image

    def configure_gradient_clipping(self, optimizer, gradient_clip_algorithm, gradient_clip_val=1):
        if self.current_epoch > 5:
            gradient_clip_val = gradient_clip_val

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)