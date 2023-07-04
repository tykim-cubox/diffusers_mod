import os
import math
# random
import random
# pytorch
import torch
import torch.nn as nn
import torchvision
import lightning.pytorch as pl

from utils import instantiate_from_config, exists, count_params, discard_kwargs, load_state_dict, get_model_size
from optimizer import optimizers, lr_schedulers
from diffaug import DiffAugment
# from .losses import i2iloss

import sys
sys.path.append('/purestorage/project/tyk/project9/diffusers_mod/src/diffusers')

class SimpleCoach(pl.LightningModule):
    def __init__(self, model_config, loss_config, training_config, use_ema=False, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        # Model
        # self.load_model(self.model_config)
        self.load_model()
        # Objectives
        self.setup_objectives()
        self.save_hyperparameters()

    def load_model(self):
        # super().load_model(model_config)
        self.model = instantiate_from_config(self.model_config.generator)
        count_params(self.model, verbose=True)
        size_mb = get_model_size(self.model)
        self.log(f'model size', size_mb)
        if self.training_config.resume_path:
            self.model.load_state_dict(load_state_dict(self.training_config.resume_path,
                                                       location='cpu'))
    def setup_objectives(self):
        # super().setup_objectives()
        # **kwargs 
        self.loss = instantiate_from_config(self.loss_config)

    def configure_optimizers(self):
        lr = self.training_config.lr
        op_name = self.training_config.optimizer
        params = list(self.model.parameters())
        opt = discard_kwargs(optimizers[op_name])(net=self.model, params=params,
                                                 lr=lr)
        return [opt], []
    
    def training_step(self, batch, batch_idx):
        pred = self(batch['src'])
        gt = batch['tgt']
        loss_dict, loss = self.loss.calc_loss(pred, gt)
        {self.log(f'{key}', value) for key, value in loss_dict.items()}
        return loss

    def forward(self, src_img):
        return self.model(src_img)
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self(batch['src'])
        grid = torchvision.utils.make_grid(pred) 
        # self.logger.experiment.add_image('generated_images', grid, 0) 
        self.logger.log_image("name", [grid, batch['tgt']])
        return pred


# class AuxCoach(SimpleCoach):
#     def __init__(self, ):
#         super().__init__()
#         self.
        
#     def load_model(self, model_config):
#         self.generator = 
#         return super().load_model(model_config)


#     def training_step(self, )

class GANCoach(pl.LightningModule):
    # def __init__(self, gan_config, *args, **kwargs):
    def __init__(self, model_config, loss_config, training_config, use_ema=False, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        # self.gan_config = gan_config
        self.automatic_optimization = False
        # super().__init__(*args, **kwargs)
        self.load_model()
        self.setup_objectives()
        self.save_hyperparameters()
        
    def load_model(self):
        # super().load_model(model_config)
        self.model = instantiate_from_config(self.model_config.generator)
        count_params(self.model, verbose=True)
        size_mb = get_model_size(self.model)
        self.log(f'model size', size_mb)
        
        self.discriminator = instantiate_from_config(self.model_config.discriminator)
        count_params(self.discriminator, verbose=True)

        if self.training_config.resume_path:
            self.model.load_state_dict(load_state_dict(self.training_config.resume_path,
                                                       location='cpu'))
            
            self.discriminator.load_state_dict(load_state_dict(self.training_config.resume_path,
                                                       location='cpu'))


    def setup_objectives(self):
        # super().setup_objectives()
        # **kwargs 
        self.loss = instantiate_from_config(self.loss_config)
        # self.loss = i2i_loss.I2ILoss(**self.loss_config)

    def configure_optimizers(self):
        lr = self.training_config.lr
        disc_lr = self.training_config.disc_lr
        op_name = self.training_config.optimizer
        params_g = list(self.model.parameters())
        params_d = list(self.discriminator.parameters())

        opt_g = discard_kwargs(optimizers[op_name])(net=self.model, params=params_g, lr=lr)
        opt_d = discard_kwargs(optimizers[op_name])(net=self.discriminator, params=params_d, lr=disc_lr)

        scheduler_lst = []
        if self.training_config.g_scheduler:
            genertator_lr_scheduler = lr_schedulers[self.training_config.g_scheduler](opt_g, T_0=self.training_config.scheduler_T0, T_mult=self.training_config.g_t_mult)
            scheduler_lst.append(genertator_lr_scheduler)
            
        if self.training_config.d_scheduler:
            discriminator_lr_scheduler = lr_schedulers[self.training_config.d_scheduler](opt_d, T_0=self.training_config.scheduler_T0, T_mult=self.training_config.d_t_mult)
            scheduler_lst.append(discriminator_lr_scheduler)

        
        return [opt_g, opt_d], scheduler_lst

    def forward(self, src_img):
        return self.model(src_img).sample
    
    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_lst = self.lr_schedulers()
        if scheduler_lst:
            scheduler_g, scheduler_d = scheduler_lst[0], scheduler_lst[1]
        gt = batch['tgt']
        # train d
        if self.loss.use_loss['gan']:
            self.toggle_optimizer(optimizer_d)
            if self.global_step % self.training_config.d_reg_every == 0:
                optimizer_d.zero_grad()
                gt.requires_grad = True
                real_logit = self.discriminator(gt)
                reg_d_dict, reg_d_loss = self.loss.regularize_d(real_logit, gt)
                {self.log(f'{key}', value) for key, value in reg_d_dict.items()}
                self.manual_backward(reg_d_loss * self.training_config.d_reg_every)
                optimizer_d.step()
            else:
                optimizer_d.zero_grad()
                fake_logit = self.discriminator(self(batch['src']).detach())
                real_logit = self.discriminator(gt)
                d_loss_dict, d_loss = self.loss.loss_d(fake_logit, real_logit)
                {self.log(f'{key}', value) for key, value in d_loss_dict.items()}
                self.manual_backward(d_loss)
                optimizer_d.step()
                if scheduler_lst:
                    scheduler_d.step()
            self.untoggle_optimizer(optimizer_d)
        
        # traing g
        self.toggle_optimizer(optimizer_g)
        for _ in range(self.training_config.generator_step):
            optimizer_g.zero_grad()
            pred = self(batch['src'])
            gt = batch['tgt']    
            fake_logit = self.discriminator(pred) if self.loss.use_loss['gan'] else None
            if self.loss.use_loss['gan'] and self.loss_config.gan_loss_type == 'dual':
                real_logit = self.discriminator(gt)

            if self.training_config.topk_training:
                k_frac = max(self.training_config.generator_top_k_gamma** self.current_epoch, self.training_config.top_k_frac)
                k = math.ceil(self.training_config.batch_size * k_frac)
                if k != self.training_config.batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)
            g_loss_dict, g_loss = self.loss.loss_g(pred, gt, real_logit, fake_logit)
            {self.log(f'{key}', value) for key, value in g_loss_dict.items()}
            self.manual_backward(g_loss)
            optimizer_g.step()
            if scheduler_lst:
                scheduler_g.step()


        self.untoggle_optimizer(optimizer_g)

        self.loss.toggle_loss(self.global_step)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self(batch['src'])
        grid = torchvision.utils.make_grid(pred) 
        # self.logger.experiment.add_image('generated_images', grid, 0) 
        self.logger.log_image("name", [grid, batch['tgt']])
        return pred
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        if self.current_epoch > 5:
            gradient_clip_val = gradient_clip_val * 2

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)
        

import torch.nn.functional as F

class PreGANCoach(pl.LightningModule):
    def __init__(self, model_config, loss_config, training_config, use_ema=False, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        # self.gan_config = gan_config
        self.automatic_optimization = False
        # super().__init__(*args, **kwargs)
        self.load_model()
        self.setup_objectives()
        self.save_hyperparameters()
        self.policy = 'color,translation'
        
    def load_model(self):
        # super().load_model(model_config)
        self.model = instantiate_from_config(self.model_config.generator)
        count_params(self.model, verbose=True)
        size_mb = get_model_size(self.model)
        self.log(f'model size', size_mb)
        
        self.discriminator = instantiate_from_config(self.model_config.discriminator)
        count_params(self.discriminator, verbose=True)

        if self.training_config.resume_path:
            self.model.load_state_dict(load_state_dict(self.training_config.resume_path,
                                                       location='cpu'))
            
            self.discriminator.load_state_dict(load_state_dict(self.training_config.resume_path,
                                                       location='cpu'))


    def setup_objectives(self):
        # super().setup_objectives()
        # **kwargs 
        self.loss = instantiate_from_config(self.loss_config)
        # self.loss = i2i_loss.I2ILoss(**self.loss_config)

    def configure_optimizers(self):
        lr = self.training_config.lr
        disc_lr = self.training_config.disc_lr
        op_name = self.training_config.optimizer
        params_g = list(self.model.parameters())
        params_d = list(self.discriminator.parameters())

        opt_g = discard_kwargs(optimizers[op_name])(net=self.model, params=params_g, lr=lr)
        opt_d = discard_kwargs(optimizers[op_name])(net=self.discriminator, params=params_d, lr=disc_lr)

        scheduler_lst = []
        if self.training_config.g_scheduler:
            genertator_lr_scheduler = lr_schedulers[self.training_config.g_scheduler](opt_g, T_0=self.training_config.scheduler_T0, T_mult=self.training_config.g_t_mult)
            scheduler_lst.append(genertator_lr_scheduler)
            
        if self.training_config.d_scheduler:
            discriminator_lr_scheduler = lr_schedulers[self.training_config.d_scheduler](opt_d, T_0=self.training_config.scheduler_T0, T_mult=self.training_config.d_t_mult)
            scheduler_lst.append(discriminator_lr_scheduler)

        
        return [opt_g, opt_d], scheduler_lst

    def forward(self, src_img):
        return self.model(src_img)
    
    def train_d(net, data, label="real"):
        """Train function of discriminator"""
        if label=="real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
            err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
                percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
                percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
                percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = net(data, label)
            err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()
        data = batch['tgt']
        batch_size = self.training_config.batch_size
        c, h, w = self.training_config.inp_size

        self.toggle_optimizer(optimizer_d)
        if self.global_step % self.training_config.d_reg_every == 0:
            optimizer_d.zero_grad()
            gt.requires_grad = True
            real_logit = self.discriminator(gt)
            reg_d_dict, reg_d_loss = self.loss.regularize_d(real_logit, gt)
            {self.log(f'{key}', value) for key, value in reg_d_dict.items()}
            self.manual_backward(reg_d_loss * self.training_config.d_reg_every)
            optimizer_d.step()
        else:
            optimizer_d.zero_grad()
            noise = torch.Tensor(batch_size, c, h, w).normal_(0, 1)

            real_imgs = DiffAugment(data, policy=self.policy)
            fake_imgs = DiffAugment(self(noise))

            real_logit = self.discriminator(real_imgs)
            fake_logit = self.discriminator(fake_imgs)
        
            d_loss_dict, d_loss = self.loss.loss_d(fake_logit, real_logit)
            {self.log(f'{key}', value) for key, value in d_loss_dict.items()}
            self.manual_backward(d_loss)
            optimizer_d.step()
            scheduler_d.step()

        self.untoggle_optimizer(optimizer_d)
    
        # traing g
        self.toggle_optimizer(optimizer_g)
        for _ in range(self.training_config.generator_step):
            optimizer_g.zero_grad()
            pred = self(batch['src'])
            gt = batch['tgt']    
            fake_logit = self.discriminator(pred) if self.loss.use_loss['gan'] else None
            if self.loss.use_loss['gan'] and self.loss_config.gan_loss_type == 'dual':
                real_logit = self.discriminator(gt)

            if self.training_config.topk_training:
                k_frac = max(self.training_config.generator_top_k_gamma** self.current_epoch, self.training_config.top_k_frac)
                k = math.ceil(self.training_config.batch_size * k_frac)
                if k != self.training_config.batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)
            g_loss_dict, g_loss = self.loss.loss_g(pred, gt, real_logit, fake_logit)
            {self.log(f'{key}', value) for key, value in g_loss_dict.items()}
            self.manual_backward(g_loss)
            optimizer_g.step()
            scheduler_g.step()


        self.untoggle_optimizer(optimizer_g)

        self.loss.toggle_loss(self.global_step)

################################################
############### Distiller ######################
################################################
class Distiller(pl.LightningModule):
    def __init__(self, model_config, loss_config, training_config, distiller_config):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        self.distiller_config = distiller_config
        
    def training_step(self, batch):
        ...

    def loss(self):
        ...
    @torch.no_grad()
    def make_sample(self, batch):
        if self.cfg.gan_teacher is True:
            ...

class GANDistiller(Distiller):
    def __init__(self, model_config, loss_config, training_config, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        ...

class DiffusionDistiller(Distiller):
    def __init__(self, model_config, loss_config, training_config, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        ...