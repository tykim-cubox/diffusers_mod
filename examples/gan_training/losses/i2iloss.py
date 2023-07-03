import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import *
from lpips import LPIPS
from focal_frequency_loss import FocalFrequencyLoss as FFL
from pytorch_wavelets import DWTInverse, DWTForward

from core.utils import exists

class SimpleLoss(nn.Module):
    def __init__(self, domain_rgb, l1_lambda, l2_lambda):
        super().__init__()
        
        params = locals()
        params.pop('self')
        for key, value in params.items():
            setattr(self, key, value)

        self.l1_loss = nn.L1Loss() if self.l1_lambda != 0 else None
        self.l2_loss = nn.MSELoss() if self.l2_lambda !=0 else None
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        self.idwt = DWTInverse(mode="zero", wave="db1")

        
    def calc_loss(self, pred, gt):
        loss = 0.0
        loss_dict = {}
        if self.l1_loss : 
            l1_loss_val = self.l1_lambda * self.l1_loss(pred, gt)
            loss_dict.update({'l1_loss' : l1_loss_val})
            loss += l1_loss_val
            
            pred_new_domain = self.img_to_dwt(pred) if self.domain_rgb else self.dwt_to_img(pred)
            gt_new_domain = self.img_to_dwt(gt) if self.domain_rgb else self.dwt_to_img(gt)
            freq_loss_val = self.l1_lambda * self.l1_loss(pred_new_domain, gt_new_domain)
            loss_dict.update({'freq_loss' : freq_loss_val})
            loss += freq_loss_val
            
        if self.l2_loss :
            l2_loss_val = self.l2_lambda * self.l2_loss(pred, gt)
            loss_dict.update({'l2_loss_2' : l2_loss_val})
            loss += l2_loss_val

        return loss_dict, loss
    
    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))
        
class I2ILoss(nn.Module):
    def __init__(self, domain_rgb, l1_lambda , l2_lambda, 
                 lpips_lambda, lpips_type, lpips_model_path,
                 tv_lambda, ssim_lambda, cnt_lambda, 
                 id_lambda, id_backbone_path, ffl_w, ffl_alpha, gan_loss_type, gan_lambda, r1_gamma,
                 clip_lambda, clip_loss_type, clip_loss_batch,
                 l1_apply=0, l2_apply=0,
                 lpips_apply=0, tv_apply=0, ssim_apply=0, cnt_apply=0,
                 id_apply=0, ffl_apply=0, gan_apply=0, clip_apply=0
                 ):
        super().__init__()

        params = locals()
        # print(params)
        # self.domain_rgb = domain_rgb
        # self.l1_lambda = l1_lambda
        # self.l2_lambda = l2_lambda
        # self.lpips_lambda = lpips_lambda
        # self.lpips_type = lpips_type
        # self.tv_lambda = tv_lambda
        # self.ssim_lambda = ssim_lambda
        # self.cnt_lambda = cnt_lambda
        # self.id_lambda = id_lambda
        # self.ffl_w = ffl_w
        # self.ffl_alpha = ffl_alpha
        # self.gan_loss_type = gan_loss_type
        # self.r1_gamma = r1_gamma

        params.pop('self')
        for key, value in params.items():
            setattr(self, key, value)

        self.loss_apply = dict()
        self.use_loss = dict()

        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        self.idwt = DWTInverse(mode="zero", wave="db1")
        
        self.l1_loss = nn.L1Loss() if self.l1_lambda != 0 else None
        self.loss_apply.update({'l1': l1_apply})
        self.use_loss.update({'l1': False})

        self.l2_loss = nn.MSELoss() if self.l2_lambda != 0 else None
        self.loss_apply.update({'l2': l2_apply})
        self.use_loss.update({'l2': False})

        self.lpips_loss = LPIPS(net=lpips_type, pnet_rand=True, pretrained=True, model_path=lpips_model_path) if self.lpips_lambda else None
        if self.lpips_loss:
            self.lpips_loss.train = disabled_train 
        self.loss_apply.update({'lpips': lpips_apply})
        self.use_loss.update({'lpips': False})
        
        self.tv_loss = tv_loss if tv_lambda else None
        self.loss_apply.update({'tv': tv_apply})
        self.use_loss.update({'tv': False})
        
        self.ssim_loss = SSIM_Loss() if self.ssim_lambda else None
        self.loss_apply.update({'ssim': ssim_apply})
        self.use_loss.update({'ssim': False})
        
        self.content_loss = ContentLoss() if self.cnt_lambda else None
        self.loss_apply.update({'cnt': cnt_apply})
        self.use_loss.update({'cnt': False})

        self.id_loss = IDLoss(id_backbone_path) if self.id_lambda else None
        self.loss_apply.update({'id': id_apply})
        self.use_loss.update({'id': False})

        self.ff_loss = FFL(loss_weight=self.ffl_w, alpha=self.ffl_alpha,
                           patch_factor=1,ave_spectrum=True, log_matrix=True, batch_matrix=True) if self.ffl_w else None
        self.loss_apply.update({'ffl': ffl_apply})
        self.use_loss.update({'ffl': False})

        
        self.g_loss = gan_losses[gan_loss_type][0] if gan_loss_type else None
        self.d_loss = gan_losses[gan_loss_type][1] if gan_loss_type else None
        self.loss_apply.update({'gan': gan_apply})
        self.use_loss.update({'gan': False})
        self.r1_reg = d_r1_loss if self.r1_gamma else None

        self.clip_loss = CLIPLoss(batch_for_patch=clip_loss_batch) if self.clip_lambda else None
        self.loss_apply.update({'clip': clip_apply})
        self.use_loss.update({'clip': False})

        self.toggle_loss(0)

    def toggle_loss(self, step):
        for key, value in self.loss_apply.items():
            start, end = value
            if end >= step >= start:
                self.use_loss[key] = True
            else:
                self.use_loss[key] = False

    def loss_g(self, pred, gt, real_logit, fake_logit, m=None):
        loss = 0.0
        loss_dict = {}
        # Recon loss
        pred_new_domain = self.img_to_dwt(pred) if self.domain_rgb else self.dwt_to_img(pred)
        gt_new_domain = self.img_to_dwt(gt) if self.domain_rgb else self.dwt_to_img(gt)

        if exists(self.l1_loss) and self.use_loss['l1'] : 
            l1_loss_val = self.l1_lambda * self.l1_loss(pred, gt)
            loss_dict.update({'l1_loss' : l1_loss_val})
            loss += l1_loss_val
            
            pred_new_domain = self.img_to_dwt(pred) if self.domain_rgb else self.dwt_to_img(pred)
            gt_new_domain = self.img_to_dwt(gt) if self.domain_rgb else self.dwt_to_img(gt)
            freq_loss_val = self.l1_lambda * self.l1_loss(pred_new_domain, gt_new_domain)
            loss_dict.update({'freq_loss' : freq_loss_val})
            loss += freq_loss_val
            
        if exists(self.l2_loss) and self.use_loss['l2']:
            l2_loss_val = self.l2_lambda * self.l2_loss(pred, gt)
            loss_dict.update({'l2_loss_2' : l2_loss_val})
            loss += l2_loss_val
        
        if exists(self.lpips_loss) and self.use_loss['lpips'] : 
            lpips_loss_val = self.lpips_lambda * self.lpips_loss(pred, gt).mean()
            loss_dict.update({'lpips_loss' : lpips_loss_val})
            loss += lpips_loss_val
        if exists(self.tv_loss) and self.use_loss['tv'] :
            tv_loss_val = self.tv_lambda * self.tv_loss(pred) 
            loss_dict.update({'tv_loss' : tv_loss_val})
            loss += tv_loss_val
        if exists(self.ssim_loss) and self.use_loss['ssim'] : 
            ssim_loss_val = self.ssim_lambda * self.ssim_loss(pred, gt)
            loss_dict.update({'ssim_loss' : ssim_loss_val})
            loss += ssim_loss_val
        if exists(self.content_loss) and self.use_loss['cnt'] : 
            content_loss_val = self.cnt_lambda * self.content_loss(pred, gt, m)
            loss_dict.update({'content_loss' : content_loss_val})
            loss += content_loss_val
        if exists(self.id_loss) and self.use_loss['id'] : 
            id_loss_val = self.id_lambda * self.id_loss(pred, gt).mean()
            loss_dict.update({'id_loss' : id_loss_val})
            loss += id_loss_val
        if exists(self.ff_loss) and self.use_loss['ffl'] : 
            ff_loss_val = self.ff_loss(pred, gt)
            loss_dict.update({'ff_loss' : ff_loss_val})
            loss += ff_loss_val
        if exists(self.g_loss) and self.use_loss['gan']: 
            g_loss_val = self.gan_lambda * self.g_loss(real_logit, fake_logit)
            loss_dict.update({'g_loss' : g_loss_val})
            loss += g_loss_val

        if exists(self.clip_loss) and self.use_loss['clip'] : 
            clip_loss_val = self.clip_loss.clip_sim_loss(pred, gt)
            loss_dict.update({'clip_loss' : clip_loss_val})
            loss += clip_loss_val

        loss_dict.update({'total_g_loss' : loss})
        return loss_dict, loss
        
    def loss_d(self, fake_logit, real_logit):
        loss = 0.0
        loss_dict = {}

        if self.use_loss['gan'] : loss = self.d_loss(real_logit, fake_logit)
        loss_dict.update({'d_loss' : loss})
        return loss_dict, loss

    def regularize_d(self, real_logit, real_img):
        reg_d_dict = {}
        if exists(self.r1_reg):
            r1 = self.r1_reg(real_logit, real_img, self.r1_gamma)
            reg_d_dict.update({'r1_reg' : r1})
            return reg_d_dict, r1

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))