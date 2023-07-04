import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torch.autograd import grad
import torchvision.transforms as transforms
from .msssim import SSIM
from .insight_face.model_irse import Backbone

from core.utils import load_state_dict
import clip
import numpy as np

from einops import rearrange, repeat

def disabled_train(self, mode=True):
    return self


def g_logistic_loss(real_pred, fake_pred):
    return F.softplus(-fake_pred).mean()

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    d_loss = real_loss + fake_loss
    return d_loss.mean()

def g_hinge(d_logit_real, d_logit_fake):
    return -torch.mean(d_logit_fake)

def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

def d_r1_loss(real_logit, real_img, r1_gamma):
    grad_real, = grad(outputs=real_logit.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return 0.5 * r1_gamma * grad_penalty

gan_losses = {'hinge' : [g_hinge, d_hinge],
              'ns' : [g_logistic_loss, d_logistic_loss],
              'dual' : [dual_contrastive_loss, dual_contrastive_loss]}

def tv_loss(img):
    # https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/image/tv.py
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = (h_variance + w_variance)
    return loss



class SSIM_Loss(SSIM):            
    def forward(self, img1, img2):
        return ( 1 - super(SSIM_Loss, self).forward(img1, img2) )


class IDLoss(nn.Module):
    # sh ./download_from_google_drive.sh 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn model_ir_se50.pth
    def __init__(self, backbone_path):
        super(IDLoss, self).__init__()   
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        try:
            self.facenet.load_state_dict(load_state_dict(backbone_path, location='cpu'))
        except IOError:
            self.facenet.load_state_dict(torch.load('/apdcephfs/share_916081/amosyhliu/pretrained_models/model_ir_se50.pth'))
        
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.train = disabled_train
        for param in self.facenet.parameters():
            param.requires_grad = False
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, x_hat):
        self.facenet.eval()
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        x_feats = x_feats.detach()

        x_hat_feats = self.extract_feats(x_hat)
        losses = []
        for i in range(n_samples):
            loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
            losses.append(loss_sample.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        return losses / n_samples


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, tgt, m):
        return self.mse(m*pred, m*tgt)


class ConstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_dtype = torch.bool

    def forward(self, feat_q,feat_k):

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = 1

        loss_co2 = 0
        qsim = []
        ksim = []
        for i in range(feat_q.size(0)):
            q_temp = F.cosine_similarity(feat_q[i].unsqueeze(0),feat_q)
            k_temp = F.cosine_similarity(feat_k[i].unsqueeze(0),feat_k)
            for j in range(feat_q.size(0)):
                if i!=j:
                    qsim.append(q_temp[j].unsqueeze(0))
                    ksim.append(k_temp[j].unsqueeze(0))

        qsim = torch.cat(qsim,dim=0)
        ksim = torch.cat(ksim,dim=0)

        loss_co2 = torch.mean((qsim-ksim)**2)

        return loss_co2

    
class PatchLoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch = batch_size
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        feat_q_norm = feat_q.clone().norm(p=2,dim=-1,keepdim=True)
        feat_k_norm = feat_k.clone().norm(p=2,dim=-1,keepdim=True)
        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos_norm = feat_k_norm*feat_q_norm
        l_pos = l_pos.view(batchSize, 1) / l_pos_norm


        batch_dim_for_bmm = self.batch

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        
        feat_q_norm = feat_q_norm.view(batch_dim_for_bmm, -1, 1)
        feat_k_norm = feat_k_norm.view(batch_dim_for_bmm, -1, 1)
        
        npatches = feat_q.size(1)
        l_neg_norm = torch.bmm(feat_q_norm, feat_k_norm.transpose(2, 1))
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) / l_neg_norm
        

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class CLIPLoss(nn.Module):
    def __init__(self, clip_model="ViT-B/32", device='cpu', 
                 clip_loss_type='clip_sim_loss', batch_for_patch=32):
        super().__init__()

        if clip_model == "ViT-B/32":
            clip_model = "/env/pretrained/ViT-B-32.pt"
        
        self.model, clip_preprocess = clip.load(clip_model, device=device, jit=False)
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + 
                                              clip_preprocess.transforms[:2] +
                                              clip_preprocess.transforms[4:])
        self.clip_loss_type = clip_loss_type
        self.texture_loss = torch.nn.MSELoss()
        self.sim = nn.CosineSimilarity()
        self.patch_loss = PatchLoss(batch_for_patch)
        self.const_loss = ConstLoss()
        self.calc_loss = getattr(self, clip_loss_type)

    def forward(self, pred, gt):
        return self.calc_loss(pred, gt)
    
    def encode(self, img):
        processed_img = self.preprocess(img)
        image_features = self.model.encode_image(processed_img)
        return image_features

    def txt_loss(self, pred, gt):
        pred_feat = self.encode(pred)
        gt_feat = self.encode(gt)
        return self.texture_loss(pred_feat, gt_feat)
    
    def clip_sim_loss(self, pred, gt):
        pred_feat = self.encode(pred)
        gt_feat = self.encode(gt)
        pred_feat /= pred_feat.clone().norm(dim=-1, keepdim=True)
        gt_feat /= gt_feat.clone().norm(dim=-1, keepdim=True)
        clip_loss = clip_loss = (1.0-self.sim(pred_feat,gt_feat)).mean()
        return clip_loss

    def rand_sampling_mult(self,sizes,crop_size,num_crops,content_image,out_image,prop=None):
        bbxl=[]
        bbyl=[]
        crop_image = []
        tar_image = []

        for cc in range(num_crops):
            bbx1, bby1 = self.rand_bbox(sizes, crop_size,prop)
            crop_image.append(content_image[:,:,bby1:bby1+crop_size,bbx1:bbx1+crop_size])
            tar_image.append(out_image[:,:,bby1:bby1+crop_size,bbx1:bbx1+crop_size])
        crop_image = torch.cat(crop_image,dim=0)
        tar_image = torch.cat(tar_image,dim=0)
        return crop_image,tar_image
    
    def rand_bbox(self, size, res,prop=None):
        W = size
        H = size
        cut_w = res
        cut_h = res
        if prop is not None:
            res = np.random.rand()*(prop[1]-prop[0])
            cut_w = int(res*W)
            cut_h = int(res*H)
        tx = np.random.randint(0,W-cut_w)
        ty = np.random.randint(0,H-cut_h)
        bbx1 = tx
        bby1 = ty
        return bbx1, bby1
    
    def clip_patch_loss(self, pred, gt, size, crop_size, num_crop):
        gt_img_crop, pred_img_crop = self.rand_sampling_mult(size, crop_size, num_crop, gt, pred)
        gt_patch = self.encode(gt_img_crop)
        pred_patch = self.encode(pred_img_crop)
        gt_patch /= gt_patch.clone().norm(dim=-1, keepdim=True)
        pred_patch /= pred_patch.clone().norm(dim=-1, keepdim=True)
        return self.patch_loss(pred_patch, gt_patch).mean()
    
    def clip_const_loss(self, pred, gt):
        pred_feat = self.encode(pred)
        gt_feat = self.encode(gt)
        pred_feat /= pred_feat.clone().norm(dim=-1, keepdim=True)
        gt_feat /= gt_feat.clone().norm(dim=-1, keepdim=True)
        return self.const_loss(pred_feat, gt_feat)
