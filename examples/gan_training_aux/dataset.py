import os
from pathlib import Path

import random

import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import einops

# 일단 jpg이라서 고려x
try:
    import pyspng
except ImportError:
    pyspng = None

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

interpolation = {"nearest": InterpolationMode.NEAREST,
                "box": InterpolationMode.BOX,
                "bilinear": InterpolationMode.BILINEAR,
                "bicubic": InterpolationMode.BICUBIC,
                "lanczos": InterpolationMode.LANCZOS}

##########################################
############### Loader ###################
##########################################

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if not img.mode == 'RGB':
            img = img.convert("RGB")
        return np.array(img)
    
def cv_loader(path):
    image = cv2.imread(path) # np.array [H, W, C]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def png_loader(file_path):
    with open(file_path, 'rb') as f:
        image = pyspng.load(f.read())
    return image

loaders = {'pil': pil_loader, 'cv' : cv_loader, 'png' : png_loader}

##########################################
######### Transformation #################
##########################################
class CenterCropMargin(object):
    def __init__(self, fraction=0.95):
        super().__init__()
        self.fraction=fraction
        
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size)*self.fraction)

    def __repr__(self):
        return self.__class__.__name__


class Normalize(nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, data_dict):
        data_dict = {k: TF.normalize(v, self.mean, self.std, self.inplace) for k, v in data_dict.items()}
        return data_dict
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data_dict):

        if torch.rand(1) < self.p:
            return {k: TF.hflip(v) for k, v in data_dict.items()}
        return data_dict

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, data_dict):
        return {k: TF.resize(v, self.size, self.interpolation, self.max_size, self.antialias) for k, v in data_dict.items()}

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)


class ToTensor:
    def __call__(self, data_dict):
        return {k: TF.to_tensor(v) for k, v in data_dict.items()}

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
###################################
######### Dataset #################
###################################
class PairedDataset(Dataset):
    
    def __init__(self, root, file_list_path, src_name='source', cond_name='target', tgt_name='condition',
                 crop=False, resize_size=None, normalize=True, random_flip=0.5, exclude_cond=True, 
                 loader_type='cv', inte_mode='bicubic'):
        super().__init__()
        # self.root = Path(root)
        # self.src_path = self.root.joinpath(src_name)
        # self.tgt_path = self.root.joinpath(tgt_name)
        # self.cond_path = self.root.joinpath(cond_name)
        self.root = root
        self.src_path = os.path.join(root, src_name)
        self.tgt_path = os.path.join(root, tgt_name)
        self.cond_path = os.path.join(root, cond_name)
        self.exclude_cond = exclude_cond

        Image.init()

        self.fnames = []
        with open(file_list_path, 'r') as file:
            for line in file:
                self.fnames.append(line.strip('\n'))

        self.loader = loaders[loader_type]
        self.interpolation = interpolation[inte_mode]

        self.trsf_list = []
        if crop:
            self.crop = CenterCropMargin(fraction=0.95)
            self.trsf_list.append(self.crop)
        
        self.trsf_list.append(ToTensor())

        if resize_size is not None and interpolation != 'wo_resize':
            self.resizer = Resize(resize_size, interpolation=self.interpolation)
            self.trsf_list.append(self.resizer)

        if random_flip > 0:
            self.flipper = RandomHorizontalFlip(random_flip)
            self.trsf_list.append(self.flipper)
        
        if normalize:
            self.normalizer = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.trsf_list.append(self.normalizer)

        self.trsf = transforms.Compose(self.trsf_list)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    # def transformation(self, src_img, r_img):
    #     if self.crop is not None:
    #         p_img, r_img = self.crop(p_img), self.crop(r_img)

    #     if self.resizer is not None:
    #         p_img, r_img = self.resizer(p_img), self.resizer(r_img)

    #     if self.random_flip and random.random() > 0.5:
    #         p_img = TF.hflip(p_img)
    #         r_img = TF.hflip(r_img)

    #     p_img, r_img = self.to_tensor(p_img), self.to_tensor(r_img)

    #     if self.normalizer is not None:
    #         p_img, r_img = self.normalizer(p_img), self.normalizer(r_img)
    #     return p_img, r_img



    def _load_raw_image(self, raw_idx):
        fname = self.fnames[raw_idx]
        
        src_img = self.loader(os.path.join(self.src_path, fname))
        tgt_img = self.loader(os.path.join(self.tgt_path, fname))

        data = {'src': src_img, 'tgt':tgt_img}
        if not self.exclude_cond:
            cond_img = self.loader(os.path.join(self.cond_path, fname))
            # data['cond'] = cond_img
            data.update({'cond' : cond_img})
        
        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis] # HW => HWC
        # image = image.transpose(2, 0, 1) # HWC => CHW


        return data


    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        data = self._load_raw_image(idx)
        return self.trsf(data)


class MultipleLabelDataset(Dataset):
    def __init__(self, root, file_list_path, src_name='source', cond_name='target', tgt_name='condition',
                 depth_name='depth', scribble_name='scribble', seg_name='seg',
                 crop=False, resize_size=None, normalize=True, random_flip=0.5, exclude_cond=True, 
                 loader_type='cv', inte_mode='bicubic'):
        super().__init__()
        # self.root = Path(root)
        # self.src_path = self.root.joinpath(src_name)
        # self.tgt_path = self.root.joinpath(tgt_name)
        # self.cond_path = self.root.joinpath(cond_name)
        self.root = root
        self.src_path = os.path.join(root, src_name)
        self.tgt_path = os.path.join(root, tgt_name)
        self.cond_path = os.path.join(root, cond_name)
        self.depth_path = os.path.join(root, depth_name)
        self.scribble_path = os.path.join(root, scribble_name)
        self.seg_path = os.path.join(root, seg_name)
        self.exclude_cond = exclude_cond

        Image.init()

        self.fnames = []
        with open(file_list_path, 'r') as file:
            for line in file:
                self.fnames.append(line.strip('\n'))

        self.loader = loaders[loader_type]
        self.interpolation = interpolation[inte_mode]

        self.trsf_list = []
        self.trsf_list.append(ToTensor())
        
        if normalize:
            self.normalizer = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.trsf_list.append(self.normalizer)

        self.rgb_trsf = transforms.Compose(self.trsf_list)

        self.flipper = RandomHorizontalFlip(random_flip) if random_flip > 0 else None
        self.resizer = Resize(resize_size, interpolation=self.interpolation)
        self.resizer_and_flipper = transforms.Compose([self.flipper, self.resizer])
        
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, raw_idx):
        fname = self.fnames[raw_idx]
        
        src_img = self.loader(os.path.join(self.src_path, fname))
        tgt_img = self.loader(os.path.join(self.tgt_path, fname))
        depth_img = self.loader(os.path.join(self.tgt_path, fname))
        seg_img = self.loader(os.path.join(self.tgt_path, fname))
        scribble_img = self.loader(os.path.join(self.tgt_path, fname))
        
        data = {'src': src_img, 'tgt':tgt_img, 'depth': depth_img, 'seg': seg_img, 'scribble': scribble_img}
        if not self.exclude_cond:
            cond_img = self.loader(os.path.join(self.cond_path, fname))
            # data['cond'] = cond_img
            data.update({'cond' : cond_img})
        
        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis] # HW => HWC
        # image = image.transpose(2, 0, 1) # HWC => CHW


        return data


    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        data = self._load_raw_image(idx)

        rgb_data = {k: data[k] for k in ['src', 'tgt']}
        rgb_data = self._rgb_trsf(rgb_data)
        data.update(rgb_data)
        
        scribble_data = self._scribble_trsf(data['scribble'])
        data.update(scribble_data)

        depth_data = self._depth_trsf(data['depth'])
        data.update(depth_data)

        seg_data= self._seg_trsf(data['seg'])
        data.update(seg_data)
        
        return self.resizer_and_flipper(data)

    def _rgb_trsf(self, data):
        return self.rgb_trsf(data)
        
    def _scribble_trsf(self, scr):
        scr = self.HWC3(scr)
        scr = self.nms(scr, 127, 3.0)
        scr = cv2.GaussianBlur(scr, (0, 0), 3.0)
        scr[scr > 4] = 255
        scr[scr < 255] = 0
        torch.from_numpy(scr.copy()).float() / 255.0 
        scr = einops.rearrange(scr, 'b h w c -> b c h w').clone()
        return scr

    def _depth_trsf(self, depth):
        depth = self.HWC3(depth)
        depth = torch.from_numpy(depth.copy()).float().cuda() / 255.0
        depth = einops.rearrange(depth, 'b h w c -> b c h w').clone()
        return depth

    def _seg_trsf(self, seg):
        seg = self.HWC3(seg)
        seg = torch.from_numpy(seg.copy()).float().cuda() / 255.0
        seg = einops.rearrange(seg, 'b h w c -> b c h w').clone()
        return seg
    
    def nms(self, x, t, s):
        x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

        f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
        f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

        y = np.zeros_like(x)

        for f in [f1, f2, f3, f4]:
            np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

        z = np.zeros_like(y, dtype=np.uint8)
        z[y > t] = 255
        return z
    
    def HWC3(self, x):
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y