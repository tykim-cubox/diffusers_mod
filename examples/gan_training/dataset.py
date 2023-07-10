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


class PairedDatasetOld(Dataset):
    def __init__(self, src_path, tgt_path, crop=False, resize_size=None, normalize=True, random_flip=0.5, exclude_cond=True, loader_type='cv', inte_mode='bicubic'):
        super().__init__()
        # self.root = Path(root)
        # self.src_path = self.root.joinpath(src_name)
        # self.tgt_path = self.root.joinpath(tgt_name)
        # self.cond_path = self.root.joinpath(cond_name)
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.exclude_cond = exclude_cond

        Image.init()

        self.src_image_fnames = sorted([f for f in os.listdir(self.src_path) if self._file_ext(f) in Image.EXTENSION])
        self.tgt_image_fnames = sorted([f for f in os.listdir(self.tgt_path) if self._file_ext(f) in Image.EXTENSION])
        

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
        src_fname = self.src_image_fnames[raw_idx]
        src_img = self.loader(os.path.join(self.src_path, src_fname))

        tgt_fname = self.tgt_image_fnames[raw_idx]
        tgt_img = self.loader(os.path.join(self.tgt_path, tgt_fname))

        data = {'src': src_img, 'tgt':tgt_img}

        return data

    def __len__(self):
        return len(self.src_image_fnames)
    
    def __getitem__(self, idx):
        data = self._load_raw_image(idx)
        return self.trsf(data)