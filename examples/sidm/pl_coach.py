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

import sys
sys.path.append('/purestorage/project/tyk/project9/diffusers_mod/src/diffusers')

class DiffusionCoach(pl.LightningModule):
    def __init__(self, model_config, loss_config, training_config, use_ema=False, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        
        self.automatic_optimization = False
        self.load_model()
        self.setup_objectives()
        self.save_hyperparameters()

    def load_model(self):
        self.unet = instantiate_from_config(self.model_config.generator)
        count_params(self.unet, verbose=True)
        size_mb = get_model_size(self.unet)
        self.log(f'model size', size_mb)

        self.enc = 


    def setup_objectives(self):
        self.loss = instantiate_from_config(self.loss_config)

    