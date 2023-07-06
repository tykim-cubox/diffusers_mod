import argparse
import inspect
import math
import os
from copy import deepcopy
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Union, List
import shutil
import numpy as np

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DiffusionPipeline, DDIMPipeline
from diffusers.optimization import get_scheduler
from diffusers.pipelines.ddim.pipeline_ddim import DistilledDDIMPipeline
from diffusers.schedulers.scheduling_ddim import DDIMExtendedScheduler, DistilledDDIMScheduler
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddim import _ddim_scheduler_from_ddpm_scheduler



def main(args):
    DEBUG = False
    if DEBUG:
        args.num_epochs = 20


    pipeline = DiffusionPipeline
    distillsched = [int(x) for x in args.distill_schedule.split(" ")]
    distillsched = list(zip(distillsched[:-1], distillsched[1:]))
    prevtimesteps = None

    totalnumepochs = args.num_epochs
    numphases = len(distillsched)



    for distillphase in range(len(distillsched)):
        if args.use_ema:
            ema_model = EMAModel(studentmodel, inv_gamma=args.ema_inv_gamma, power=args.ema_power, max_value=args.ema_max_decay)
        