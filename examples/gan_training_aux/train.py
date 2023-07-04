import argparse
import os
import torch
from torch.utils.data import DataLoader, Subset, random_split
import wandb

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything
from lightning.pytorch.strategies import DDPStrategy


import coach as coach
from dataset import PairedDataset
from utils import create_model, create_coach, load_state_dict

from omegaconf import OmegaConf


import os
os.chdir('../../')

# Lees wandb
# wandb_key = 'local-e406cacbb94467035e1afb68491cd1c82feed276'
# wandb_host = 'http://211.168.94.174:8080'

# Hongs wandb
wandb_key = 'local-d20a4c3fd6cffd419ca148decace4cb95004b226'
wandb_host = 'http://211.168.94.228:8080'


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to checkpoint_dir")
parser.add_argument("--max_steps", type=int, default=25000, help="training step")
parser.add_argument("--val_freq", type=int, default=0, help="check validation every n train batches")
parser.add_argument("--config_path", type=str, default=None, help="training model configuration path")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--num_nodes", type=int, default=2, help="num nodes")    
parser.add_argument("--deterministic", action='store_true', help="reproducibility")
parser.add_argument("--resume_path", type=str, default=None, help="resume checkpoint path")
parser.add_argument("--exp_name", type=str, default='myexp', help="exp name")
args = parser.parse_args()


wandb.login(key=wandb_key, host=wandb_host, force=True,)
wandb_logger = WandbLogger(name='test_1', project='ai_service_model', log_model=True)

ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
# ddp = DDPStrategy(process_group_backend="nccl")
# ddp = 'ddp'

trainer_cfg = dict(accelerator="gpu", devices=args.gpus, precision=32, 
                     num_nodes=args.num_nodes, strategy=ddp,
                     logger=wandb_logger, max_steps=args.max_steps, val_check_interval=args.val_freq)


if args.deterministic:
    seed_everything(1, workers=True)
    trainer_cfg.update({'deterministic': True})

# Dataset
root='/purestorage/datasets/DGM/iti_pairset'
src_name='origin'
cond_name='spiga'
tgt_name='style012'
train_list_path = root+f'/list/{tgt_name}_train.txt'
val_list_path = root+f'/list/{tgt_name}_val.txt'


train_dataset = PairedDataset(root, train_list_path, src_name, cond_name, tgt_name, resize_size=256)
val_dataset = PairedDataset(root, val_list_path, src_name, cond_name, tgt_name, resize_size=256)
# train_size = int(0.95 * len(dataset))
# val_size = len(dataset) - train_size 
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8)

# dataset = PairedDataset(root, src_name, cond_name, tgt_name, resize_size=256)

# val_indices = [1, 2, 3, 4, 5, 6, 7, 8]
# all_indices = list(range(len(dataset)))
# train_indices = [idx for idx in all_indices if idx not in val_indices]

# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)

# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
# val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8)

# Model
config = OmegaConf.load(args.config_path)
config.coach.params.training_config.scheduler_T0 = len(train_dataloader) * 2
# coach_model = create_coach(args.config_path).cpu()
coach_model = create_coach(config).cpu()
if args.resume_path:
    coach_model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))

# 이거 안되는 듯
# AttributeError: 'function' object has no attribute 'update'
# wandb_logger.experiment.config.update(config)
    
# Trainer
trainer = pl.Trainer(**trainer_cfg) # callbacks=[logger])
trainer.fit(coach_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# python train.py --gpus=8 --config_path=/home/aiteam/tykim/cubox/predefined_iti_train/configs/v2_1.yaml --num_nodes=1 --val_freq=100 --deterministic