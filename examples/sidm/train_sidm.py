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

from omegaconf import OmegaConf

import sidm_coach as coach
from dataset import PairedDataset
from utils import create_model, create_coach, load_state_dict


import sys
sys.path.append('/purestorage/project/tyk/project9/diffusers_mod/src/')
os.chdir('../../')


# Hongs wandb
wandb_key = 'local-d20a4c3fd6cffd419ca148decace4cb95004b226'
wandb_host = 'http://211.168.94.228:8080'


wandb.login(key=wandb_key, host=wandb_host, force=True,)
wandb_logger = WandbLogger(name='test_1', project='ai_service_model', log_model=True)


# Args and Configs
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to checkpoint_dir")
parser.add_argument("--max_epochs", type=int, default=200, help="training epochs")
parser.add_argument("--val_freq", type=int, default=0, help="check validation every n train batches")
parser.add_argument("--config_path", type=str, default=None, help="training model configuration path")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--num_nodes", type=int, default=2, help="num nodes")    
parser.add_argument("--deterministic", action='store_true', help="reproducibility")
parser.add_argument("--resume_path", type=str, default=None, help="resume checkpoint path")
parser.add_argument("--exp_name", type=str, default='myexp', help="exp name")
args = parser.parse_args()

ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)

trainer_cfg = dict(accelerator="gpu", devices=args.gpus, precision=32, 
                     num_nodes=args.num_nodes, strategy=ddp,
                     logger=wandb_logger, max_epochs=args.max_epochs, val_check_interval=args.val_freq)

if args.deterministic:
    seed_everything(1, workers=True)
    trainer_cfg.update({'deterministic': True})

# Dataloader
root='/purestorage/datasets/DGM/iti_pairset'
src_name='origin'
cond_name='spiga'
tgt_name='style012'
train_list_path = root+f'/list/{tgt_name}_train.txt'
val_list_path = root+f'/list/{tgt_name}_val.txt'


train_dataset = PairedDataset(root, train_list_path, src_name, cond_name, tgt_name, resize_size=512, exclude_cond=False)
val_dataset = PairedDataset(root, val_list_path, src_name, cond_name, tgt_name, resize_size=512, exclude_cond=False)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8)

# Model
config = OmegaConf.load(args.config_path)
# Lr scheudler를 위해
config.coach.params.training_config.train_length = len(train_dataloader)
coach_model = create_coach(config).cpu()
if args.resume_path:
    coach_model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))

# Trainer
trainer = pl.Trainer(**trainer_cfg) # callbacks=[logger])
trainer.fit(coach_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)