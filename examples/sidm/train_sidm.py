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



# Dataloader

config = OmegaConf.load(args.config_path)
# Lr scheudler를 위해
config.coach.params.training_config.train_length = len(train_dataloader)


# Trainer
trainer = pl.Trainer(**trainer_cfg) # callbacks=[logger])
trainer.fit(coach_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)