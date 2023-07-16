import argparse


import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def parse_args(input_ages=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to checkpoint_dir")
    parser.add_argument("--max_steps", type=int, default=25000, help="training step")
    parser.add_argument("--val_freq", type=int, default=0, help="check validation every n train batches")
    parser.add_argument("--config_path", type=str, default=None, help="training model configuration path")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--num_nodes", type=int, default=2, help="num nodes")    
    parser.add_argument("--deterministic", action='store_true', help="reproducibility")
    parser.add_argument("--resume_path", type=str, default=None, help="resume checkpoint path")
    parser.add_argument("--exp_name", type=str, default='myexp', help="exp name")
    args = parser.parse_args()