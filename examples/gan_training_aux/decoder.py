import torch
import torch.nn as nn

import os
os.chdir('../../')
from src.diffusers.models.unet_2d_blocks import *
from src.diffusers.models.activations import get_activation


class AuxDecoder(nn.Module):
    def __init__(self, 
                 up_block_types = ["UpDecoderBlockTimeless2D", "AttnUpDecoderBlockTimeless2D", "UpDecoderBlockTimeless2D"],
                 block_out_channels = [64, 128, 256, 512],
                 layers_per_block = 2,
                 norm_eps: float = 1e-5,
                 act_fn: str = "silu",
                 norm_num_groups: int = 32,
                 attention_head_dim: Optional[int] = 8,
                 start_spatial_dim = 32,
                 end_spatial_dim = 256,
                 final_outchannel = 3
                 ):
        super().__init__()
        reversed_block_out_channels = list(reversed(block_out_channels)) # [512, 256, 128, 64]
        self.up_blocks = nn.ModuleList([])

        num_upsamples = int(np.log2(end_spatial_dim // start_spatial_dim))
        upsample_flags = [True] * num_upsamples + [False] * (len(up_block_types) - num_upsamples)

        for i, up_block_type in enumerate(up_block_types):
            input_channel = reversed_block_out_channels[i]
            output_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel = None,
                temb_channels = None,
                add_upsample=upsample_flags[i],
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
            )
            self.up_blocks.append(up_block)

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None
            
        conv_out_kernel = 3

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], final_outchannel, kernel_size=conv_out_kernel, padding=conv_out_padding
        )
    def forward(self,
                sample: torch.FloatTensor,
                return_dict: bool = True):
        for upsample_block in self.up_blocks:
            sample = upsample_block(sample)
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return sample
        #return UNet2DConditionOutput(sample=sample)