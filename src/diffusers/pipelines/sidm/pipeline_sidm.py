from typing import List, Optional, Tuple, Union

import torch

from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

class SIDMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        cond,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        batch_size, cond_channel, h, w = cond.size()

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size, 
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)


        image = randn_tensor(image_shape, generator=generator)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # predict noise model_output
                model_output = self.unet(image, cond, t).sample

                # compute previous image: x_t -> x_t-1
                image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        image = (image / 2 + 0.5).clamp(0.1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)