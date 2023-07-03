

# import correct text encoder class
text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet)



class Distiller(pl.LightningModule):
    def __init__(self, model_config, loss_config, training_config, distiller_config):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.training_config = training_config
        self.distiller_config = distiller_config

        self.load_model()
        self.setup_objectives()
        self.save_hyperparameters()

    def load_model(self):
        print("load mapping network...")

        # self.teacher = UNet.from_pretrained() # DiffusionPipeline.from_pretrained(args.load_dir)
        # self.student = UNet.from_pretrained()
        pipeline = DiffusionPipeline.from_pretrained(args.load_dir)
        studentmodel = pipeline.unet
        

    def setup_objectives(self):
        ...

        


import argparse
import os
import torch
import wandb

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything
from lightning.pytorch.strategies import DDPStrategy


# Hongs wandb
wandb_key = 'local-d20a4c3fd6cffd419ca148decace4cb95004b226'
wandb_host = 'http://211.168.94.228:8080'


wandb.login(key=wandb_key, host=wandb_host, force=True,)
wandb_logger = WandbLogger(name='test_1', project='ai_service_model', log_model=True)


# Args and Configs
trainer_cfg = dict(accelerator="gpu", devices=args.gpus, precision=32, 
                     num_nodes=args.num_nodes, strategy=ddp,
                     logger=wandb_logger, max_steps=args.max_steps, val_check_interval=args.val_freq)



# Dataloader


# Trainer
trainer = pl.Trainer(**trainer_cfg) # callbacks=[logger])
trainer.fit(coach_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)