import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
import random
from utils.val_utils import compute_psnr_ssim


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.mse_lambda = 0.1
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.l1_loss(restored,clean_patch) + self.mse_lambda * self.mse_loss(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,)
        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        restored = self.net(degrad_patch)
        loss = self.l1_loss(restored, clean_patch) + self.mse_lambda * self.mse_loss(restored, clean_patch)

        restored = torch.clamp(restored, 0.0, 1.0)
        psnr, n = compute_psnr_ssim(restored, clean_patch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_psnr", psnr, on_step=True, on_epoch=True, prog_bar=False)
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)
    logger = TensorBoardLogger(save_dir = opt.log_dir)

    rain_ids = [opt.derain_dir + id_.strip() for id_ in open(opt.data_file_dir + "rainy/rainTrain.txt")]
    snow_ids = [opt.desnow_dir + id_.strip() for id_ in open(opt.data_file_dir + "snowy/snowTrain.txt")]
    
    rng = random.Random(42)
    val_ratio = 0.1

    rain_val_count = int(len(rain_ids) * val_ratio)
    snow_val_count = int(len(snow_ids) * val_ratio)

    rain_ids_val = set(rng.sample(rain_ids, k=rain_val_count))
    rain_ids_train = [p for p in rain_ids if p not in rain_ids_val]
    rain_ids_val = sorted(rain_ids_val)

    snow_ids_val = set(rng.sample(snow_ids, k=snow_val_count))
    snow_ids_train = [p for p in snow_ids if p not in snow_ids_val]
    snow_ids_val = sorted(snow_ids_val)

    trainset = PromptTrainDataset(opt, rain_ids_train, snow_ids_train, 8, "train")
    valset = PromptTrainDataset(opt, rain_ids_val, snow_ids_val, 1, "val")

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    valloader = DataLoader(valset, batch_size=opt.batch_size, pin_memory=True, shuffle=False,
                             drop_last=False, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    progress_bar = TQDMProgressBar(refresh_rate=50)

    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="auto",logger=logger,callbacks=[checkpoint_callback, progress_bar], log_every_n_steps=50,enable_progress_bar=True,)
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()



