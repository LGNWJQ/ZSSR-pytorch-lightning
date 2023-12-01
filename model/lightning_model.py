import torch
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torchvision.utils import save_image

import os

from .resnet_model import ZSSR_RES
from  .model import ZSSR_Net


class ZSSR_lightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = ZSSR_RES(config.in_channels, config.channels, config.num_layer)
        self.optimizer = Adam(self.model.parameters(), config.lr)

        self.num_epoch = config.num_epoch
        self.lr = config.lr

        self.val_sr_path = os.path.join('./SR_Result/', config.exp_name)
        if not os.path.exists(self.val_sr_path):
            os.makedirs(self.val_sr_path)
        # 其他
        self.save_hyperparameters(config)
        self.example_input_array = torch.randn(1, 3, 256, 256)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        HR, LR = batch
        HR, LR = HR[0], LR[0]

        r_HR = self.model(HR)
        loss = F.l1_loss(r_HR, HR)

        self.log("loss", loss.item(), prog_bar=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        LR_upscale, LR = batch
        r_HR = self.model(LR_upscale)

        tensorboard = self.logger.experiment
        tensorboard.add_image('LR', LR[0], global_step=0)
        tensorboard.add_image('r_HR', r_HR[0], global_step=self.global_step)

        save_image(r_HR, os.path.join(self.val_sr_path, f'step{self.global_step}.png'))

    def configure_optimizers(self):
        scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epoch,
            eta_min=self.lr / 1e3
        )
        return [self.optimizer], [scheduler]
