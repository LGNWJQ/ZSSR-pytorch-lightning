import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from model import ZSSR_lightning
from dataset import Single_Image_dataset, Pari_Image_dataset
from config import set_config


if __name__ == '__main__':
    config = set_config()
    logger = TensorBoardLogger(
        save_dir="./lightning_logs", name=config.exp_name,
        log_graph=True,
    )

    model = ZSSR_lightning(config)
    train_dataset = Pari_Image_dataset(
        image_path=config.image_path,
        sr_factor=config.sr_factor,
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        num_scale=config.num_scale
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=False,
        persistent_workers=True
    )
    val_dataset = Single_Image_dataset(
        image_path=config.image_path,
        sr_factor=config.sr_factor
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=False,
        persistent_workers=True
    )

    tqdm_callback = TQDMProgressBar(refresh_rate=2, process_position=0)
    MS_callback = ModelSummary(max_depth=2)
    callback_list = [tqdm_callback, MS_callback]

    trainer = pl.Trainer(
        max_epochs=config.num_epoch,
        log_every_n_steps=10,
        callbacks=callback_list,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        logger=logger,
        num_sanity_val_steps=2,
        accelerator=config.accelerator,
        devices=1
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )











